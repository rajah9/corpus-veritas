"""
Integration tests for deletion_detector.py and models.py

Tests cover:
  TestDeletionFlag          — enum values, ordering, and properties
  TestDetectionSignals      — signal counting and flag derivation
  TestWithholdingRecord     — model validation, serialisation, lifecycle
  TestDeletionRecord        — evidence-graded record creation
  TestFBI302SeriesCheck     — selective withholding detection
  TestFactoryFunctions      — create_acknowledged_withholding and create_deletion_finding
  TestWSJScenario           — end-to-end fixture modelling the March 2026 WSJ findings
"""

import pytest
from pipeline.models import DeletionFlag, DocumentState, WithholdingRecord
from pipeline.deletion_detector import (
    DetectionSignals,
    FBI302SeriesResult,
    DeletionRecord,
    check_302_series,
    create_acknowledged_withholding,
    create_deletion_finding,
)


# ---------------------------------------------------------------------------
# TestDeletionFlag
# ---------------------------------------------------------------------------

class TestDeletionFlag:

    def test_all_values_exist(self):
        expected = {
            "DELETION_CONFIRMED", "DELETION_SUSPECTED", "DELETION_POSSIBLE",
            "REFERENCE_UNRESOLVED", "WITHHELD_ACKNOWLEDGED", "WITHHELD_SELECTIVELY",
        }
        assert {f.value for f in DeletionFlag} == expected

    def test_ordering_weakest_to_strongest(self):
        ordered = [
            DeletionFlag.REFERENCE_UNRESOLVED,
            DeletionFlag.DELETION_POSSIBLE,
            DeletionFlag.DELETION_SUSPECTED,
            DeletionFlag.DELETION_CONFIRMED,
            DeletionFlag.WITHHELD_SELECTIVELY,
            DeletionFlag.WITHHELD_ACKNOWLEDGED,
        ]
        for i in range(len(ordered) - 1):
            assert ordered[i] < ordered[i + 1], (
                f"Expected {ordered[i]} < {ordered[i+1]}"
            )

    def test_ordering_greater_than(self):
        assert DeletionFlag.WITHHELD_ACKNOWLEDGED > DeletionFlag.DELETION_CONFIRMED
        assert DeletionFlag.DELETION_CONFIRMED > DeletionFlag.DELETION_SUSPECTED

    def test_ordering_less_than_or_equal(self):
        assert DeletionFlag.DELETION_POSSIBLE <= DeletionFlag.DELETION_POSSIBLE
        assert DeletionFlag.DELETION_POSSIBLE <= DeletionFlag.DELETION_SUSPECTED

    def test_confidence_tier_confirmed_flags(self):
        assert DeletionFlag.DELETION_CONFIRMED.confidence_tier == "CONFIRMED"
        assert DeletionFlag.WITHHELD_ACKNOWLEDGED.confidence_tier == "CONFIRMED"
        assert DeletionFlag.WITHHELD_SELECTIVELY.confidence_tier == "CONFIRMED"

    def test_confidence_tier_lower_flags(self):
        assert DeletionFlag.DELETION_SUSPECTED.confidence_tier == "CORROBORATED"
        assert DeletionFlag.DELETION_POSSIBLE.confidence_tier == "SINGLE_SOURCE"
        assert DeletionFlag.REFERENCE_UNRESOLVED.confidence_tier == "SPECULATIVE"

    def test_requires_human_review(self):
        assert DeletionFlag.DELETION_POSSIBLE.requires_human_review
        assert DeletionFlag.REFERENCE_UNRESOLVED.requires_human_review
        assert not DeletionFlag.DELETION_CONFIRMED.requires_human_review
        assert not DeletionFlag.WITHHELD_ACKNOWLEDGED.requires_human_review

    def test_is_government_acknowledged(self):
        assert DeletionFlag.WITHHELD_ACKNOWLEDGED.is_government_acknowledged
        assert DeletionFlag.WITHHELD_SELECTIVELY.is_government_acknowledged
        assert not DeletionFlag.DELETION_CONFIRMED.is_government_acknowledged
        assert not DeletionFlag.DELETION_SUSPECTED.is_government_acknowledged

    def test_string_value_matches_name(self):
        """DeletionFlag inherits from str — value must equal the string."""
        assert DeletionFlag.DELETION_CONFIRMED == "DELETION_CONFIRMED"
        assert DeletionFlag.WITHHELD_ACKNOWLEDGED == "WITHHELD_ACKNOWLEDGED"

    def test_sort_by_severity(self):
        flags = [
            DeletionFlag.WITHHELD_ACKNOWLEDGED,
            DeletionFlag.DELETION_POSSIBLE,
            DeletionFlag.DELETION_CONFIRMED,
            DeletionFlag.REFERENCE_UNRESOLVED,
        ]
        sorted_flags = sorted(flags)
        assert sorted_flags[0] == DeletionFlag.REFERENCE_UNRESOLVED
        assert sorted_flags[-1] == DeletionFlag.WITHHELD_ACKNOWLEDGED


# ---------------------------------------------------------------------------
# TestDetectionSignals
# ---------------------------------------------------------------------------

class TestDetectionSignals:

    def test_signal_count_zero(self):
        s = DetectionSignals()
        assert s.signal_count == 0

    def test_signal_count_one(self):
        s = DetectionSignals(efta_gap=True)
        assert s.signal_count == 1

    def test_signal_count_two(self):
        s = DetectionSignals(efta_gap=True, discovery_log_entry=True)
        assert s.signal_count == 2

    def test_signal_count_three(self):
        s = DetectionSignals(efta_gap=True, discovery_log_entry=True, document_stamp_gap=True)
        assert s.signal_count == 3

    def test_derived_flag_one_signal_is_possible(self):
        assert DetectionSignals(efta_gap=True).derived_flag == DeletionFlag.DELETION_POSSIBLE

    def test_derived_flag_two_signals_is_suspected(self):
        s = DetectionSignals(efta_gap=True, discovery_log_entry=True)
        assert s.derived_flag == DeletionFlag.DELETION_SUSPECTED

    def test_derived_flag_three_signals_is_confirmed(self):
        s = DetectionSignals(efta_gap=True, discovery_log_entry=True, document_stamp_gap=True)
        assert s.derived_flag == DeletionFlag.DELETION_CONFIRMED

    def test_derived_flag_zero_signals_raises(self):
        with pytest.raises(ValueError, match="zero signals"):
            DetectionSignals().derived_flag

    def test_any_single_signal_gives_possible(self):
        """Each individual signal alone produces DELETION_POSSIBLE."""
        for kwargs in [
            {"efta_gap": True},
            {"discovery_log_entry": True},
            {"document_stamp_gap": True},
        ]:
            s = DetectionSignals(**kwargs)
            assert s.derived_flag == DeletionFlag.DELETION_POSSIBLE


# ---------------------------------------------------------------------------
# TestWithholdingRecord
# ---------------------------------------------------------------------------

class TestWithholdingRecord:

    def _make_acknowledged(self, **kwargs) -> WithholdingRecord:
        defaults = dict(
            record_id="test-001",
            document_identifiers=["DOC-001"],
            deletion_flag=DeletionFlag.WITHHELD_ACKNOWLEDGED,
            acknowledgment_source="Test source",
            acknowledgment_date="2026-03-01",
        )
        defaults.update(kwargs)
        return WithholdingRecord(**defaults)

    def test_valid_acknowledged_record(self):
        r = self._make_acknowledged()
        assert r.deletion_flag == DeletionFlag.WITHHELD_ACKNOWLEDGED
        assert r.document_count == 1

    def test_valid_selective_record(self):
        r = self._make_acknowledged(
            deletion_flag=DeletionFlag.WITHHELD_SELECTIVELY,
            sibling_document_ids=["SIBLING-001"],
        )
        assert r.deletion_flag == DeletionFlag.WITHHELD_SELECTIVELY

    def test_rejects_evidence_graded_flag(self):
        """WithholdingRecord must not accept evidence-graded flags."""
        with pytest.raises(ValueError, match="WITHHELD_ACKNOWLEDGED or WITHHELD_SELECTIVELY"):
            self._make_acknowledged(deletion_flag=DeletionFlag.DELETION_CONFIRMED)

    def test_rejects_empty_document_identifiers(self):
        with pytest.raises(ValueError, match="at least one document_identifier"):
            self._make_acknowledged(document_identifiers=[])

    def test_rejects_released_without_date(self):
        with pytest.raises(ValueError, match="release_date"):
            self._make_acknowledged(released=True, release_date=None)

    def test_mark_released(self):
        r = self._make_acknowledged()
        assert not r.released
        r.mark_released("2026-03-07")
        assert r.released
        assert r.release_date == "2026-03-07"

    def test_is_overdue_no_expected_date(self):
        r = self._make_acknowledged()
        assert not r.is_overdue

    def test_is_overdue_past_date(self):
        r = self._make_acknowledged(expected_release_date="2020-01-01")
        assert r.is_overdue

    def test_is_overdue_future_date(self):
        r = self._make_acknowledged(expected_release_date="2099-12-31")
        assert not r.is_overdue

    def test_is_overdue_already_released(self):
        r = self._make_acknowledged(
            expected_release_date="2020-01-01",
            released=True,
            release_date="2020-01-02",
        )
        assert not r.is_overdue

    def test_round_trip_serialisation(self):
        r = self._make_acknowledged(
            stated_reason="Review required",
            sibling_document_ids=["SIBLING-001"],
            subject_entities=[{"type": "PERSON", "name": "Test Subject"}],
            notes="Test note",
        )
        d = r.to_dict()
        r2 = WithholdingRecord.from_dict(d)
        assert r2.record_id == r.record_id
        assert r2.deletion_flag == r.deletion_flag
        assert r2.sibling_document_ids == r.sibling_document_ids
        assert r2.subject_entities == r.subject_entities

    def test_document_count(self):
        r = self._make_acknowledged(
            document_identifiers=["DOC-001", "DOC-002", "DOC-003"]
        )
        assert r.document_count == 3


# ---------------------------------------------------------------------------
# TestFBI302SeriesCheck
# ---------------------------------------------------------------------------

class TestFBI302SeriesCheck:

    def test_fully_released_series_not_selective(self):
        result = check_302_series(
            series_identifier="FBI-302-SERIES-001",
            all_series_ids=["302-A", "302-B", "302-C"],
            released_ids=["302-A", "302-B", "302-C"],
            total_expected=3,
        )
        assert not result.is_selective
        assert result.withheld_ids == []

    def test_fully_withheld_series_not_selective(self):
        """All withheld with none released is not *selective* — it's total withholding."""
        result = check_302_series(
            series_identifier="FBI-302-SERIES-002",
            all_series_ids=["302-A", "302-B"],
            released_ids=[],
        )
        assert not result.is_selective
        assert len(result.withheld_ids) == 2

    def test_partial_release_is_selective(self):
        """The core test: one released, others withheld = WITHHELD_SELECTIVELY."""
        result = check_302_series(
            series_identifier="FBI-302-TRUMP-SERIES",
            all_series_ids=["302-001", "302-002", "302-003", "302-004"],
            released_ids=["302-001"],
            total_expected=4,
        )
        assert result.is_selective
        assert len(result.released_ids) == 1
        assert len(result.withheld_ids) == 3
        assert "302-002" in result.withheld_ids
        assert "302-001" not in result.withheld_ids

    def test_release_rate_with_known_total(self):
        result = check_302_series(
            series_identifier="FBI-302-SERIES-003",
            all_series_ids=["302-A", "302-B", "302-C", "302-D"],
            released_ids=["302-A", "302-B"],
            total_expected=4,
        )
        assert result.release_rate == 0.5

    def test_release_rate_none_without_total(self):
        result = check_302_series(
            series_identifier="FBI-302-SERIES-004",
            all_series_ids=["302-A", "302-B"],
            released_ids=["302-A"],
        )
        assert result.release_rate is None


# ---------------------------------------------------------------------------
# TestFactoryFunctions
# ---------------------------------------------------------------------------

class TestFactoryFunctions:

    def test_create_acknowledged_no_siblings_gives_acknowledged_flag(self):
        r = create_acknowledged_withholding(
            document_identifiers=["DOC-001"],
            acknowledgment_source="Test",
            acknowledgment_date="2026-03-01",
        )
        assert r.deletion_flag == DeletionFlag.WITHHELD_ACKNOWLEDGED
        assert isinstance(r, WithholdingRecord)

    def test_create_acknowledged_with_siblings_gives_selective_flag(self):
        r = create_acknowledged_withholding(
            document_identifiers=["DOC-002", "DOC-003"],
            acknowledgment_source="Test",
            acknowledgment_date="2026-03-01",
            sibling_document_ids=["DOC-001"],
        )
        assert r.deletion_flag == DeletionFlag.WITHHELD_SELECTIVELY

    def test_create_acknowledged_assigns_uuid(self):
        r = create_acknowledged_withholding(
            document_identifiers=["DOC-001"],
            acknowledgment_source="Test",
            acknowledgment_date="2026-03-01",
        )
        assert len(r.record_id) == 36  # UUID4 format

    def test_create_deletion_finding_three_signals(self):
        signals = DetectionSignals(efta_gap=True, discovery_log_entry=True, document_stamp_gap=True)
        r = create_deletion_finding(
            document_identifiers=["EFTA-001234"],
            signals=signals,
            acknowledgment_source="EFTA index reconciliation",
            acknowledgment_date="2026-03-01",
        )
        assert r.deletion_flag == DeletionFlag.DELETION_CONFIRMED
        assert isinstance(r, DeletionRecord)

    def test_create_deletion_finding_one_signal(self):
        signals = DetectionSignals(efta_gap=True)
        r = create_deletion_finding(
            document_identifiers=["EFTA-005678"],
            signals=signals,
            acknowledgment_source="EFTA index reconciliation",
            acknowledgment_date="2026-03-01",
        )
        assert r.deletion_flag == DeletionFlag.DELETION_POSSIBLE


# ---------------------------------------------------------------------------
# TestWSJScenario
# ---------------------------------------------------------------------------

class TestWSJScenario:
    """
    End-to-end fixture modelling the March 2026 WSJ findings.

    Two findings:
    1. 47,635 files confirmed offline by DOJ → WITHHELD_ACKNOWLEDGED
    2. Trump 302 series: one released, three withheld → WITHHELD_SELECTIVELY

    This test ensures the full data model correctly captures the scenario
    that prompted the addition of WITHHELD_* flags to the taxonomy.

    Constitution reference:
    Hard Limit 3 — DOJ characterisation of Trump 302s as 'baseless' is
    the DOJ's stated position. The system records it as such, not as fact.
    Hard Limit 1 — subject_entities here contain a public figure (Trump),
    not a victim. Victim-adjacent entities must be separately flagged.
    """

    def test_bulk_offline_withholding(self):
        """47,635 DOJ files confirmed offline — WITHHELD_ACKNOWLEDGED."""
        r = create_acknowledged_withholding(
            document_identifiers=[f"DOJ-OFFLINE-{i:06}" for i in range(100)],  # sample
            acknowledgment_source="DOJ statement to Wall Street Journal, March 2026",
            acknowledgment_date="2026-03-01",
            stated_reason="Additional review required prior to public release",
            expected_release_date="2026-03-07",
            notes="DOJ confirmed 47,635 files offline after WSJ inquiry. "
                  "Exact EFTA identifiers to be mapped when DOJ releases manifest.",
        )
        assert r.deletion_flag == DeletionFlag.WITHHELD_ACKNOWLEDGED
        assert r.deletion_flag.is_government_acknowledged
        assert r.deletion_flag.confidence_tier == "CONFIRMED"
        assert r.expected_release_date == "2026-03-07"
        # The DOJ stated the files would be released by end of the week of March 7, 2026.
        # That date has now passed without release — the withholding is correctly overdue.
        # This assertion documents a real-world fact, not a test defect.
        assert r.is_overdue

    def test_future_release_date_not_overdue(self):
        """A withholding with a genuinely future deadline is not yet overdue."""
        r = create_acknowledged_withholding(
            document_identifiers=["DOJ-OFFLINE-FUTURE"],
            acknowledgment_source="Hypothetical future withholding",
            acknowledgment_date="2026-03-01",
            expected_release_date="2099-12-31",
        )
        assert not r.is_overdue

    def test_trump_302_selective_withholding(self):
        """
        Four-interview series: one 302 released (Epstein conduct),
        three withheld (Trump-related allegations).

        The DOJ's characterisation of the claims as baseless is recorded
        as a stated_reason — not adopted as a system finding.
        """
        series = check_302_series(
            series_identifier="FBI-302-TRUMP-ALLEGATIONS-2019",
            all_series_ids=["302-EPSTEIN-CONDUCT", "302-TRUMP-001",
                            "302-TRUMP-002", "302-TRUMP-003"],
            released_ids=["302-EPSTEIN-CONDUCT"],
            total_expected=4,
            notes="Female subject alleged sexual misconduct by Trump across four interviews.",
        )
        assert series.is_selective
        assert series.release_rate == 0.25
        assert len(series.withheld_ids) == 3

        # Create the WithholdingRecord for the withheld 302s
        r = create_acknowledged_withholding(
            document_identifiers=series.withheld_ids,
            acknowledgment_source="DOJ statement to Wall Street Journal, March 2026",
            acknowledgment_date="2026-03-01",
            stated_reason=(
                "DOJ described the underlying claims as unverified and baseless. "
                "NOTE: This is the DOJ's stated characterisation, not a system finding."
            ),
            sibling_document_ids=series.released_ids,
            subject_entities=[
                {
                    "type": "PERSON",
                    "name": "Donald J. Trump",
                    "role": "SUBJECT_OF_ALLEGATIONS",
                    "victim_flag": False,
                    "note": "Public figure. DOJ states claims are baseless — "
                            "recorded as stated position only.",
                },
            ],
            notes="Three of four interview 302s withheld. "
                  "Epstein-conduct 302 released. Pattern is WITHHELD_SELECTIVELY.",
        )

        assert r.deletion_flag == DeletionFlag.WITHHELD_SELECTIVELY
        assert r.deletion_flag > DeletionFlag.DELETION_CONFIRMED
        assert r.deletion_flag.is_government_acknowledged
        assert len(r.sibling_document_ids) == 1
        assert r.sibling_document_ids[0] == "302-EPSTEIN-CONDUCT"
        assert len(r.subject_entities) == 1
        assert r.subject_entities[0]["name"] == "Donald J. Trump"

    def test_wsj_findings_have_higher_confidence_than_efta_gap_alone(self):
        """
        Government-acknowledged flags rank above evidence-graded flags.
        A withheld document whose existence is confirmed by the DOJ itself
        is more strongly evidenced than a gap detected by index reconciliation.
        """
        efta_gap = DeletionFlag.DELETION_CONFIRMED  # strongest evidence-graded flag
        wsj_finding = DeletionFlag.WITHHELD_ACKNOWLEDGED

        assert wsj_finding > efta_gap
        assert wsj_finding.confidence_tier == efta_gap.confidence_tier == "CONFIRMED"
        # Both are CONFIRMED tier — but government acknowledgment ranks higher
        # in the ordering, making it more defensible in reporting.