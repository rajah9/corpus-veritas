"""
Integration tests for Layer 1: Sequence Numbers & Corpus Evaluator

Tests the SequenceNumber ABC and its concrete subclasses (BatesNumber, EFTANumber),
plus the corpus_evaluator.py reconciliation pipeline.

Real document content is never committed to this repository.
All fixtures use synthetic sequence numbers and placeholder text.

Test structure
--------------
TestBatesNumberScheme     — unit tests for BatesNumber methods
TestEFTANumberScheme      — unit tests for EFTANumber methods
TestBatesReconciliation   — reconciliation logic via BatesNumber
TestEFTAReconciliation    — reconciliation logic via EFTANumber, including DS9 gap handling
TestSequenceSchemeContract — ABC contract tests that both subclasses must satisfy
"""

import pytest
from pipeline.sequence_numbers import (
    BatesNumber,
    EFTANumber,
    ReconciliationResult,
    SequenceNumber,
    COVERAGE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bates():
    return BatesNumber()


@pytest.fixture
def efta():
    """EFTANumber with no mapping file — DS9 gap set is empty."""
    return EFTANumber()


@pytest.fixture
def efta_with_ds9_gaps():
    """EFTANumber with a synthetic DS9 gap set for testing gap_is_expected()."""
    gap_numbers = frozenset(str(n) for n in range(5000, 5500))  # 500 synthetic gaps
    return EFTANumber(ds9_gap_numbers=gap_numbers)


# ---------------------------------------------------------------------------
# TestBatesNumberScheme
# ---------------------------------------------------------------------------

class TestBatesNumberScheme:
    """Unit tests for BatesNumber scheme-specific behaviour."""

    def test_scheme_name(self, bates):
        assert bates.scheme_name == "BATES"

    def test_validate_prefixed_format(self, bates):
        assert bates.validate("DOJ-EPSTEIN-000001")
        assert bates.validate("EDNY-0042318")
        assert bates.validate("DOJ-001234")

    def test_validate_bare_numeric(self, bates):
        assert bates.validate("000001")
        assert bates.validate("123456")

    def test_validate_rejects_short_numbers(self, bates):
        assert not bates.validate("001")     # too short
        assert not bates.validate("12345")   # 5 digits — below 6 minimum

    def test_validate_rejects_plain_text(self, bates):
        assert not bates.validate("EPSTEIN")
        assert not bates.validate("")

    def test_extract_from_text_finds_prefixed(self, bates):
        text = "See document DOJ-EPSTEIN-000042 for details, also EDNY-0009981."
        found = bates.extract_from_text(text)
        assert "DOJ-EPSTEIN-000042" in found
        assert "EDNY-0009981" in found

    def test_extract_from_text_finds_bare_numeric(self, bates):
        text = "Referenced at page 000123 of the filing."
        found = bates.extract_from_text(text)
        assert "000123" in found

    def test_sort_key_numeric_ordering(self, bates):
        numbers = ["DOJ-000010", "DOJ-000002", "DOJ-000100"]
        sorted_nums = sorted(numbers, key=bates.sort_key)
        assert sorted_nums == ["DOJ-000002", "DOJ-000010", "DOJ-000100"]

    def test_gap_is_never_expected(self, bates):
        """All gaps in a Bates sequence are suspicious — no exceptions."""
        assert not bates.gap_is_expected("DOJ-000001")
        assert not bates.gap_is_expected("DOJ-999999")
        assert not bates.gap_is_expected("000001")

    def test_describe_number_contains_value(self, bates):
        desc = bates.describe_number("DOJ-000042")
        assert "DOJ-000042" in desc


# ---------------------------------------------------------------------------
# TestEFTANumberScheme
# ---------------------------------------------------------------------------

class TestEFTANumberScheme:
    """Unit tests for EFTANumber scheme-specific behaviour."""

    def test_scheme_name(self, efta):
        assert efta.scheme_name == "EFTA"

    def test_validate_positive_integer_string(self, efta):
        assert efta.validate("1")
        assert efta.validate("123456")
        assert efta.validate("2731785")   # max known EFTA in corpus

    def test_validate_rejects_zero(self, efta):
        assert not efta.validate("0")

    def test_validate_rejects_non_numeric(self, efta):
        assert not efta.validate("EFTA-001")  # canonical form is numeric only
        assert not efta.validate("abc")
        assert not efta.validate("")
        assert not efta.validate("1.5")

    def test_extract_finds_prefixed_format(self, efta):
        text = "Document EFTA-000123 references EFTA_000456 and EFTA789."
        found = efta.extract_from_text(text)
        assert "000123" in found
        assert "000456" in found
        assert "789" in found

    def test_extract_is_case_insensitive(self, efta):
        text = "See efta-001234 and EFTA-005678"
        found = efta.extract_from_text(text)
        assert "001234" in found
        assert "005678" in found

    def test_sort_key_is_integer(self, efta):
        assert efta.sort_key("42") == 42
        assert efta.sort_key("1000000") == 1000000

    def test_sort_key_orders_numerically(self, efta):
        numbers = ["10", "2", "100", "21"]
        sorted_nums = sorted(numbers, key=efta.sort_key)
        assert sorted_nums == ["2", "10", "21", "100"]

    def test_gap_is_expected_returns_false_without_mapping(self, efta):
        """Without DS9 gap data loaded, no gap is expected."""
        assert not efta.gap_is_expected("5001")
        assert not efta.gap_is_expected("999999")

    def test_gap_is_expected_with_ds9_gaps(self, efta_with_ds9_gaps):
        """DS9 gaps loaded from synthetic set are recognised as expected."""
        assert efta_with_ds9_gaps.gap_is_expected("5001")   # in gap range
        assert efta_with_ds9_gaps.gap_is_expected("5499")   # in gap range

    def test_gap_is_not_expected_outside_ds9(self, efta_with_ds9_gaps):
        assert not efta_with_ds9_gaps.gap_is_expected("4999")   # just below gap range
        assert not efta_with_ds9_gaps.gap_is_expected("5500")   # just above gap range
        assert not efta_with_ds9_gaps.gap_is_expected("1")

    def test_ds9_gap_count_reflects_loaded_set(self, efta_with_ds9_gaps):
        assert efta_with_ds9_gaps.ds9_gap_count == 500

    def test_ds9_gap_count_zero_without_mapping(self, efta):
        assert efta.ds9_gap_count == 0

    def test_describe_number_contains_efta_id(self, efta):
        desc = efta.describe_number("42000")
        assert "42000" in desc

    def test_describe_number_invalid_non_numeric(self, efta):
        desc = efta.describe_number("NOTANUMBER")
        assert "NOTANUMBER" in desc


# ---------------------------------------------------------------------------
# TestBatesReconciliation
# ---------------------------------------------------------------------------

class TestBatesReconciliation:
    """Reconciliation logic exercised through BatesNumber."""

    def test_full_coverage(self, bates):
        index = ["DOJ-001", "DOJ-002", "DOJ-003"]
        corpus = ["DOJ-001", "DOJ-002", "DOJ-003"]
        result = bates.reconcile(corpus, index)
        assert result.present_count == 3
        assert result.missing_from_corpus_count == 0
        assert result.coverage_pct == 1.0
        assert not result.partial_coverage
        assert result.sequence_type == "BATES"

    def test_missing_documents_become_deletion_candidates(self, bates):
        index = ["DOJ-001", "DOJ-002", "DOJ-003", "DOJ-004", "DOJ-005"]
        corpus = ["DOJ-001", "DOJ-002"]
        result = bates.reconcile(corpus, index)
        assert result.missing_from_corpus_count == 3
        assert "DOJ-003" in result.deletion_candidates
        assert "DOJ-004" in result.deletion_candidates
        # Bates has no expected gaps — all missing are deletion candidates
        assert result.expected_gap_numbers == []
        assert len(result.deletion_candidates) == 3

    def test_at_coverage_threshold_not_partial(self, bates):
        index = [f"DOJ-{i:03}" for i in range(100)]
        corpus = [f"DOJ-{i:03}" for i in range(60)]  # exactly 60%
        result = bates.reconcile(corpus, index)
        assert not result.partial_coverage

    def test_below_threshold_is_partial(self, bates):
        index = [f"DOJ-{i:03}" for i in range(100)]
        corpus = [f"DOJ-{i:03}" for i in range(59)]  # 59%
        result = bates.reconcile(corpus, index)
        assert result.partial_coverage

    def test_unindexed_documents_counted(self, bates):
        index = ["DOJ-001", "DOJ-002"]
        corpus = ["DOJ-001", "DOJ-002", "DOJ-999"]
        result = bates.reconcile(corpus, index)
        assert result.unindexed_count == 1

    def test_empty_corpus(self, bates):
        index = ["DOJ-001", "DOJ-002", "DOJ-003"]
        result = bates.reconcile([], index)
        assert result.coverage_pct == 0.0
        assert result.partial_coverage
        assert result.missing_from_corpus_count == 3
        assert len(result.deletion_candidates) == 3

    def test_empty_index_gives_zero_coverage(self, bates):
        result = bates.reconcile(["DOJ-001"], [])
        assert result.coverage_pct == 0.0
        assert result.unindexed_count == 1


# ---------------------------------------------------------------------------
# TestEFTAReconciliation
# ---------------------------------------------------------------------------

class TestEFTAReconciliation:
    """
    Reconciliation logic exercised through EFTANumber.

    The critical EFTA-specific behaviour: DS9 gap numbers must be separated
    from true deletion candidates. A gap in the EFTA sequence is only a
    deletion candidate if it is NOT in the documented DS9 gap set.
    """

    def test_full_coverage(self, efta):
        index = ["1", "2", "3", "4", "5"]
        corpus = ["1", "2", "3", "4", "5"]
        result = efta.reconcile(corpus, index)
        assert result.present_count == 5
        assert result.missing_from_corpus_count == 0
        assert result.coverage_pct == 1.0
        assert result.sequence_type == "EFTA"

    def test_missing_numbers_without_ds9_gaps_are_deletion_candidates(self, efta):
        """Without DS9 gap data, all missing EFTA numbers are deletion candidates."""
        index = [str(n) for n in range(1, 11)]   # 1–10
        corpus = [str(n) for n in range(1, 8)]   # 1–7, missing 8,9,10
        result = efta.reconcile(corpus, index)
        assert result.missing_from_corpus_count == 3
        assert result.expected_gap_numbers == []
        assert sorted(result.deletion_candidates, key=int) == ["10", "8", "9"] or \
               sorted(result.deletion_candidates, key=int) == ["8", "9", "10"]

    def test_ds9_gaps_are_expected_not_deletion_candidates(self, efta_with_ds9_gaps):
        """
        DS9 gap numbers (5000–5499 in fixture) must appear in expected_gap_numbers,
        NOT in deletion_candidates.

        This is the core EFTA business rule. Failing this test means the system
        would generate thousands of false deletion alarms from documented DS9 gaps.
        """
        # Index includes both a real gap (4999) and DS9 gaps (5001–5003)
        index = [str(n) for n in range(4995, 5010)]
        # Corpus is missing 4999 (real gap) and 5001, 5002, 5003 (DS9 gaps)
        corpus = [str(n) for n in range(4995, 5010)
                  if n not in (4999, 5001, 5002, 5003)]

        result = efta_with_ds9_gaps.reconcile(corpus, index)

        # DS9 gaps (5001-5003) go to expected_gap_numbers
        assert "5001" in result.expected_gap_numbers
        assert "5002" in result.expected_gap_numbers
        assert "5003" in result.expected_gap_numbers

        # Real gap (4999) goes to deletion_candidates
        assert "4999" in result.deletion_candidates

        # DS9 gaps must NOT be in deletion_candidates
        assert "5001" not in result.deletion_candidates
        assert "5002" not in result.deletion_candidates
        assert "5003" not in result.deletion_candidates

    def test_expected_gaps_count_toward_missing_not_coverage(self, efta_with_ds9_gaps):
        """
        Expected gaps still count toward missing_from_corpus_count (they ARE absent).
        They should NOT drag down deletion_candidates.
        Coverage is computed on present/index, so expected gaps reduce coverage.
        """
        index = [str(n) for n in range(5000, 5010)]
        # All DS9 gap numbers (5000–5499) — corpus has none of them
        corpus = []

        result = efta_with_ds9_gaps.reconcile(corpus, index)
        assert result.missing_from_corpus_count == 10
        # All 10 are in the DS9 gap range (5000-5499)
        assert len(result.expected_gap_numbers) == 10
        assert result.deletion_candidates == []

    def test_sort_order_is_numeric(self, efta):
        """EFTA numbers sort numerically, not lexicographically."""
        index = ["1", "2", "10", "20", "100"]
        corpus = ["2", "20"]
        result = efta.reconcile(corpus, index)
        # Missing: 1, 10, 100 — should sort as 1, 10, 100 not 1, 10, 100 lexicographically
        assert result.deletion_candidates == ["1", "10", "100"]

    def test_large_realistic_scale(self, efta):
        """Reconciliation handles corpus-scale numbers without performance issues."""
        n = 10_000
        index = [str(i) for i in range(1, n + 1)]
        # Simulate 5% gap rate
        corpus = [str(i) for i in range(1, n + 1) if i % 20 != 0]
        result = efta.reconcile(corpus, index)
        assert result.missing_from_corpus_count == n // 20
        assert len(result.deletion_candidates) == n // 20

    def test_empty_corpus(self, efta):
        index = ["1", "2", "3"]
        result = efta.reconcile([], index)
        assert result.coverage_pct == 0.0
        assert result.partial_coverage
        assert result.missing_from_corpus_count == 3


# ---------------------------------------------------------------------------
# TestSequenceSchemeContract
# ---------------------------------------------------------------------------

class TestSequenceSchemeContract:
    """
    Contract tests that both BatesNumber and EFTANumber must satisfy.
    Parameterised so any new SequenceNumber subclass can be added here.

    These tests verify that the ABC contract is correctly implemented —
    not the scheme-specific behaviour, but the shared interface.
    """

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_scheme_name_is_string(self, request, scheme_fixture):
        scheme = request.getfixturevalue(scheme_fixture)
        assert isinstance(scheme.scheme_name, str)
        assert len(scheme.scheme_name) > 0

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_reconcile_returns_reconciliation_result(self, request, scheme_fixture):
        scheme = request.getfixturevalue(scheme_fixture)
        result = scheme.reconcile([], [])
        assert isinstance(result, ReconciliationResult)

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_reconcile_result_sequence_type_matches_scheme(self, request, scheme_fixture):
        scheme = request.getfixturevalue(scheme_fixture)
        result = scheme.reconcile([], [])
        assert result.sequence_type == scheme.scheme_name

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_deletion_candidates_subset_of_missing(self, request, scheme_fixture):
        """deletion_candidates must always be a subset of missing_numbers."""
        scheme = request.getfixturevalue(scheme_fixture)
        index = ["1", "2", "3", "4", "5"] if scheme_fixture == "efta" \
            else ["DOJ-001", "DOJ-002", "DOJ-003", "DOJ-004", "DOJ-005"]
        corpus = index[:2]
        result = scheme.reconcile(corpus, index)
        missing_set = set(result.missing_numbers)
        for candidate in result.deletion_candidates:
            assert candidate in missing_set, (
                f"{candidate} in deletion_candidates but not in missing_numbers"
            )

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_expected_gaps_plus_deletion_candidates_equals_missing(
        self, request, scheme_fixture
    ):
        """expected_gap_numbers + deletion_candidates must equal missing_numbers exactly."""
        scheme = request.getfixturevalue(scheme_fixture)
        index = ["1", "2", "3", "4", "5"] if scheme_fixture == "efta" \
            else ["DOJ-001", "DOJ-002", "DOJ-003", "DOJ-004", "DOJ-005"]
        corpus = index[:2]
        result = scheme.reconcile(corpus, index)
        reconstructed = set(result.expected_gap_numbers) | set(result.deletion_candidates)
        assert reconstructed == set(result.missing_numbers)

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_coverage_pct_between_zero_and_one(self, request, scheme_fixture):
        scheme = request.getfixturevalue(scheme_fixture)
        index = ["1", "2", "3"] if scheme_fixture == "efta" \
            else ["DOJ-001", "DOJ-002", "DOJ-003"]
        corpus = index[:2]
        result = scheme.reconcile(corpus, index)
        assert 0.0 <= result.coverage_pct <= 1.0

    @pytest.mark.parametrize("scheme_fixture", ["bates", "efta"])
    def test_gap_is_expected_returns_bool(self, request, scheme_fixture):
        scheme = request.getfixturevalue(scheme_fixture)
        value = "1" if scheme_fixture == "efta" else "DOJ-001"
        assert isinstance(scheme.gap_is_expected(value), bool)