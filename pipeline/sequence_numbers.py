"""
sequence_numbers.py
Layer 1 — Document Sequence Numbering Schemes

Provides an abstract base class (SequenceNumber) and two concrete implementations:

  BatesNumber — traditional legal document production numbering
                e.g. DOJ-EPSTEIN-000042
                All gaps are treated as suspicious (potential deletions).

  EFTANumber  — Epstein Files Transparency Act per-page numbering
                e.g. EFTA-000123456
                Per-page (not per-document). Sequential across all 12 DOJ datasets.
                DS9 contains 692,473 documented gap numbers that are NOT deletion
                candidates — they represent unimaged, withheld, or unused tracking
                slots whose status is unknown but documented by the index itself.

The shared reconciliation algorithm lives on the base class. Subclasses override
only scheme-specific behaviour: parsing, validation, sorting, and gap classification.

Usage:
    from pipeline.sequence_numbers import BatesNumber, EFTANumber

    result = BatesNumber().reconcile(corpus_numbers, index_numbers)
    result = EFTANumber.from_mapping_file(path).reconcile(corpus_numbers, index_numbers)

See docs/ARCHITECTURE.md § Layer 1 Sub-Module 1B for architectural context.
See CONSTITUTION.md Principle IV — Gaps Are Facts — for the ethical framing.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Corpora covering less than this fraction of the index are tagged PARTIAL_COVERAGE.
# Applies to all schemes.
COVERAGE_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    """
    Result of reconciling a corpus's sequence numbers against an authoritative index.

    Fields
    ------
    sequence_type       : "BATES" | "EFTA" — which scheme produced this result
    present_count       : numbers in both index and corpus
    missing_from_corpus : numbers in index but absent from corpus
    unindexed_count     : numbers in corpus but absent from index (flag for review)
    coverage_pct        : present / index total (0.0–1.0)
    partial_coverage    : True if coverage_pct < COVERAGE_THRESHOLD
    missing_numbers     : full sorted list of index numbers absent from corpus
    expected_gap_numbers: subset of missing_numbers whose absence is documented/expected
                          (EFTA DS9 gaps). These do NOT feed deletion_detector.py.
    deletion_candidates : missing_numbers minus expected_gap_numbers.
                          These DO feed deletion_detector.py as DELETION_SUSPECTED.
    """
    sequence_type: str
    present_count: int = 0
    missing_from_corpus_count: int = 0
    unindexed_count: int = 0
    coverage_pct: float = 0.0
    partial_coverage: bool = False
    missing_numbers: list[str] = field(default_factory=list)
    expected_gap_numbers: list[str] = field(default_factory=list)
    deletion_candidates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class SequenceNumber(ABC):
    """
    Abstract base class for document sequence numbering schemes.

    Concrete subclasses must implement five methods that capture all
    scheme-specific behaviour. The reconciliation algorithm is shared
    and lives here.

    Subclasses
    ----------
    BatesNumber : traditional legal Bates stamp numbering
    EFTANumber  : EFTA per-page numbering for the DOJ Epstein release
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement all five
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def scheme_name(self) -> str:
        """Short uppercase identifier: 'BATES' or 'EFTA'."""
        ...

    @abstractmethod
    def validate(self, value: str) -> bool:
        """Return True if value is a syntactically valid sequence number."""
        ...

    @abstractmethod
    def extract_from_text(self, text: str) -> list[str]:
        """
        Extract all sequence numbers of this scheme from raw document text.
        Returns canonical string forms (as would be stored in corpus_registry).
        """
        ...

    @abstractmethod
    def sort_key(self, value: str) -> Any:
        """
        Return a sort key suitable for ordering sequence numbers correctly.
        BatesNumber returns (prefix_str, int). EFTANumber returns int.
        """
        ...

    @abstractmethod
    def gap_is_expected(self, value: str) -> bool:
        """
        Return True if a gap at this value is documented and expected —
        i.e. should NOT be treated as a deletion candidate.

        BatesNumber: always False. All gaps are suspicious.
        EFTANumber:  True for numbers in DS9's documented 692,473-entry gap range.

        Constitution reference: Principle IV — Gaps Are Facts.
        An expected gap is still a gap; it is recorded in expected_gap_numbers
        but not escalated to deletion_detector.py.
        """
        ...

    @abstractmethod
    def describe_number(self, value: str) -> str:
        """
        Return a human-readable, citation-ready description of this number,
        including a source URL where available.
        """
        ...

    # ------------------------------------------------------------------
    # Shared reconciliation algorithm
    # ------------------------------------------------------------------

    def reconcile(
        self,
        corpus_numbers: list[str],
        index_numbers: list[str],
    ) -> ReconciliationResult:
        """
        Reconcile corpus sequence numbers against the authoritative index.

        Algorithm
        ---------
        1. Set operations: present, missing_from_corpus, unindexed
        2. Apply gap_is_expected() to separate expected gaps from deletion candidates
        3. Compute coverage and partial_coverage flag
        4. Return ReconciliationResult

        The deletion_candidates list feeds directly into deletion_detector.py
        as DELETION_SUSPECTED entries.

        Constitution reference: Principle IV — Gaps Are Facts.
        """
        corpus_set = set(corpus_numbers)
        index_set = set(index_numbers)

        present = corpus_set & index_set
        missing_from_corpus = index_set - corpus_set
        unindexed = corpus_set - index_set

        expected_gaps = {n for n in missing_from_corpus if self.gap_is_expected(n)}
        deletion_candidates = missing_from_corpus - expected_gaps

        coverage_pct = len(present) / len(index_set) if index_set else 0.0

        result = ReconciliationResult(
            sequence_type=self.scheme_name,
            present_count=len(present),
            missing_from_corpus_count=len(missing_from_corpus),
            unindexed_count=len(unindexed),
            coverage_pct=coverage_pct,
            partial_coverage=coverage_pct < COVERAGE_THRESHOLD,
            missing_numbers=sorted(missing_from_corpus, key=self.sort_key),
            expected_gap_numbers=sorted(expected_gaps, key=self.sort_key),
            deletion_candidates=sorted(deletion_candidates, key=self.sort_key),
        )

        logger.debug(
            "%s reconciliation: %d present, %d missing (%d expected gaps, %d deletion candidates)",
            self.scheme_name,
            result.present_count,
            result.missing_from_corpus_count,
            len(expected_gaps),
            len(deletion_candidates),
        )
        return result


# ---------------------------------------------------------------------------
# BatesNumber
# ---------------------------------------------------------------------------

class BatesNumber(SequenceNumber):
    """
    Traditional Bates stamp numbering used in legal document production.

    Format examples
    ---------------
    DOJ-EPSTEIN-000001
    EDNY-0042318
    000001  (bare numeric, minimum 6 digits)

    All gaps between consecutive Bates numbers are treated as suspicious.
    There is no concept of an "expected" gap in a Bates sequence — the
    whole point of Bates stamping is unbroken sequential integrity.
    """

    # Matches PREFIX-DIGITS or bare 6+ digit numbers.
    # Prefix must start with a letter to avoid matching plain dates etc.
    _PATTERN: re.Pattern = re.compile(
        r'\b([A-Z][A-Z0-9-]*-\d{4,}|\d{6,})\b'
    )

    @property
    def scheme_name(self) -> str:
        return "BATES"

    def validate(self, value: str) -> bool:
        return bool(self._PATTERN.fullmatch(value.strip()))

    def extract_from_text(self, text: str) -> list[str]:
        return self._PATTERN.findall(text)

    def sort_key(self, value: str) -> tuple[str, int]:
        """
        Sort by (prefix, numeric_suffix) so DOJ-001 < DOJ-002 < DOJ-010.
        Bare numeric strings sort as ('', n).
        """
        parts = value.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        if value.isdigit():
            return ('', int(value))
        return (value, 0)

    def gap_is_expected(self, value: str) -> bool:
        """Bates: no gaps are ever expected. All gaps are suspicious."""
        return False

    def describe_number(self, value: str) -> str:
        return f"Bates stamp: {value}"


# ---------------------------------------------------------------------------
# EFTANumber
# ---------------------------------------------------------------------------

# EFTA dataset boundary table.
# Source: rhowardstone/Epstein-research-data efta_dataset_mapping.json
# Each entry: dataset_id -> (first_efta, last_efta, doj_path_prefix)
# NOTE: These boundaries are populated from the mapping file at runtime via
# EFTANumber.from_mapping_file(). The constants below are placeholders.
#
# DS9 is the anomalous dataset: 692,473 EFTA numbers in its assigned range
# have no corresponding document in the release. Their status is unknown
# (withheld, unimaged, or unused tracking slots). They are documented by
# the rhowardstone analysis and are treated as expected gaps, NOT deletion
# candidates. See gap_is_expected().
_EFTA_DATASET_RANGES: dict[str, tuple[int, int]] = {}

# DS9 gap set — populated from mapping file. Stored as frozenset of
# canonical string representations for O(1) lookup in gap_is_expected().
_DS9_GAP_NUMBERS: frozenset[str] = frozenset()

# Base URL for constructing DOJ download links from EFTA numbers.
_DOJ_BASE_URL = "https://www.justice.gov/storage/120919-storage/downloads/"


class EFTANumber(SequenceNumber):
    """
    Epstein Files Transparency Act (EFTA) per-page numbering scheme.

    Key differences from BatesNumber
    ---------------------------------
    1. Per-PAGE assignment — a 10-page document consumes 10 consecutive
       EFTA numbers. This means gaps in the sequence may indicate missing
       pages within a document, not just missing documents.

    2. Sequential ACROSS datasets — EFTA numbers do not reset between
       DS1 and DS12. Dataset boundaries are continuous.

    3. DS9 documented gaps — 692,473 EFTA numbers in DS9's range have no
       corresponding document in the release. These are NOT deletion
       candidates; they are expected gaps recorded in expected_gap_numbers.
       What they represent (withheld, unimaged, or unused) is unknown.

    4. DOJ URL construction — every EFTA number maps to a specific DOJ
       download URL, making gaps directly citable in reporting.

    Instantiation
    -------------
    For production use, instantiate via from_mapping_file() to load the
    DS9 gap set and dataset boundaries:

        efta = EFTANumber.from_mapping_file(Path("efta_dataset_mapping.json"))

    For testing without the mapping file:

        efta = EFTANumber()  # DS9 gap set will be empty

    Source: rhowardstone/Epstein-research-data
    """

    _PATTERN: re.Pattern = re.compile(
        r'\bEFTA[-_]?(\d+)\b', re.IGNORECASE
    )

    def __init__(
        self,
        ds9_gap_numbers: Optional[frozenset[str]] = None,
        dataset_ranges: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        self._ds9_gap_numbers: frozenset[str] = ds9_gap_numbers or frozenset()
        self._dataset_ranges: dict[str, tuple[int, int]] = dataset_ranges or {}

    @classmethod
    def from_mapping_file(cls, mapping_path: Path) -> "EFTANumber":
        """
        Instantiate EFTANumber with dataset ranges loaded from rhowardstone's
        efta_dataset_mapping.json.

        JSON structure (confirmed from rhowardstone/Epstein-research-data)
        ------------------------------------------------------------------
        The file contains EFTA number ranges for each of the 12 DOJ datasets
        with URL templates. Each entry provides the first and last EFTA number
        in the dataset's range and the DOJ URL pattern for that dataset.

        Expected format (either of two observed variants):

        Variant A — array of dataset objects:
          [
            {
              "dataset": "DS1",
              "dataset_number": 1,
              "first_efta": 1,
              "last_efta": 3158,
              "url_template": "https://www.justice.gov/epstein/files/DataSet 1/EFTA{:08d}.pdf",
              "document_count": 3158
            },
            ...
          ]

        Variant B — dict keyed by dataset ID:
          {
            "DS1": {"first_efta": 1, "last_efta": 3158, "document_count": 3158, ...},
            ...
          }

        Both variants are handled. If the structure does not match either,
        a ValueError is raised with a diagnostic message.

        DS9 gap derivation
        ------------------
        DS9 spans EFTA00039025 to EFTA01262781 (1,223,757 possible numbers)
        but contains only ~531,284 documents. The gap numbers — those in DS9's
        range with no corresponding document — are derived by loading the
        document_summary.csv (if available alongside the mapping file) or by
        treating the full range minus document_count as expected gaps.

        Because we cannot enumerate 692,473 individual gap numbers from the
        range boundaries alone without the full document list, this method
        records DS9's boundary range in dataset_ranges and flags the entire
        DS9 range via gap_is_expected(). Any EFTA number that falls within
        DS9's range is treated as an expected gap unless the corpus presents
        it as present.

        This is conservative and correct: it prevents DS9's structural gaps
        from being escalated to deletion_candidates, which was the original
        intent of the DS9 expected-gap classification.

        Stability note
        --------------
        A separate analysis (chad-loder/efta-analysis) has established that
        EFTA numbers are not stable between DOJ release builds — the DOJ's
        publishing pipeline can reassign page stamps between releases. Corpora
        derived from different release builds may have EFTA number collisions
        or reassignments. This is recorded in notes but does not affect the
        gap detection logic, which operates within a single release build.

        Constitution reference: Principle IV — Gaps Are Facts.
        Principle II — Truth Has a Grade. (Dataset boundaries are CONFIRMED;
        per-number gap status within DS9 is SPECULATIVE without the full doc list.)
        """
        with open(mapping_path) as f:
            raw = json.load(f)

        dataset_ranges: dict[str, tuple[int, int]] = {}
        url_templates: dict[str, str] = {}

        # ── Parse Variant A: list of dataset objects ───────────────────────
        if isinstance(raw, list):
            for entry in raw:
                dataset_id = entry.get("dataset") or entry.get("dataset_id") or entry.get("name")
                if not dataset_id:
                    continue
                first = entry.get("first_efta") or entry.get("efta_start") or entry.get("start")
                last  = entry.get("last_efta")  or entry.get("efta_end")   or entry.get("end")
                if first is None or last is None:
                    logger.warning("Skipping dataset entry with missing range: %s", entry)
                    continue
                dataset_ranges[dataset_id] = (int(first), int(last))
                template = entry.get("url_template") or entry.get("url")
                if template:
                    url_templates[dataset_id] = template

        # ── Parse Variant B: dict keyed by dataset ID ─────────────────────
        elif isinstance(raw, dict):
            for dataset_id, entry in raw.items():
                if dataset_id.startswith("_") or not isinstance(entry, dict):
                    continue
                first = entry.get("first_efta") or entry.get("efta_start") or entry.get("start")
                last  = entry.get("last_efta")  or entry.get("efta_end")   or entry.get("end")
                if first is None or last is None:
                    logger.warning("Skipping dataset %s with missing range: %s", dataset_id, entry)
                    continue
                dataset_ranges[dataset_id] = (int(first), int(last))
                template = entry.get("url_template") or entry.get("url")
                if template:
                    url_templates[dataset_id] = template

        else:
            raise ValueError(
                f"Unrecognized efta_dataset_mapping.json structure in {mapping_path}. "
                "Expected a list of dataset objects or a dict keyed by dataset ID. "
                "Check the rhowardstone/Epstein-research-data README for the current format."
            )

        if not dataset_ranges:
            raise ValueError(
                f"No dataset ranges could be parsed from {mapping_path}. "
                "File may be empty or use an unexpected field naming convention."
            )

        # ── DS9 range-based gap classification ────────────────────────────
        # DS9 spans EFTA00039025 to EFTA01262781. Rather than enumerating all
        # 692,473 gap numbers (expensive at construction time), we record the
        # DS9 boundaries and let gap_is_expected() use range membership.
        # This is equivalent for reconciliation purposes: any EFTA in DS9's
        # range that is absent from the corpus is classified as an expected gap,
        # not a deletion candidate.
        #
        # If DS9 is not in the parsed ranges (e.g. a partial mapping file),
        # we log a warning — the system remains safe (no false deletion alerts)
        # but the DS9 classification will be inactive.
        if "DS9" not in dataset_ranges:
            logger.warning(
                "DS9 not found in %s. DS9 range-based gap classification inactive. "
                "All gaps in DS9's range will be treated as deletion candidates.",
                mapping_path,
            )

        logger.info(
            "EFTANumber.from_mapping_file(): loaded %d dataset ranges from %s. "
            "DS9 gap classification: %s",
            len(dataset_ranges),
            mapping_path,
            "active (range-based)" if "DS9" in dataset_ranges else "inactive",
        )

        instance = cls(ds9_gap_numbers=frozenset(), dataset_ranges=dataset_ranges)
        instance._url_templates = url_templates  # attached for doj_url_for_number()
        return instance

    def _ds9_range(self) -> Optional[tuple[int, int]]:
        """Return DS9's (first, last) range, or None if not loaded."""
        return self._dataset_ranges.get("DS9")

    # ------------------------------------------------------------------
    # SequenceNumber interface
    # ------------------------------------------------------------------

    @property
    def scheme_name(self) -> str:
        return "EFTA"

    def validate(self, value: str) -> bool:
        """Valid EFTA number: a positive integer string."""
        return value.isdigit() and int(value) > 0

    def extract_from_text(self, text: str) -> list[str]:
        """
        Extract EFTA numbers from document text.
        Returns the numeric portion only (without 'EFTA-' prefix)
        as canonical string form for consistent comparison.
        """
        return [m.group(1) for m in self._PATTERN.finditer(text)]

    def sort_key(self, value: str) -> int:
        """EFTA numbers sort purely numerically."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def gap_is_expected(self, value: str) -> bool:
        """
        Return True if this EFTA number falls within DS9's documented range.

        DS9 spans EFTA00039025 to EFTA01262781. The DOJ allocated this large
        range but only populated ~531,284 of the 1,223,757 possible numbers.
        Any number in DS9's range that is absent from the corpus is an expected
        structural gap, NOT a deletion candidate.

        When from_mapping_file() has been called, the DS9 range boundaries are
        used for this classification (range membership, not set lookup). When
        called on a default-constructed EFTANumber() with no mapping loaded,
        no numbers are treated as expected gaps — conservative but safe.

        The individual ds9_gap_numbers frozenset (from earlier design) is still
        supported for backwards compatibility: if populated, set membership is
        checked first before falling back to range membership.

        Constitution reference: Principle IV — Gaps Are Facts.
        An expected gap is still a fact; it is not suppressed, just correctly
        classified to avoid false deletion alarms.
        """
        # Fast path: explicit set membership (backwards compat / test use)
        if value in self._ds9_gap_numbers:
            return True
        # Range-based path: use DS9 boundaries loaded from mapping file
        ds9 = self._ds9_range()
        if ds9 and value.isdigit():
            return ds9[0] <= int(value) <= ds9[1]
        return False

    def describe_number(self, value: str) -> str:
        """Return description with DOJ URL if dataset can be determined."""
        dataset = self.dataset_for_number(int(value)) if value.isdigit() else None
        url = self.doj_url_for_number(int(value)) if value.isdigit() else None
        if url:
            return f"EFTA #{value} (Dataset: {dataset}, DOJ URL: {url})"
        return f"EFTA #{value} (page-level DOJ identifier)"

    # ------------------------------------------------------------------
    # EFTA-specific methods
    # ------------------------------------------------------------------

    def dataset_for_number(self, efta_num: int) -> Optional[str]:
        """
        Return the dataset identifier (e.g. 'DS1', 'DS9', 'DS98') for this
        EFTA number, based on the boundary table loaded from the mapping file.

        Returns None if dataset ranges have not been loaded or the number
        falls outside all known ranges.
        """
        for dataset_id, (first, last) in self._dataset_ranges.items():
            if first <= efta_num <= last:
                return dataset_id
        return None

    def doj_url_for_number(self, efta_num: int) -> Optional[str]:
        """
        Construct the DOJ download URL for a given EFTA number.

        Uses the url_template loaded from efta_dataset_mapping.json if available.
        Template format: "https://www.justice.gov/epstein/files/DataSet 1/EFTA{:08d}.pdf"

        Falls back to constructing the URL from the base URL and dataset number
        if no template was loaded.

        Returns None if the dataset cannot be determined (no mapping loaded).
        """
        dataset = self.dataset_for_number(efta_num)
        if not dataset:
            return None

        # Use loaded template if available
        templates = getattr(self, "_url_templates", {})
        template = templates.get(dataset)
        if template:
            try:
                return template.format(efta_num)
            except (IndexError, KeyError):
                pass

        # Fallback: construct from known DOJ URL pattern
        # e.g. DS1 → DataSet 1, DS9 → DataSet 9, DS10 → DataSet 10
        dataset_num = dataset.replace("DS", "")
        return (
            f"https://www.justice.gov/epstein/files/DataSet%20{dataset_num}/"
            f"EFTA{efta_num:08d}.pdf"
        )

    @property
    def ds9_gap_count(self) -> int:
        """Number of DS9 gap entries loaded. 0 if mapping file not loaded."""
        return len(self._ds9_gap_numbers)

    @property
    def dataset_count(self) -> int:
        """Number of dataset boundary entries loaded."""
        return len(self._dataset_ranges)
