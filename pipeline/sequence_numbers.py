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
        Instantiate EFTANumber with DS9 gap set and dataset ranges loaded
        from rhowardstone's efta_dataset_mapping.json.

        This is the production-ready constructor. Use it whenever the
        mapping file is available to ensure gap_is_expected() is accurate.
        """
        with open(mapping_path) as f:
            mapping = json.load(f)

        # TODO: parse mapping into ds9_gap_numbers and dataset_ranges
        # The exact structure of efta_dataset_mapping.json will determine
        # the parsing logic. Update this method when the file is obtained.
        #
        # Expected outcome:
        #   ds9_gap_numbers = frozenset of str(efta_num) for each DS9 gap
        #   dataset_ranges  = {"DS1": (first, last), "DS2": ..., ...}

        logger.warning(
            "EFTANumber.from_mapping_file(): mapping parsing not yet implemented. "
            "DS9 gap set will be empty — all DS9 gaps treated as deletion candidates. "
            "Implement this method once efta_dataset_mapping.json structure is confirmed."
        )
        return cls(ds9_gap_numbers=frozenset(), dataset_ranges={})

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
        Return True if this EFTA number is in DS9's documented gap range.

        These 692,473 numbers appear in the EFTA sequential range but have
        no corresponding document in the DOJ release. Their status is unknown
        but their absence is documented by the rhowardstone analysis. They
        are recorded as expected_gap_numbers, NOT escalated to deletion_detector.

        Constitution reference: Principle IV — Gaps Are Facts.
        An expected gap is still a fact; it is not suppressed, just correctly
        classified to avoid false deletion alarms.
        """
        return value in self._ds9_gap_numbers

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

        URL format: {_DOJ_BASE_URL}{dataset}/{filename}
        Requires dataset_ranges to be populated from the mapping file.

        Returns None if the URL cannot be constructed (no mapping loaded).
        """
        # TODO: implement URL construction once efta_dataset_mapping.json
        # structure is confirmed and from_mapping_file() is complete.
        dataset = self.dataset_for_number(efta_num)
        if not dataset:
            return None
        # Placeholder — actual filename mapping needs the full mapping table
        return f"{_DOJ_BASE_URL}{dataset.lower()}/"

    @property
    def ds9_gap_count(self) -> int:
        """Number of DS9 gap entries loaded. 0 if mapping file not loaded."""
        return len(self._ds9_gap_numbers)

    @property
    def dataset_count(self) -> int:
        """Number of dataset boundary entries loaded."""
        return len(self._dataset_ranges)
