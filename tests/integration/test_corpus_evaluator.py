"""
Integration tests for Layer 1: Corpus Evaluator

Tests the corpus_evaluator.py pipeline against synthetic fixtures.
Real document content is never committed to this repository.
All test fixtures use synthetic Bates numbers and placeholder text.
"""

import json
import pytest
from pathlib import Path
from pipeline.corpus_evaluator import (
    check_bates_reconciliation,
    BatesReconciliationResult,
    BATES_COVERAGE_THRESHOLD,
)


class TestBatesReconciliation:
    """Tests for Check 2: Bates stamp reconciliation logic."""

    def test_full_coverage(self):
        index = ["DOJ-001", "DOJ-002", "DOJ-003"]
        corpus = ["DOJ-001", "DOJ-002", "DOJ-003"]
        result = check_bates_reconciliation(corpus, index)
        assert result.present_count == 3
        assert result.missing_from_corpus_count == 0
        assert result.coverage_pct == 1.0
        assert not result.partial_coverage

    def test_missing_documents_flagged(self):
        index = ["DOJ-001", "DOJ-002", "DOJ-003", "DOJ-004", "DOJ-005"]
        corpus = ["DOJ-001", "DOJ-002"]
        result = check_bates_reconciliation(corpus, index)
        assert result.missing_from_corpus_count == 3
        assert "DOJ-003" in result.missing_bates_numbers
        assert "DOJ-004" in result.missing_bates_numbers

    def test_partial_coverage_threshold(self):
        # Corpus covers exactly at the threshold — should NOT be flagged partial
        index = [f"DOJ-{i:03}" for i in range(100)]
        corpus = [f"DOJ-{i:03}" for i in range(60)]  # 60%
        result = check_bates_reconciliation(corpus, index)
        assert not result.partial_coverage

    def test_below_threshold_flagged(self):
        index = [f"DOJ-{i:03}" for i in range(100)]
        corpus = [f"DOJ-{i:03}" for i in range(59)]  # 59% — below threshold
        result = check_bates_reconciliation(corpus, index)
        assert result.partial_coverage

    def test_unindexed_documents_flagged(self):
        index = ["DOJ-001", "DOJ-002"]
        corpus = ["DOJ-001", "DOJ-002", "DOJ-999"]  # DOJ-999 not in index
        result = check_bates_reconciliation(corpus, index)
        assert result.unindexed_count == 1

    def test_empty_corpus(self):
        index = ["DOJ-001", "DOJ-002", "DOJ-003"]
        corpus = []
        result = check_bates_reconciliation(corpus, index)
        assert result.coverage_pct == 0.0
        assert result.partial_coverage
        assert result.missing_from_corpus_count == 3
