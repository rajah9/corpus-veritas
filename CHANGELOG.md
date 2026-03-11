# Changelog

All notable changes to corpus-veritas will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — Layer 1 in development

### Added
- `pipeline/models.py` — shared enums and data models:
  - `DeletionFlag(str, Enum)` — six-value taxonomy with ordering operators and `confidence_tier`, `requires_human_review`, `is_government_acknowledged` properties
  - `WithholdingRecord` — structured model for government-acknowledged withheld documents with DynamoDB serialisation, `mark_released()`, `is_overdue` alerting
  - `DocumentState` — document lifecycle enum for DynamoDB registry
- `pipeline/deletion_detector.py` — full implementation:
  - `DetectionSignals` — three-signal model with `derived_flag` property
  - `DeletionRecord` — evidence-graded deletion finding with signal provenance
  - `FBI302SeriesResult` — selective withholding detection for 302 interview series
  - `create_deletion_finding()` — factory for evidence-graded `DeletionRecord`
  - `create_acknowledged_withholding()` — factory for `WithholdingRecord`; derives `WITHHELD_SELECTIVELY` vs `WITHHELD_ACKNOWLEDGED` from sibling presence
  - `check_302_series()` — detects partial release within an FBI 302 interview series
- `tests/integration/test_deletion_detector.py` — full test suite including `TestWSJScenario` modelling the March 2026 WSJ/DOJ findings

### Changed
- `docs/ARCHITECTURE.md` — deletion flag taxonomy table updated with `WITHHELD_ACKNOWLEDGED` and `WITHHELD_SELECTIVELY`; DynamoDB storage row updated (v0.2)
- `CHANGELOG.md` — this entry

### Added
- `pipeline/sequence_numbers.py` — `SequenceNumber` ABC with `BatesNumber` and `EFTANumber` concrete subclasses
- `EFTANumber.from_mapping_file()` — production constructor loading DS9 gap set from rhowardstone mapping file (parsing implementation pending)
- `ReconciliationResult` — replaces `BatesReconciliationResult`; adds `expected_gap_numbers` and `deletion_candidates` as distinct fields
- Contract test class `TestSequenceSchemeContract` — parameterised tests both schemes must satisfy
- Full `EFTANumber` test suite including DS9 gap separation, numeric sort ordering, and large-scale reconciliation

### Changed
- `corpus_evaluator.py` Check 2 refactored from `check_bates_reconciliation()` to scheme-agnostic `check_sequence_reconciliation()` accepting any `SequenceNumber` instance
- `evaluate_corpus()` now accepts `sequence_scheme` parameter; defaults to `EFTANumber()` for DOJ Epstein corpora
- `corpus_registry.json` registry entries now record `sequence_scheme` and `deletion_candidates_count` alongside coverage metrics
- `CONSTITUTION.md` Principle IV updated to document EFTA expected gap distinction (v0.2)
- `docs/ARCHITECTURE.md` updated with `SequenceNumber` class hierarchy table and DS9 gap explanation (v0.2)

### Removed
- `BatesReconciliationResult` dataclass (replaced by `ReconciliationResult`)
- `check_bates_reconciliation()` standalone function (logic moved to `BatesNumber.reconcile()`)
- `BATES_COVERAGE_THRESHOLD` module constant (moved to `sequence_numbers.COVERAGE_THRESHOLD`)

---

## [0.1.0] — Initial scaffold

### Added
- `CONSTITUTION.md` — ethical framework governing all project decisions
- `docs/ARCHITECTURE.md` — full five-layer technical specification
- `corpus_registry.json` — schema and placeholder for external corpus tracking
- `trusted_endorsers.json` — schema and placeholder for community vetting list
- Initial repository structure and pipeline skeletons

---

*Versions will be tagged when each layer branch is merged to main.*
