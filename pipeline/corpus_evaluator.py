"""
corpus_evaluator.py
Layer 1, Sub-Module 1B: Corpus Verification Pipeline

Runs three sequential checks on any external corpus candidate before ingestion:
  1. Git history audit
  2. Sequence number reconciliation against the DOJ index
     (scheme-agnostic: accepts BatesNumber or EFTANumber via the SequenceNumber ABC)
  3. Community vetting against trusted_endorsers.json

Each check gates the next. Results are written back to corpus_registry.json.

The DOJ Epstein release uses EFTA numbers (per-page, sequential across all datasets),
not traditional Bates stamps. Pass an EFTANumber instance for Check 2 when evaluating
DOJ Epstein corpora. BatesNumber remains supported for other legal document corpora.

See docs/ARCHITECTURE.md § Layer 1, Sub-Module 1B for full specification.
See CONSTITUTION.md Article III Hard Limit 6 for the rejection rule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pipeline.sequence_numbers import (
    BatesNumber,
    EFTANumber,
    ReconciliationResult,
    SequenceNumber,
)

# GitHub API client — requires PyGithub
# from github import Github

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "corpus_registry.json"
ENDORSERS_PATH = Path(__file__).parent.parent / "trusted_endorsers.json"

# If initial and most recent commits differ by more than this fraction of total bytes
# with no descriptive commit message, flag for review
GIT_DIFF_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GitIntegrityResult:
    score: str  # CLEAN | REVIEW_RECOMMENDED | REJECTED
    notes: str
    commit_hash_evaluated: Optional[str] = None
    forced_pushes_detected: bool = False
    suspicious_modification_gap: bool = False


@dataclass
class CommunityVettingResult:
    endorsed: bool = False
    endorsing_orgs: list[str] = field(default_factory=list)
    provenance_tag: str = "PROVENANCE_UNVERIFIED"


@dataclass
class EvaluationResult:
    corpus_id: str
    github_url: str
    evaluation_date: str
    git_integrity: GitIntegrityResult
    sequence_reconciliation: ReconciliationResult
    community_vetting: CommunityVettingResult
    final_provenance_tag: str
    ingestion_approved: bool
    notes: str


# ---------------------------------------------------------------------------
# Check 1: Git History Audit
# ---------------------------------------------------------------------------

def check_git_integrity(github_url: str) -> GitIntegrityResult:
    """
    Audit the commit history of a GitHub corpus repository.

    Flags repositories where:
    - Initial and most recent commits differ by >GIT_DIFF_THRESHOLD of total bytes
      with no descriptive commit message
    - Forced pushes (--force) are detected in history
    - Modification gap is suspicious relative to DOJ release date (Feb 2025)

    Returns a GitIntegrityResult with score CLEAN, REVIEW_RECOMMENDED, or REJECTED.

    Constitution reference: Hard Limit 6 — REJECTED corpora will not be ingested.
    """
    # TODO: Implement using PyGithub
    # Milestone: Layer 1 branch
    #
    # Skeleton:
    #   g = Github(os.environ["GITHUB_TOKEN"])
    #   repo = g.get_repo(parse_repo_path(github_url))
    #   commits = list(repo.get_commits())
    #   ...evaluate commit history...
    #
    raise NotImplementedError(
        "check_git_integrity() not yet implemented. "
        "See Layer 1 branch for implementation target."
    )


# ---------------------------------------------------------------------------
# Check 2: Sequence Number Reconciliation (scheme-agnostic)
# ---------------------------------------------------------------------------

def check_sequence_reconciliation(
    corpus_numbers: list[str],
    index_numbers: list[str],
    scheme: SequenceNumber,
) -> ReconciliationResult:
    """
    Reconcile sequence numbers in the corpus against the DOJ index manifest,
    using the provided SequenceNumber scheme.

    Pass a BatesNumber() instance for traditional legal corpora.
    Pass an EFTANumber() or EFTANumber.from_mapping_file(path) instance
    for the DOJ Epstein release corpora.

    The returned ReconciliationResult.deletion_candidates list feeds directly
    into deletion_detector.py as DELETION_SUSPECTED entries.

    Note on EFTA: expected_gap_numbers are NOT deletion candidates. They are
    DS9 gaps documented by rhowardstone/Epstein-research-data whose absence
    is expected. They are recorded for transparency, not escalated.

    Constitution reference: Principle IV — Gaps Are Facts.
    """
    return scheme.reconcile(corpus_numbers, index_numbers)


# ---------------------------------------------------------------------------
# Check 3: Community Vetting
# ---------------------------------------------------------------------------

def check_community_vetting(
    github_url: str,
    endorsers_path: Path = ENDORSERS_PATH,
) -> CommunityVettingResult:
    """
    Cross-reference the corpus against trusted_endorsers.json.

    If the repo is cited or used by a trusted endorser, upgrades provenance tag
    from PROVENANCE_UNVERIFIED to PROVENANCE_COMMUNITY_VOUCHED.

    Note: endorsement is not the same as cryptographic verification.
    A COMMUNITY_VOUCHED corpus has been used by a trusted source AND passed
    sequence reconciliation. It does not mean SHA-256 hash verification.
    """
    with open(endorsers_path) as f:
        endorsers_data = json.load(f)  # noqa: F841 — used in TODO below

    # TODO: Implement endorser lookup
    # Query GitHub stargazers/citation lists and cross-reference against
    # the endorser registry. For now, returns UNVERIFIED by default.
    # Milestone: Layer 1 branch

    return CommunityVettingResult(
        endorsed=False,
        endorsing_orgs=[],
        provenance_tag="PROVENANCE_UNVERIFIED",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_corpus(
    corpus_id: str,
    github_url: str,
    index_numbers: list[str],
    corpus_numbers: Optional[list[str]] = None,
    sequence_scheme: Optional[SequenceNumber] = None,
) -> EvaluationResult:
    """
    Run all three checks in sequence. Each check gates the next.

    Parameters
    ----------
    corpus_id        : unique identifier for this corpus (stored in corpus_registry.json)
    github_url       : GitHub URL of the corpus repository
    index_numbers    : authoritative sequence numbers from the DOJ index manifest
    corpus_numbers   : sequence numbers extracted from the corpus (None = skip Check 2)
    sequence_scheme  : SequenceNumber instance defining the numbering scheme.
                       Defaults to EFTANumber() for the DOJ Epstein release.
                       Pass BatesNumber() for traditional legal corpora.

    Returns an EvaluationResult and writes it to corpus_registry.json.

    Constitution reference: Article III Hard Limit 6.
    """
    if sequence_scheme is None:
        sequence_scheme = EFTANumber()
        logger.info("No sequence scheme specified — defaulting to EFTANumber")

    logger.info(
        "Evaluating corpus %s: %s (scheme: %s)",
        corpus_id, github_url, sequence_scheme.scheme_name,
    )
    evaluation_date = datetime.now(timezone.utc).isoformat()

    # Check 1: Git integrity — gates all subsequent checks
    try:
        git_result = check_git_integrity(github_url)
    except NotImplementedError:
        logger.warning("check_git_integrity() not implemented — skipping for scaffold")
        git_result = GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes="Git check not yet implemented — manual review required",
        )

    if git_result.score == "REJECTED":
        return EvaluationResult(
            corpus_id=corpus_id,
            github_url=github_url,
            evaluation_date=evaluation_date,
            git_integrity=git_result,
            sequence_reconciliation=ReconciliationResult(
                sequence_type=sequence_scheme.scheme_name
            ),
            community_vetting=CommunityVettingResult(),
            final_provenance_tag="PROVENANCE_REJECTED",
            ingestion_approved=False,
            notes="Rejected at Check 1 (git integrity). Will not ingest per Hard Limit 6.",
        )

    # Check 2: Sequence reconciliation
    if corpus_numbers is None:
        logger.warning("No sequence numbers provided — skipping reconciliation")
        reconciliation_result = ReconciliationResult(
            sequence_type=sequence_scheme.scheme_name,
            coverage_pct=0.0,
            partial_coverage=True,
        )
    else:
        reconciliation_result = check_sequence_reconciliation(
            corpus_numbers, index_numbers, sequence_scheme
        )

    # Check 3: Community vetting
    vetting_result = check_community_vetting(github_url)

    # Determine final provenance tag
    if vetting_result.endorsed and not reconciliation_result.partial_coverage:
        final_tag = "PROVENANCE_COMMUNITY_VOUCHED"
    elif git_result.score == "REVIEW_RECOMMENDED":
        final_tag = "PROVENANCE_FLAGGED"
    else:
        final_tag = "PROVENANCE_UNVERIFIED"

    approved = final_tag != "PROVENANCE_REJECTED"

    result = EvaluationResult(
        corpus_id=corpus_id,
        github_url=github_url,
        evaluation_date=evaluation_date,
        git_integrity=git_result,
        sequence_reconciliation=reconciliation_result,
        community_vetting=vetting_result,
        final_provenance_tag=final_tag,
        ingestion_approved=approved,
        notes=(
            f"Scheme: {sequence_scheme.scheme_name}. "
            f"Coverage: {reconciliation_result.coverage_pct:.1%}. "
            f"Missing from corpus: {reconciliation_result.missing_from_corpus_count}. "
            f"Deletion candidates: {len(reconciliation_result.deletion_candidates)}. "
            f"Expected gaps (not deletion candidates): "
            f"{len(reconciliation_result.expected_gap_numbers)}."
        ),
    )

    _write_to_registry(result)
    return result


def _write_to_registry(result: EvaluationResult) -> None:
    """Append or update evaluation result in corpus_registry.json."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    rec = result.sequence_reconciliation
    entry = {
        "corpus_id": result.corpus_id,
        "status": "EVALUATED",
        "github_url": result.github_url,
        "evaluated_commit_hash": result.git_integrity.commit_hash_evaluated,
        "evaluation_date": result.evaluation_date,
        "sequence_scheme": rec.sequence_type,
        "git_integrity_score": result.git_integrity.score,
        "git_integrity_notes": result.git_integrity.notes,
        "coverage_pct": round(rec.coverage_pct * 100, 1),
        "missing_from_corpus_count": rec.missing_from_corpus_count,
        "deletion_candidates_count": len(rec.deletion_candidates),
        "expected_gaps_count": len(rec.expected_gap_numbers),
        "partial_coverage": rec.partial_coverage,
        "community_endorsed": result.community_vetting.endorsed,
        "endorsing_orgs": result.community_vetting.endorsing_orgs,
        "provenance_tag_assigned": result.final_provenance_tag,
        "ingestion_approved": result.ingestion_approved,
        "ingestion_notes": result.notes,
    }

    existing_ids = [c.get("corpus_id") for c in registry.get("corpora", [])]
    if result.corpus_id in existing_ids:
        registry["corpora"] = [
            entry if c.get("corpus_id") == result.corpus_id else c
            for c in registry["corpora"]
        ]
    else:
        registry["corpora"].append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info(
        "Corpus %s recorded: %s, approved=%s",
        result.corpus_id, result.final_provenance_tag, result.ingestion_approved,
    )
