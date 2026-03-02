"""
corpus_evaluator.py
Layer 1, Sub-Module 1B: Corpus Verification Pipeline

Runs three sequential checks on any external corpus candidate before ingestion:
  1. Git history audit
  2. Bates stamp reconciliation against DOJ index
  3. Community vetting against trusted_endorsers.json

Each check gates the next. Results are written back to corpus_registry.json.

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

# GitHub API client — requires PyGithub
# from github import Github

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "corpus_registry.json"
ENDORSERS_PATH = Path(__file__).parent.parent / "trusted_endorsers.json"

# Corpora covering less than this fraction of the DOJ index are tagged PARTIAL_COVERAGE
BATES_COVERAGE_THRESHOLD = 0.60

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
class BatesReconciliationResult:
    present_count: int = 0
    missing_from_corpus_count: int = 0
    unindexed_count: int = 0
    coverage_pct: float = 0.0
    partial_coverage: bool = False
    missing_bates_numbers: list[str] = field(default_factory=list)


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
    bates_reconciliation: BatesReconciliationResult
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
# Check 2: Bates Stamp Reconciliation
# ---------------------------------------------------------------------------

def check_bates_reconciliation(
    corpus_bates_numbers: list[str],
    doj_index_bates_numbers: list[str],
) -> BatesReconciliationResult:
    """
    Reconcile Bates numbers in the corpus against the DOJ index manifest.

    Produces three counts:
    - PRESENT: in both index and corpus
    - MISSING_FROM_CORPUS: in index, not in corpus → feeds deletion_detector.py
    - UNINDEXED: in corpus, not in index → flag for human review

    Corpora with coverage < BATES_COVERAGE_THRESHOLD are tagged PARTIAL_COVERAGE.

    The list of missing_bates_numbers is passed to deletion_detector.py as
    DELETION_SUSPECTED candidates.
    """
    corpus_set = set(corpus_bates_numbers)
    index_set = set(doj_index_bates_numbers)

    present = corpus_set & index_set
    missing_from_corpus = index_set - corpus_set
    unindexed = corpus_set - index_set

    coverage_pct = len(present) / len(index_set) if index_set else 0.0

    return BatesReconciliationResult(
        present_count=len(present),
        missing_from_corpus_count=len(missing_from_corpus),
        unindexed_count=len(unindexed),
        coverage_pct=coverage_pct,
        partial_coverage=coverage_pct < BATES_COVERAGE_THRESHOLD,
        missing_bates_numbers=sorted(missing_from_corpus),
    )


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
    Bates reconciliation. It does not mean SHA-256 hash verification.
    """
    with open(endorsers_path) as f:
        endorsers_data = json.load(f)

    # TODO: Implement endorser lookup
    # This will eventually query GitHub stargazers/citation lists and cross-reference
    # against the endorser registry. For now, returns UNVERIFIED by default.
    #
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
    doj_index_bates_numbers: list[str],
    corpus_bates_numbers: Optional[list[str]] = None,
) -> EvaluationResult:
    """
    Run all three checks in sequence. Each check gates the next.

    Returns an EvaluationResult with a final provenance tag and ingestion decision.
    Writes the result back to corpus_registry.json.

    Constitution reference: Article III Hard Limit 6.
    """
    logger.info(f"Evaluating corpus {corpus_id}: {github_url}")
    evaluation_date = datetime.now(timezone.utc).isoformat()

    # Check 1: Git integrity
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
            bates_reconciliation=BatesReconciliationResult(),
            community_vetting=CommunityVettingResult(),
            final_provenance_tag="PROVENANCE_REJECTED",
            ingestion_approved=False,
            notes="Rejected at Check 1 (git integrity). Will not ingest per Hard Limit 6.",
        )

    # Check 2: Bates reconciliation
    if corpus_bates_numbers is None:
        logger.warning("No Bates numbers provided — skipping reconciliation")
        bates_result = BatesReconciliationResult(
            coverage_pct=0.0,
            partial_coverage=True,
        )
    else:
        bates_result = check_bates_reconciliation(
            corpus_bates_numbers, doj_index_bates_numbers
        )

    # Check 3: Community vetting
    vetting_result = check_community_vetting(github_url)

    # Determine final provenance tag
    if vetting_result.endorsed and not bates_result.partial_coverage:
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
        bates_reconciliation=bates_result,
        community_vetting=vetting_result,
        final_provenance_tag=final_tag,
        ingestion_approved=approved,
        notes=f"Coverage: {bates_result.coverage_pct:.1%}. "
              f"Missing from corpus: {bates_result.missing_from_corpus_count} documents.",
    )

    _write_to_registry(result)
    return result


def _write_to_registry(result: EvaluationResult) -> None:
    """Append or update evaluation result in corpus_registry.json."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    entry = {
        "corpus_id": result.corpus_id,
        "status": "EVALUATED",
        "github_url": result.github_url,
        "evaluated_commit_hash": result.git_integrity.commit_hash_evaluated,
        "evaluation_date": result.evaluation_date,
        "git_integrity_score": result.git_integrity.score,
        "git_integrity_notes": result.git_integrity.notes,
        "bates_coverage_pct": round(result.bates_reconciliation.coverage_pct * 100, 1),
        "missing_from_corpus_count": result.bates_reconciliation.missing_from_corpus_count,
        "partial_coverage": result.bates_reconciliation.partial_coverage,
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
        f"Corpus {result.corpus_id} recorded: "
        f"{result.final_provenance_tag}, approved={result.ingestion_approved}"
    )
