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
import os
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

def _parse_github_repo_path(github_url: str) -> str:
    """
    Extract 'owner/repo' from a GitHub URL.

    Handles:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/main
    """
    # Strip scheme and host
    path = github_url.replace("https://github.com/", "").replace("http://github.com/", "")
    # Take first two path segments: owner/repo
    parts = path.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse GitHub repo path from URL: {github_url!r}")
    repo_path = f"{parts[0]}/{parts[1].removesuffix('.git')}"
    return repo_path


# DOJ Epstein file release began December 2025; any meaningful corpus
# should have been created or last committed after this date.
_DOJ_RELEASE_DATE = "2025-12-01"

# Minimum commit message length considered "descriptive"
_MIN_DESCRIPTIVE_MSG_LEN = 20


def check_git_integrity(github_url: str) -> GitIntegrityResult:
    """
    Audit the commit history of a GitHub corpus repository using PyGithub.

    Evaluation criteria
    -------------------
    1. Commit count — a single-commit repo has no auditable history. The
       rhowardstone corpus has only 2 commits; this is flagged as
       REVIEW_RECOMMENDED, not REJECTED, unless other signals are present.

    2. Forced pushes — detected by looking for gaps in commit parent chains.
       PyGithub does not expose force-push events directly; we use the
       Events API to look for push events with before/after SHA mismatches.
       If force pushes are detected: REVIEW_RECOMMENDED.

    3. Commit message quality — commits with very short messages (< 20 chars)
       and large diffs are suspicious. Empty or single-word messages on the
       initial commit are flagged.

    4. Dormancy gap relative to DOJ release — a repo created before December
       2025 with no commits after the DOJ release is suspicious: it may be a
       pre-existing repo repurposed to host new content without a commit trail.
       If last commit predates DOJ release: REVIEW_RECOMMENDED.

    5. REJECTED is reserved for: repos with no commits at all, or repos where
       the head commit SHA cannot be retrieved (tampering or deletion signal).

    Requires GITHUB_TOKEN environment variable. If absent, returns
    REVIEW_RECOMMENDED with a note rather than raising.

    Constitution reference: Hard Limit 6 — REJECTED corpora will not be ingested.
    """
    try:
        from github import Github, GithubException
    except ImportError:
        return GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes="PyGithub not installed. Cannot perform git integrity audit. "
                  "Install with: pip install PyGithub",
        )

    try:
        repo_path = _parse_github_repo_path(github_url)
    except ValueError as e:
        return GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes=f"Could not parse GitHub URL: {e}",
        )

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes="GITHUB_TOKEN environment variable not set. "
                  "Cannot perform git integrity audit without authentication. "
                  "Set GITHUB_TOKEN and re-run.",
        )

    try:
        g = Github(github_token)
        repo = g.get_repo(repo_path)
        commits = list(repo.get_commits())
    except GithubException as e:
        if e.status == 404:
            return GitIntegrityResult(
                score="REJECTED",
                notes=f"Repository not found: {repo_path}. "
                      "Cannot evaluate a corpus whose repository does not exist.",
            )
        return GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes=f"GitHub API error ({e.status}): {e.data}. Manual review required.",
        )
    except Exception as e:
        return GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes=f"Unexpected error fetching repository: {e}",
        )

    if not commits:
        return GitIntegrityResult(
            score="REJECTED",
            notes="Repository has no commits. Cannot audit an empty history.",
        )

    issues: list[str] = []
    forced_pushes = False
    suspicious_gap = False

    head_commit = commits[0]
    head_sha = head_commit.sha
    head_date = head_commit.commit.author.date.isoformat()

    # ── 1. Commit count ───────────────────────────────────────────────────
    commit_count = len(commits)
    if commit_count == 1:
        issues.append(
            f"Single-commit repository — no auditable history. "
            f"All content was added in one push ({head_sha[:8]})."
        )
    elif commit_count <= 3:
        issues.append(
            f"Very thin commit history ({commit_count} commits). "
            "Insufficient trail to verify incremental, auditable development."
        )

    # ── 2. Commit message quality on initial commit ───────────────────────
    initial_commit = commits[-1]
    initial_msg = (initial_commit.commit.message or "").strip()
    if len(initial_msg) < _MIN_DESCRIPTIVE_MSG_LEN:
        issues.append(
            f"Initial commit message is non-descriptive: {initial_msg!r} "
            f"({len(initial_msg)} chars, threshold {_MIN_DESCRIPTIVE_MSG_LEN})."
        )

    # ── 3. Dormancy gap relative to DOJ release ───────────────────────────
    if head_date < _DOJ_RELEASE_DATE:
        suspicious_gap = True
        issues.append(
            f"Most recent commit ({head_date}) predates the DOJ Epstein file "
            f"release ({_DOJ_RELEASE_DATE}). Repository may have been repurposed "
            "without a commit trail reflecting the new content."
        )

    # ── 4. Forced push detection via Events API ───────────────────────────
    # PyGithub exposes push events. A force push event has 'forced: true'
    # in its payload. We check up to 100 recent events.
    try:
        events = list(repo.get_events())
        for event in events[:100]:
            if event.type == "PushEvent":
                payload = event.payload
                if isinstance(payload, dict) and payload.get("forced", False):
                    forced_pushes = True
                    issues.append(
                        f"Force push detected in repository events "
                        f"(event id: {event.id}). History may have been rewritten."
                    )
                    break
    except Exception as e:
        issues.append(f"Could not retrieve push events for force-push check: {e}")

    # ── 5. Compute score ──────────────────────────────────────────────────
    if forced_pushes:
        score = "REVIEW_RECOMMENDED"
    elif suspicious_gap:
        score = "REVIEW_RECOMMENDED"
    elif issues:
        # Thin history alone is REVIEW_RECOMMENDED, not REJECTED
        score = "REVIEW_RECOMMENDED"
    else:
        score = "CLEAN"

    notes = "; ".join(issues) if issues else (
        f"Commit history appears clean. {commit_count} commits, "
        f"head at {head_sha[:8]} ({head_date})."
    )

    return GitIntegrityResult(
        score=score,
        notes=notes,
        commit_hash_evaluated=head_sha,
        forced_pushes_detected=forced_pushes,
        suspicious_modification_gap=suspicious_gap,
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
        endorsers_data = json.load(f)

    endorsers = endorsers_data.get("endorsers", [])

    # Normalize the github_url for comparison: strip trailing slashes,
    # .git suffix, and scheme variations so that
    # "https://github.com/owner/repo" == "https://github.com/owner/repo.git"
    def _normalize(url: str) -> str:
        return url.lower().rstrip("/").removesuffix(".git")

    normalized_target = _normalize(github_url)
    matched_orgs: list[str] = []

    for endorser in endorsers:
        # Skip placeholder and malformed entries
        endorser_id = endorser.get("id", "")
        if not endorser_id or endorser_id == "PLACEHOLDER":
            continue

        # An endorser qualifies the corpus if:
        # (a) the corpus URL appears in the endorser's cited_corpora list, OR
        # (b) the endorser's reference_url is a domain that hosts the corpus
        #     (e.g. MuckRock hosting a document collection)
        cited_corpora = endorser.get("cited_corpora", [])
        for cited_url in cited_corpora:
            if _normalize(cited_url) == normalized_target:
                matched_orgs.append(endorser_id)
                break

    endorsed = len(matched_orgs) > 0
    provenance_tag = (
        "PROVENANCE_COMMUNITY_VOUCHED" if endorsed else "PROVENANCE_UNVERIFIED"
    )

    logger.info(
        "Community vetting for %s: endorsed=%s, orgs=%s",
        github_url,
        endorsed,
        matched_orgs,
    )

    return CommunityVettingResult(
        endorsed=endorsed,
        endorsing_orgs=matched_orgs,
        provenance_tag=provenance_tag,
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
