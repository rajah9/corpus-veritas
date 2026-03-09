"""
tests/pipeline/test_corpus_evaluator.py

Unit tests for pipeline/corpus_evaluator.py.

Every public function and dataclass is covered here.  The integration suite
in tests/integration/test_corpus_evaluator.py is not duplicated -- that file
tests reconciliation logic via BatesNumber/EFTANumber end-to-end.  This file
tests the evaluator functions in isolation, mocking their collaborators.

Coverage targets
----------------
_parse_github_repo_path()       -- all URL normalisation branches
check_git_integrity()           -- precondition failures (no token, bad URL)
                                   + all mocked evaluation paths (CLEAN,
                                   REVIEW_RECOMMENDED, REJECTED, force push,
                                   dormant repo, null/empty commit message,
                                   events API exception, 404/403/unknown error)
check_sequence_reconciliation() -- thin delegation wrapper
check_community_vetting()       -- match, no-match, URL normalisation,
                                   PLACEHOLDER guard, multiple endorsers,
                                   missing cited_corpora key
_write_to_registry()            -- new entry, update-in-place, two distinct IDs
evaluate_corpus()               -- all provenance tag branches, registry write,
                                   no-corpus-numbers path, scheme selection
GitIntegrityResult              -- all dataclass fields
CommunityVettingResult          -- all dataclass fields
EvaluationResult                -- all dataclass fields

Temp-file strategy
------------------
* TestCheckCommunityVetting: setUp creates a temp dir; each test writes its
  own endorsers file into it; tearDown removes the dir.

* TestWriteToRegistry: setUp creates a temp dir containing an empty registry
  file; tearDown removes the dir.  All tests in the class share the setUp
  registry and write fresh entries into it.

* TestEvaluateCorpus: setUp creates a temp dir with a fresh empty registry;
  tearDown removes the dir.  evaluate_corpus() is called with REGISTRY_PATH
  patched to that file.

Implementation note -- check_community_vetting default argument
---------------------------------------------------------------
check_community_vetting() is defined as:

    def check_community_vetting(github_url, endorsers_path=ENDORSERS_PATH):

The default is evaluated once at import time.  Patching the module-level name
ENDORSERS_PATH after import has no effect on calls that use the default.
evaluate_corpus() calls check_community_vetting(github_url) without passing
endorsers_path, so TestEvaluateCorpus mocks check_community_vetting itself
rather than trying to patch ENDORSERS_PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.corpus_evaluator import (
    CommunityVettingResult,
    EvaluationResult,
    GitIntegrityResult,
    _parse_github_repo_path,
    check_community_vetting,
    check_git_integrity,
    check_sequence_reconciliation,
    evaluate_corpus,
    _write_to_registry,
)
from pipeline.sequence_numbers import BatesNumber, EFTANumber, ReconciliationResult


# ---------------------------------------------------------------------------
# Shared GitHub mock helpers
# ---------------------------------------------------------------------------

# The GithubException class injected into sys.modules["github"].
# Must be the SAME class object used for exception instances so isinstance()
# works inside check_git_integrity()'s except clause.
GH_EXC = type("GithubException", (Exception,), {})


def _make_commit(sha: str, date_iso: str, message) -> MagicMock:
    """Build a minimal PyGithub Commit mock. message may be None."""
    c = MagicMock()
    c.sha = sha
    c.commit.author.date = datetime.fromisoformat(date_iso).replace(
        tzinfo=timezone.utc
    )
    c.commit.message = message
    return c


def _run_git_check(commits, push_events=None, side_effect=None) -> GitIntegrityResult:
    """Run check_git_integrity() with fully mocked PyGithub internals."""
    mock_repo = MagicMock()
    mock_repo.get_commits.return_value = commits
    mock_repo.get_events.return_value = push_events or []

    mock_gh_instance = MagicMock()
    if side_effect is not None:
        mock_gh_instance.get_repo.side_effect = side_effect
    else:
        mock_gh_instance.get_repo.return_value = mock_repo

    mock_gh_module = MagicMock()
    mock_gh_module.Github = MagicMock(return_value=mock_gh_instance)
    mock_gh_module.GithubException = GH_EXC  # same class as used in instances

    with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
        with patch.dict("sys.modules", {"github": mock_gh_module}):
            return check_git_integrity("https://github.com/owner/repo")


def _good_commits(n: int = 10, date: str = "2026-01-15"):
    return [
        _make_commit(
            f"sha{i:04d}", f"{date}T00:00:00", f"Descriptive commit message {i}"
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Shared endorsers data
# ---------------------------------------------------------------------------

ENDORSERS_SIMPLE = {
    "_schema_version": "0.2",
    "endorsers": [
        {
            "id": "RHOWARDSTONE",
            "name": "rhowardstone",
            "cited_corpora": [
                "https://github.com/rhowardstone/Epstein-research-data"
            ],
        },
        {
            "id": "NPR",
            "name": "NPR",
            "cited_corpora": [],
        },
    ],
}

RHOWARDSTONE_URL = "https://github.com/rhowardstone/Epstein-research-data"


# ---------------------------------------------------------------------------
# _parse_github_repo_path
# ---------------------------------------------------------------------------

class TestParseGithubRepoPath(unittest.TestCase):

    def test_plain_https(self):
        self.assertEqual(
            _parse_github_repo_path("https://github.com/owner/repo"),
            "owner/repo",
        )

    def test_git_suffix_stripped(self):
        self.assertEqual(
            _parse_github_repo_path("https://github.com/owner/repo.git"),
            "owner/repo",
        )

    def test_trailing_slash_stripped(self):
        self.assertEqual(
            _parse_github_repo_path("https://github.com/owner/repo/"),
            "owner/repo",
        )

    def test_tree_path_ignored(self):
        self.assertEqual(
            _parse_github_repo_path("https://github.com/owner/repo/tree/main"),
            "owner/repo",
        )

    def test_http_url(self):
        self.assertEqual(
            _parse_github_repo_path("http://github.com/owner/repo"),
            "owner/repo",
        )

    def test_hyphenated_repo_with_git_suffix(self):
        self.assertEqual(
            _parse_github_repo_path(
                "https://github.com/rhowardstone/Epstein-research-data.git"
            ),
            "rhowardstone/Epstein-research-data",
        )

    def test_single_path_segment_raises(self):
        with self.assertRaisesRegex(ValueError, "Cannot parse"):
            _parse_github_repo_path("https://github.com/only-owner")


# ---------------------------------------------------------------------------
# check_sequence_reconciliation
# ---------------------------------------------------------------------------

class TestCheckSequenceReconciliation(unittest.TestCase):
    """Thin delegation wrapper -- verify it passes through to scheme.reconcile()."""

    def test_delegates_to_efta(self):
        result = check_sequence_reconciliation(
            corpus_numbers=["1", "2"],
            index_numbers=["1", "2", "3"],
            scheme=EFTANumber(),
        )
        self.assertIsInstance(result, ReconciliationResult)
        self.assertEqual(result.sequence_type, "EFTA")
        self.assertIn("3", result.deletion_candidates)

    def test_delegates_to_bates(self):
        result = check_sequence_reconciliation(
            corpus_numbers=["DOJ-000001"],
            index_numbers=["DOJ-000001", "DOJ-000002"],
            scheme=BatesNumber(),
        )
        self.assertEqual(result.sequence_type, "BATES")
        self.assertIn("DOJ-000002", result.deletion_candidates)

    def test_full_coverage_yields_no_deletion_candidates(self):
        nums = ["1", "2", "3"]
        result = check_sequence_reconciliation(nums, nums, EFTANumber())
        self.assertEqual(result.deletion_candidates, [])
        self.assertEqual(result.coverage_pct, 1.0)


# ---------------------------------------------------------------------------
# check_community_vetting
# ---------------------------------------------------------------------------

class TestCheckCommunityVetting(unittest.TestCase):
    """
    setUp creates a temp dir; each test writes its own endorsers JSON into it
    with a descriptive filename.  tearDown removes the dir wholesale.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write(self, name: str, data) -> Path:
        p = self.tmp_dir / name
        p.write_text(json.dumps(data))
        return p

    def test_cited_corpus_endorsed(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting(RHOWARDSTONE_URL, endorsers_path=p)
        self.assertTrue(r.endorsed)
        self.assertEqual(r.provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")
        self.assertIn("RHOWARDSTONE", r.endorsing_orgs)

    def test_unknown_repo_not_endorsed(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting("https://github.com/x/unknown", endorsers_path=p)
        self.assertFalse(r.endorsed)
        self.assertEqual(r.provenance_tag, "PROVENANCE_UNVERIFIED")
        self.assertEqual(r.endorsing_orgs, [])

    def test_git_suffix_normalised(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting(RHOWARDSTONE_URL + ".git", endorsers_path=p)
        self.assertTrue(r.endorsed)

    def test_trailing_slash_normalised(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting(RHOWARDSTONE_URL + "/", endorsers_path=p)
        self.assertTrue(r.endorsed)

    def test_case_insensitive_match(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting(RHOWARDSTONE_URL.upper(), endorsers_path=p)
        self.assertTrue(r.endorsed)

    def test_placeholder_id_never_endorses(self):
        data = {
            "_schema_version": "0.2",
            "endorsers": [{
                "id": "PLACEHOLDER",
                "cited_corpora": [RHOWARDSTONE_URL],
            }],
        }
        p = self._write("placeholder.json", data)
        r = check_community_vetting(RHOWARDSTONE_URL, endorsers_path=p)
        self.assertFalse(r.endorsed)

    def test_multiple_orgs_endorse_same_corpus(self):
        data = {
            "_schema_version": "0.2",
            "endorsers": [
                {"id": "ORG_A", "cited_corpora": ["https://github.com/owner/repo"]},
                {"id": "ORG_B", "cited_corpora": ["https://github.com/owner/repo"]},
            ],
        }
        p = self._write("multi.json", data)
        r = check_community_vetting("https://github.com/owner/repo", endorsers_path=p)
        self.assertTrue(r.endorsed)
        self.assertIn("ORG_A", r.endorsing_orgs)
        self.assertIn("ORG_B", r.endorsing_orgs)
        self.assertEqual(len(r.endorsing_orgs), 2)

    def test_empty_cited_corpora_does_not_endorse(self):
        # NPR in ENDORSERS_SIMPLE has an empty cited_corpora list
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting("https://github.com/some/repo", endorsers_path=p)
        self.assertFalse(r.endorsed)

    def test_missing_cited_corpora_key_is_safe(self):
        data = {"_schema_version": "0.2", "endorsers": [{"id": "X", "name": "X"}]}
        p = self._write("no_corpora_key.json", data)
        r = check_community_vetting("https://github.com/any/repo", endorsers_path=p)
        self.assertFalse(r.endorsed)

    def test_returns_community_vetting_result_type(self):
        p = self._write("simple.json", ENDORSERS_SIMPLE)
        r = check_community_vetting("https://github.com/x/y", endorsers_path=p)
        self.assertIsInstance(r, CommunityVettingResult)

    def test_real_trusted_endorsers_json_endorses_rhowardstone(self):
        from pipeline.corpus_evaluator import ENDORSERS_PATH
        if not ENDORSERS_PATH.exists():
            self.skipTest("trusted_endorsers.json not present in repo root")
        r = check_community_vetting(RHOWARDSTONE_URL, endorsers_path=ENDORSERS_PATH)
        self.assertTrue(r.endorsed)
        self.assertEqual(r.provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")


# ---------------------------------------------------------------------------
# check_git_integrity -- preconditions (no network, no token needed)
# ---------------------------------------------------------------------------

class TestCheckGitIntegrityPreconditions(unittest.TestCase):
    """
    Tests that are reachable without mocking PyGithub.
    setUp pops GITHUB_TOKEN so these always run in no-credentials mode;
    tearDown restores the original value.
    """

    def setUp(self):
        self._saved_token = os.environ.pop("GITHUB_TOKEN", None)

    def tearDown(self):
        if self._saved_token is not None:
            os.environ["GITHUB_TOKEN"] = self._saved_token
        else:
            os.environ.pop("GITHUB_TOKEN", None)

    def test_returns_git_integrity_result(self):
        self.assertIsInstance(
            check_git_integrity("https://github.com/s/r"), GitIntegrityResult
        )

    def test_missing_credentials_review_recommended(self):
        self.assertEqual(
            check_git_integrity("https://github.com/s/r").score,
            "REVIEW_RECOMMENDED",
        )

    def test_missing_credentials_notes_are_diagnostic(self):
        notes = check_git_integrity("https://github.com/s/r").notes
        self.assertTrue("PyGithub" in notes or "GITHUB_TOKEN" in notes)

    def test_single_segment_url_review_recommended(self):
        self.assertEqual(
            check_git_integrity("https://github.com/single-seg").score,
            "REVIEW_RECOMMENDED",
        )

    def test_score_is_valid_value(self):
        self.assertIn(
            check_git_integrity("https://github.com/s/r").score,
            {"CLEAN", "REVIEW_RECOMMENDED", "REJECTED"},
        )

    def test_notes_is_non_empty_string(self):
        r = check_git_integrity("https://github.com/s/r")
        self.assertIsInstance(r.notes, str)
        self.assertGreater(len(r.notes), 0)


# ---------------------------------------------------------------------------
# check_git_integrity -- full evaluation via mocked PyGithub
# ---------------------------------------------------------------------------

class TestCheckGitIntegrityEvaluation(unittest.TestCase):

    def test_healthy_repo_is_clean(self):
        r = _run_git_check(_good_commits(10))
        self.assertEqual(r.score, "CLEAN")
        self.assertFalse(r.forced_pushes_detected)
        self.assertFalse(r.suspicious_modification_gap)

    def test_clean_result_records_head_sha(self):
        r = _run_git_check(_good_commits(10))
        self.assertEqual(r.commit_hash_evaluated, "sha0000")

    def test_single_commit_review_recommended(self):
        r = _run_git_check([_make_commit("abc", "2026-01-15T00:00:00", "Initial commit with data")])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")
        self.assertIn("Single-commit", r.notes)

    def test_two_commits_review_recommended(self):
        r = _run_git_check([
            _make_commit("s1", "2026-01-15T00:00:00", "Add corpus data from DOJ release"),
            _make_commit("s0", "2026-01-10T00:00:00", "Initial detailed corpus setup"),
        ])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")

    def test_three_commits_review_recommended(self):
        self.assertEqual(_run_git_check(_good_commits(3)).score, "REVIEW_RECOMMENDED")

    def test_four_commits_can_be_clean(self):
        self.assertEqual(_run_git_check(_good_commits(4)).score, "CLEAN")

    def test_dormant_before_doj_release_sets_suspicious_gap_flag(self):
        r = _run_git_check(_good_commits(10, date="2024-06-01"))
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")
        self.assertTrue(r.suspicious_modification_gap)

    def test_non_descriptive_initial_commit_message(self):
        r = _run_git_check([
            _make_commit("s1", "2026-01-15T00:00:00", "Add more data files to the corpus"),
            _make_commit("s0", "2026-01-10T00:00:00", "init"),  # < 20 chars
        ])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")
        self.assertIn("non-descriptive", r.notes.lower())

    def test_null_commit_message_handled_gracefully(self):
        r = _run_git_check([
            _make_commit("s1", "2026-01-15T00:00:00", "Add comprehensive corpus data"),
            _make_commit("s0", "2026-01-10T00:00:00", None),
        ])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")

    def test_empty_commit_message_handled_gracefully(self):
        r = _run_git_check([
            _make_commit("s1", "2026-01-15T00:00:00", "Add comprehensive corpus data"),
            _make_commit("s0", "2026-01-10T00:00:00", ""),
        ])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")

    def test_force_push_event_sets_flag(self):
        evt = MagicMock()
        evt.type = "PushEvent"
        evt.payload = {"forced": True}
        evt.id = "evt-001"
        r = _run_git_check(_good_commits(10), push_events=[evt])
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")
        self.assertTrue(r.forced_pushes_detected)
        self.assertIn("Force push", r.notes)

    def test_normal_push_event_not_flagged(self):
        evt = MagicMock()
        evt.type = "PushEvent"
        evt.payload = {"forced": False}
        r = _run_git_check(_good_commits(10), push_events=[evt])
        self.assertFalse(r.forced_pushes_detected)

    def test_non_push_event_type_ignored(self):
        evt = MagicMock()
        evt.type = "IssuesEvent"
        evt.payload = {"forced": True}  # irrelevant -- wrong type
        r = _run_git_check(_good_commits(10), push_events=[evt])
        self.assertFalse(r.forced_pushes_detected)

    def test_events_api_exception_recorded_in_notes(self):
        mock_repo = MagicMock()
        mock_repo.get_commits.return_value = _good_commits(10)
        mock_repo.get_events.side_effect = RuntimeError("Events API unavailable")
        mock_gh_module = MagicMock()
        mock_gh_module.Github = MagicMock(
            return_value=MagicMock(get_repo=MagicMock(return_value=mock_repo))
        )
        mock_gh_module.GithubException = GH_EXC
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
            with patch.dict("sys.modules", {"github": mock_gh_module}):
                r = check_git_integrity("https://github.com/owner/repo")
        self.assertIn("Could not retrieve push events", r.notes)

    def test_empty_repo_is_rejected(self):
        r = _run_git_check([])
        self.assertEqual(r.score, "REJECTED")
        self.assertIn("no commits", r.notes.lower())

    def test_404_repo_not_found_is_rejected(self):
        exc = GH_EXC()
        exc.status = 404
        exc.data = {}
        r = _run_git_check([], side_effect=exc)
        self.assertEqual(r.score, "REJECTED")
        self.assertIn("not found", r.notes.lower())

    def test_non_404_api_error_is_review_recommended(self):
        exc = GH_EXC()
        exc.status = 403
        exc.data = {"message": "rate limit exceeded"}
        r = _run_git_check([], side_effect=exc)
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")

    def test_unexpected_exception_is_review_recommended(self):
        r = _run_git_check([], side_effect=RuntimeError("network timeout"))
        self.assertEqual(r.score, "REVIEW_RECOMMENDED")
        self.assertIn("Unexpected error", r.notes)


# ---------------------------------------------------------------------------
# _write_to_registry
# ---------------------------------------------------------------------------

class TestWriteToRegistry(unittest.TestCase):
    """
    setUp creates a temp dir containing a fresh empty registry file.
    All tests write into that same registry; tearDown removes the dir.
    Using setUp (not setUpClass) so each test starts from a clean empty state.
    """

    EMPTY_REGISTRY = {"_schema_version": "0.1", "corpora": []}

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.registry = self.tmp_dir / "corpus_registry.json"
        self.registry.write_text(json.dumps(self.EMPTY_REGISTRY))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _result(self, corpus_id: str = "C-001", tag: str = "PROVENANCE_UNVERIFIED"):
        return EvaluationResult(
            corpus_id=corpus_id,
            github_url="https://github.com/test/repo",
            evaluation_date="2026-03-09T00:00:00Z",
            git_integrity=GitIntegrityResult(
                score="CLEAN", notes="OK", commit_hash_evaluated="abc123"
            ),
            sequence_reconciliation=ReconciliationResult(
                sequence_type="EFTA", present_count=10, coverage_pct=1.0,
            ),
            community_vetting=CommunityVettingResult(endorsed=False),
            final_provenance_tag=tag,
            ingestion_approved=True,
            notes="Test result",
        )

    def _read(self):
        return json.loads(self.registry.read_text())

    def test_new_corpus_appended(self):
        with patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry):
            _write_to_registry(self._result())
        ids = [c["corpus_id"] for c in self._read()["corpora"]]
        self.assertIn("C-001", ids)

    def test_existing_entry_updated_not_duplicated(self):
        with patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry):
            _write_to_registry(self._result(tag="PROVENANCE_UNVERIFIED"))
            _write_to_registry(self._result(tag="PROVENANCE_COMMUNITY_VOUCHED"))
        entries = [c for c in self._read()["corpora"] if c["corpus_id"] == "C-001"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(
            entries[0]["provenance_tag_assigned"], "PROVENANCE_COMMUNITY_VOUCHED"
        )

    def test_two_distinct_ids_both_written(self):
        with patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry):
            _write_to_registry(self._result("C-001"))
            _write_to_registry(self._result("C-002"))
        ids = [c["corpus_id"] for c in self._read()["corpora"]]
        self.assertIn("C-001", ids)
        self.assertIn("C-002", ids)

    def test_provenance_tag_persisted(self):
        with patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry):
            _write_to_registry(self._result(tag="PROVENANCE_FLAGGED"))
        entry = next(
            c for c in self._read()["corpora"] if c["corpus_id"] == "C-001"
        )
        self.assertEqual(entry["provenance_tag_assigned"], "PROVENANCE_FLAGGED")

    def test_commit_hash_persisted(self):
        with patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry):
            _write_to_registry(self._result())
        entry = next(
            c for c in self._read()["corpora"] if c["corpus_id"] == "C-001"
        )
        self.assertEqual(entry["evaluated_commit_hash"], "abc123")


# ---------------------------------------------------------------------------
# evaluate_corpus
# ---------------------------------------------------------------------------

class TestEvaluateCorpus(unittest.TestCase):
    """
    setUp creates a temp dir with a fresh empty registry.
    All calls to evaluate_corpus() patch REGISTRY_PATH to that file and mock
    both check_git_integrity() and check_community_vetting() so no real I/O
    or network access occurs.
    tearDown removes the dir.

    check_community_vetting() is mocked (not ENDORSERS_PATH) because its
    default argument is bound at import time -- see module docstring.
    """

    EMPTY_REGISTRY = {"_schema_version": "0.1", "corpora": []}

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.registry = self.tmp_dir / "corpus_registry.json"
        self.registry.write_text(json.dumps(self.EMPTY_REGISTRY))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _run(
        self,
        git_score: str = "CLEAN",
        corpus_numbers=None,
        index_numbers=None,
        vetting_result=None,
        scheme=None,
    ) -> EvaluationResult:
        git_r = GitIntegrityResult(
            score=git_score, notes=f"Mocked: {git_score}", commit_hash_evaluated="abc"
        )
        vet_r = vetting_result or CommunityVettingResult(
            endorsed=False, provenance_tag="PROVENANCE_UNVERIFIED"
        )
        with (
            patch("pipeline.corpus_evaluator.check_git_integrity", return_value=git_r),
            patch("pipeline.corpus_evaluator.check_community_vetting", return_value=vet_r),
            patch("pipeline.corpus_evaluator.REGISTRY_PATH", self.registry),
        ):
            return evaluate_corpus(
                corpus_id="T-001",
                github_url="https://github.com/test/repo",
                index_numbers=index_numbers or ["1", "2", "3"],
                corpus_numbers=corpus_numbers,
                sequence_scheme=scheme or EFTANumber(),
            )

    def _read(self):
        return json.loads(self.registry.read_text())

    # Provenance tag branches

    def test_rejected_git_produces_provenance_rejected(self):
        r = self._run(git_score="REJECTED")
        self.assertEqual(r.final_provenance_tag, "PROVENANCE_REJECTED")

    def test_rejected_git_sets_ingestion_approved_false(self):
        self.assertFalse(self._run(git_score="REJECTED").ingestion_approved)

    def test_review_recommended_produces_provenance_flagged(self):
        r = self._run(git_score="REVIEW_RECOMMENDED")
        self.assertEqual(r.final_provenance_tag, "PROVENANCE_FLAGGED")

    def test_clean_full_coverage_not_flagged_or_rejected(self):
        r = self._run(git_score="CLEAN", corpus_numbers=["1", "2", "3"])
        self.assertNotIn(
            r.final_provenance_tag, {"PROVENANCE_REJECTED", "PROVENANCE_FLAGGED"}
        )

    def test_endorsed_clean_full_coverage_community_vouched(self):
        vouched = CommunityVettingResult(
            endorsed=True,
            endorsing_orgs=["TEST_ORG"],
            provenance_tag="PROVENANCE_COMMUNITY_VOUCHED",
        )
        r = self._run(
            git_score="CLEAN",
            corpus_numbers=["1", "2", "3"],
            index_numbers=["1", "2", "3"],
            vetting_result=vouched,
        )
        self.assertEqual(r.final_provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")
        self.assertTrue(r.community_vetting.endorsed)

    def test_endorsed_but_partial_coverage_not_community_vouched(self):
        # Only 1 of 10 numbers in corpus -- well below COVERAGE_THRESHOLD
        vouched = CommunityVettingResult(
            endorsed=True, provenance_tag="PROVENANCE_COMMUNITY_VOUCHED"
        )
        r = self._run(
            git_score="CLEAN",
            corpus_numbers=["1"],
            index_numbers=[str(i) for i in range(10)],
            vetting_result=vouched,
        )
        self.assertNotEqual(r.final_provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")

    # Reconciliation behaviour

    def test_no_corpus_numbers_skips_reconciliation(self):
        r = self._run(git_score="CLEAN", corpus_numbers=None)
        self.assertEqual(r.sequence_reconciliation.coverage_pct, 0.0)
        self.assertTrue(r.sequence_reconciliation.partial_coverage)

    def test_defaults_to_efta_scheme(self):
        r = self._run(scheme=None)
        self.assertEqual(r.sequence_reconciliation.sequence_type, "EFTA")

    def test_bates_scheme_used_when_passed(self):
        r = self._run(
            corpus_numbers=["DOJ-000001"],
            index_numbers=["DOJ-000001"],
            scheme=BatesNumber(),
        )
        self.assertEqual(r.sequence_reconciliation.sequence_type, "BATES")

    # Result fields

    def test_evaluation_date_is_set(self):
        self.assertTrue(self._run().evaluation_date)

    def test_corpus_id_preserved(self):
        self.assertEqual(self._run().corpus_id, "T-001")

    def test_github_url_preserved(self):
        self.assertEqual(
            self._run().github_url, "https://github.com/test/repo"
        )

    def test_notes_contain_coverage_info(self):
        r = self._run(git_score="CLEAN", corpus_numbers=["1", "2", "3"])
        self.assertTrue("Coverage" in r.notes or "coverage" in r.notes)

    # Registry writes

    def test_approved_result_written_to_registry(self):
        self._run(git_score="CLEAN")
        ids = [c["corpus_id"] for c in self._read()["corpora"]]
        self.assertIn("T-001", ids)

    def test_rejected_result_not_written_to_registry(self):
        # REJECTED gates at Check 1 and returns before _write_to_registry
        self._run(git_score="REJECTED")
        ids = [c["corpus_id"] for c in self._read()["corpora"]]
        self.assertNotIn("T-001", ids)


# ---------------------------------------------------------------------------
# Dataclass field completeness
# ---------------------------------------------------------------------------

class TestGitIntegrityResultFields(unittest.TestCase):

    def test_required_fields(self):
        r = GitIntegrityResult(score="CLEAN", notes="ok")
        self.assertEqual(r.score, "CLEAN")
        self.assertEqual(r.notes, "ok")

    def test_optional_fields_default(self):
        r = GitIntegrityResult(score="CLEAN", notes="ok")
        self.assertIsNone(r.commit_hash_evaluated)
        self.assertFalse(r.forced_pushes_detected)
        self.assertFalse(r.suspicious_modification_gap)

    def test_all_fields_explicit(self):
        r = GitIntegrityResult(
            score="REVIEW_RECOMMENDED",
            notes="thin history",
            commit_hash_evaluated="abc1234",
            forced_pushes_detected=True,
            suspicious_modification_gap=True,
        )
        self.assertEqual(r.commit_hash_evaluated, "abc1234")
        self.assertTrue(r.forced_pushes_detected)
        self.assertTrue(r.suspicious_modification_gap)


class TestCommunityVettingResultFields(unittest.TestCase):

    def test_defaults(self):
        r = CommunityVettingResult()
        self.assertFalse(r.endorsed)
        self.assertEqual(r.endorsing_orgs, [])
        self.assertEqual(r.provenance_tag, "PROVENANCE_UNVERIFIED")

    def test_endorsed(self):
        r = CommunityVettingResult(
            endorsed=True,
            endorsing_orgs=["NPR", "MUCKROCK"],
            provenance_tag="PROVENANCE_COMMUNITY_VOUCHED",
        )
        self.assertTrue(r.endorsed)
        self.assertEqual(len(r.endorsing_orgs), 2)


class TestEvaluationResultFields(unittest.TestCase):

    def test_all_fields_accessible(self):
        r = EvaluationResult(
            corpus_id="X",
            github_url="https://github.com/x/y",
            evaluation_date="2026-03-09",
            git_integrity=GitIntegrityResult(score="CLEAN", notes="ok"),
            sequence_reconciliation=ReconciliationResult(sequence_type="EFTA"),
            community_vetting=CommunityVettingResult(),
            final_provenance_tag="PROVENANCE_UNVERIFIED",
            ingestion_approved=True,
            notes="test",
        )
        self.assertEqual(r.corpus_id, "X")
        self.assertTrue(r.ingestion_approved)
        self.assertEqual(r.final_provenance_tag, "PROVENANCE_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
