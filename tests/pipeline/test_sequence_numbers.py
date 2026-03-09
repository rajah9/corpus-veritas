"""
tests/pipeline/test_sequence_numbers.py

Unit tests for pipeline/sequence_numbers.py.

Covers paths not reached by tests/integration/test_corpus_evaluator.py, which
already tests BatesNumber basics, EFTANumber basics (frozenset path),
reconcile() logic, and the SequenceNumber ABC contract.

New paths covered here
----------------------
BatesNumber
  sort_key()  -- bare-numeric branch -> ('', n)
                 non-digit suffix fallback -> (value, 0)
  reconcile() -- unindexed_count, exact coverage-threshold boundary,
                 sorted missing_numbers and deletion_candidates

EFTANumber.from_mapping_file()
  Variant A   -- list-of-objects JSON
  Variant B   -- dict-keyed-by-dataset JSON
  Alt fields  -- efta_start/end, start/end, dataset_id, name, url alias
  Error cases -- wrong root type, empty list/dict, no parseable entries,
                 entries without range fields, missing DS9 (warning not error)

EFTANumber.gap_is_expected()  -- range-based DS9 path
  boundary values, frozenset checked before range, non-numeric / empty / zero

EFTANumber.dataset_for_number()  -- all range boundaries and gaps between datasets
EFTANumber.doj_url_for_number()  -- template path, fallback path, no mapping loaded
EFTANumber.describe_number()     -- with and without mapping
EFTANumber.ds9_gap_count         -- 0 when range-based (no frozenset populated)
EFTANumber.sort_key()            -- invalid input -> 0 fallback

Temp-file strategy
------------------
* Classes that reuse one file across all tests: setUpClass creates a temp dir
  and the shared file inside it; tearDownClass removes the whole dir.

* Classes where every test needs a different file structure: setUp creates a
  fresh temp dir; tearDown removes it wholesale.  A _write(name, data) helper
  writes named files into that dir so each test gets a clean, distinct path.

Both approaches guarantee cleanup runs even when a test fails mid-way.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.sequence_numbers import (
    BatesNumber,
    COVERAGE_THRESHOLD,
    EFTANumber,
)


# ---------------------------------------------------------------------------
# BatesNumber -- sort_key edge cases
# ---------------------------------------------------------------------------

class TestBatesNumberSortKeyEdgeCases(unittest.TestCase):
    """
    The integration suite covers the prefixed case DOJ-000010 -> ('DOJ', 10).
    We add the two remaining branches:
      bare numeric -> ('', n)
      non-digit suffix -> (value, 0) fallback
    """

    def setUp(self):
        self.b = BatesNumber()

    def test_bare_numeric_returns_empty_prefix_and_int(self):
        self.assertEqual(self.b.sort_key("000001"), ("", 1))

    def test_bare_numeric_sorts_correctly(self):
        nums = ["000010", "000002", "000100"]
        self.assertEqual(
            sorted(nums, key=self.b.sort_key),
            ["000002", "000010", "000100"],
        )

    def test_hyphen_with_non_digit_suffix_uses_fallback(self):
        # rsplit('-', 1) -> parts[1] = "ABCD", not isdigit() -> (value, 0)
        self.assertEqual(self.b.sort_key("DOJ-ABCD"), ("DOJ-ABCD", 0))

    def test_plain_alpha_string_uses_fallback(self):
        self.assertEqual(self.b.sort_key("NODASH"), ("NODASH", 0))

    def test_mixed_domains_do_not_crash(self):
        nums = ["000010", "DOJ-000002", "EDNY-000100"]
        self.assertEqual(len(sorted(nums, key=self.b.sort_key)), 3)


# ---------------------------------------------------------------------------
# BatesNumber -- reconcile() unindexed items and coverage threshold
# ---------------------------------------------------------------------------

class TestBatesReconcileAdditional(unittest.TestCase):
    """
    The integration suite covers full coverage and empty corpus/index.
    We add unindexed items and the exact coverage-threshold boundary.
    """

    def setUp(self):
        self.b = BatesNumber()

    def test_unindexed_items_counted(self):
        result = self.b.reconcile(
            corpus_numbers=["DOJ-000001", "DOJ-000002", "DOJ-EXTRA-9999"],
            index_numbers=["DOJ-000001", "DOJ-000002"],
        )
        self.assertEqual(result.unindexed_count, 1)

    def test_exactly_at_threshold_is_not_partial(self):
        n, k = 10, int(10 * COVERAGE_THRESHOLD)
        index = [f"DOJ-{i:06d}" for i in range(n)]
        self.assertFalse(self.b.reconcile(index[:k], index).partial_coverage)

    def test_one_below_threshold_is_partial(self):
        n, k = 100, int(100 * COVERAGE_THRESHOLD) - 1
        index = [f"DOJ-{i:06d}" for i in range(n)]
        self.assertTrue(self.b.reconcile(index[:k], index).partial_coverage)

    def test_missing_numbers_are_sorted(self):
        index = ["DOJ-000010", "DOJ-000002", "DOJ-000030"]
        result = self.b.reconcile(["DOJ-000010"], index)
        self.assertEqual(
            result.missing_numbers,
            sorted(result.missing_numbers, key=self.b.sort_key),
        )

    def test_deletion_candidates_are_sorted(self):
        index = ["DOJ-000010", "DOJ-000002", "DOJ-000030"]
        result = self.b.reconcile([], index)
        self.assertEqual(
            result.deletion_candidates,
            sorted(result.deletion_candidates, key=self.b.sort_key),
        )


# ---------------------------------------------------------------------------
# EFTANumber.from_mapping_file -- Variant A (list of objects)
# ---------------------------------------------------------------------------

class TestEFTAFromMappingFileVariantA(unittest.TestCase):
    """
    Primary JSON structure: a list of dataset objects.
    setUpClass creates a temp dir with a single shared mapping file.
    DS10 intentionally has no url_template so both the template and fallback
    URL-construction paths are exercised.
    tearDownClass removes the entire temp dir.
    """

    MAPPING = [
        {
            "dataset": "DS1",
            "first_efta": 1,
            "last_efta": 3158,
            "url_template": (
                "https://www.justice.gov/epstein/files/DataSet 1/EFTA{:08d}.pdf"
            ),
        },
        {
            "dataset": "DS9",
            "first_efta": 39025,
            "last_efta": 1262781,
            "url_template": (
                "https://www.justice.gov/epstein/files/DataSet 9/EFTA{:08d}.pdf"
            ),
        },
        {
            "dataset": "DS10",
            "first_efta": 1262782,
            "last_efta": 2731785,
            # no url_template -- exercises fallback construction path
        },
    ]

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        p = cls.tmp_dir / "mapping.json"
        p.write_text(json.dumps(cls.MAPPING))
        cls.efta = EFTANumber.from_mapping_file(p)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    # dataset_for_number -- range boundaries

    def test_dataset_count(self):
        self.assertEqual(self.efta.dataset_count, 3)

    def test_ds1_first_boundary(self):
        self.assertEqual(self.efta.dataset_for_number(1), "DS1")

    def test_ds1_last_boundary(self):
        self.assertEqual(self.efta.dataset_for_number(3158), "DS1")

    def test_ds9_first_boundary(self):
        self.assertEqual(self.efta.dataset_for_number(39025), "DS9")

    def test_ds9_last_boundary(self):
        self.assertEqual(self.efta.dataset_for_number(1262781), "DS9")

    def test_ds10_first_boundary(self):
        self.assertEqual(self.efta.dataset_for_number(1262782), "DS10")

    def test_gap_between_ds1_and_ds9_returns_none(self):
        self.assertIsNone(self.efta.dataset_for_number(3159))

    def test_above_all_ranges_returns_none(self):
        self.assertIsNone(self.efta.dataset_for_number(9_999_999))

    def test_zero_returns_none(self):
        self.assertIsNone(self.efta.dataset_for_number(0))

    # gap_is_expected -- DS9 range-based path

    def test_ds9_gap_expected_at_first(self):
        self.assertTrue(self.efta.gap_is_expected("39025"))

    def test_ds9_gap_expected_in_middle(self):
        self.assertTrue(self.efta.gap_is_expected("700000"))

    def test_ds9_gap_expected_at_last(self):
        self.assertTrue(self.efta.gap_is_expected("1262781"))

    def test_ds1_not_expected_gap(self):
        self.assertFalse(self.efta.gap_is_expected("1000"))

    def test_ds10_not_expected_gap(self):
        self.assertFalse(self.efta.gap_is_expected("1262782"))

    def test_between_datasets_not_expected_gap(self):
        self.assertFalse(self.efta.gap_is_expected("3159"))

    # doj_url_for_number -- template path

    def test_template_url_for_ds1(self):
        url = self.efta.doj_url_for_number(1000)
        self.assertIsNotNone(url)
        self.assertIn("00001000", url)
        self.assertIn("DataSet 1", url)

    def test_template_url_for_ds9(self):
        url = self.efta.doj_url_for_number(39025)
        self.assertIsNotNone(url)
        self.assertIn("DataSet 9", url)

    # doj_url_for_number -- fallback path (DS10 has no template)

    def test_fallback_url_for_ds10(self):
        url = self.efta.doj_url_for_number(1300000)
        self.assertIsNotNone(url)
        self.assertIn("01300000", url)
        self.assertIn("10", url)

    def test_url_starts_with_https(self):
        self.assertTrue(self.efta.doj_url_for_number(1000).startswith("https://"))

    def test_url_ends_with_pdf(self):
        self.assertTrue(self.efta.doj_url_for_number(1000).endswith(".pdf"))

    def test_out_of_range_url_is_none(self):
        self.assertIsNone(self.efta.doj_url_for_number(9_999_999))

    # ds9_gap_count -- range-based never populates the frozenset

    def test_ds9_gap_count_is_zero_when_range_based(self):
        self.assertEqual(self.efta.ds9_gap_count, 0)

    # describe_number

    def test_describe_number_includes_dataset_and_url(self):
        desc = self.efta.describe_number("1000")
        self.assertIn("DS1", desc)
        self.assertIn("00001000", desc)

    def test_describe_number_out_of_range_falls_back(self):
        desc = self.efta.describe_number("9999999")
        self.assertIn("9999999", desc)

    def test_scheme_name(self):
        self.assertEqual(self.efta.scheme_name, "EFTA")


# ---------------------------------------------------------------------------
# EFTANumber.from_mapping_file -- Variant B (dict keyed by dataset ID)
# ---------------------------------------------------------------------------

class TestEFTAFromMappingFileVariantB(unittest.TestCase):
    """
    Alternate JSON structure: a dict keyed by dataset ID.
    Underscore-prefixed keys are schema metadata and must be ignored.

    setUpClass creates a temp dir holding the shared mapping file.
    test_non_dict_entry_ignored writes an additional file into the same dir.
    tearDownClass removes the whole dir.
    """

    MAPPING = {
        "DS1":  {"first_efta": 1,       "last_efta": 3158},
        "DS9":  {"first_efta": 39025,   "last_efta": 1262781},
        "DS10": {"first_efta": 1262782, "last_efta": 2731785},
        "_schema": "metadata value -- must be ignored",
    }

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        p = cls.tmp_dir / "mapping.json"
        p.write_text(json.dumps(cls.MAPPING))
        cls.efta = EFTANumber.from_mapping_file(p)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_dataset_count_excludes_underscore_keys(self):
        self.assertEqual(self.efta.dataset_count, 3)

    def test_ds9_gap_expected(self):
        self.assertTrue(self.efta.gap_is_expected("500000"))

    def test_ds1_range_loaded(self):
        self.assertEqual(self.efta.dataset_for_number(1), "DS1")

    def test_non_dict_entry_ignored(self):
        # A string value under a dataset key is not a dataset object -- skip it.
        # Written into the class temp dir; removed by tearDownClass.
        data = {
            "DS1": {"first_efta": 1, "last_efta": 100},
            "COMMENT": "string value, not a dataset object",
        }
        p = self.tmp_dir / "non_dict.json"
        p.write_text(json.dumps(data))
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_count, 1)


# ---------------------------------------------------------------------------
# EFTANumber.from_mapping_file -- alternate field-name aliases
# ---------------------------------------------------------------------------

class TestEFTAFromMappingFileAltFieldNames(unittest.TestCase):
    """
    from_mapping_file() accepts several aliases for the range and URL fields.
    Each test writes a differently-structured file into the per-test temp dir
    created in setUp and removed wholesale in tearDown.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write(self, name: str, data) -> Path:
        """Write JSON data to a named file in this test's temp dir."""
        p = self.tmp_dir / name
        p.write_text(json.dumps(data))
        return p

    def test_efta_start_and_efta_end(self):
        p = self._write("efta_start_end.json", [
            {"dataset": "DS1", "efta_start": 1,     "efta_end": 100},
            {"dataset": "DS9", "efta_start": 39025, "efta_end": 1262781},
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_for_number(50), "DS1")
        self.assertTrue(efta.gap_is_expected("700000"))

    def test_start_and_end(self):
        p = self._write("start_end.json", [
            {"dataset": "DS1", "start": 1,     "end": 100},
            {"dataset": "DS9", "start": 39025, "end": 1262781},
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_for_number(1), "DS1")

    def test_dataset_id_field(self):
        p = self._write("dataset_id.json", [
            {"dataset_id": "DS1", "first_efta": 1, "last_efta": 100},
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_for_number(50), "DS1")

    def test_name_field(self):
        p = self._write("name_field.json", [
            {"name": "DS1", "first_efta": 1, "last_efta": 100},
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_for_number(1), "DS1")

    def test_url_alias_for_url_template(self):
        p = self._write("url_alias.json", [{
            "dataset": "DS1",
            "first_efta": 1,
            "last_efta": 100,
            "url": "https://example.com/EFTA{:08d}.pdf",
        }])
        efta = EFTANumber.from_mapping_file(p)
        self.assertIn("00000042", efta.doj_url_for_number(42))

    def test_entry_missing_range_fields_is_skipped(self):
        p = self._write("partial.json", [
            {"dataset": "DS1", "first_efta": 1, "last_efta": 100},
            {"dataset": "DS_BAD"},  # no range fields -- skipped with a warning
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_count, 1)

    def test_missing_ds9_does_not_raise(self):
        """A mapping without DS9 is valid. Gap classification becomes inactive."""
        p = self._write("no_ds9.json", [
            {"dataset": "DS1", "first_efta": 1, "last_efta": 100},
        ])
        efta = EFTANumber.from_mapping_file(p)
        self.assertEqual(efta.dataset_count, 1)
        # Without DS9 boundaries, no number is classified as an expected gap
        self.assertFalse(efta.gap_is_expected("700000"))


# ---------------------------------------------------------------------------
# EFTANumber.from_mapping_file -- error cases
# ---------------------------------------------------------------------------

class TestEFTAFromMappingFileErrors(unittest.TestCase):
    """
    from_mapping_file() must raise ValueError for unrecognized root structures
    and for files that yield no usable dataset ranges.

    Each test writes a malformed file into the per-test temp dir created in
    setUp and removed wholesale in tearDown.
    """

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _write(self, name: str, data) -> Path:
        """Write JSON data to a named file in this test's temp dir."""
        p = self.tmp_dir / name
        p.write_text(json.dumps(data))
        return p

    def test_string_at_root_raises_unrecognized(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            EFTANumber.from_mapping_file(self._write("str.json", "not_a_list_or_dict"))

    def test_integer_at_root_raises(self):
        with self.assertRaises(ValueError):
            EFTANumber.from_mapping_file(self._write("int.json", 42))

    def test_empty_list_raises_no_dataset_ranges(self):
        with self.assertRaisesRegex(ValueError, "No dataset ranges"):
            EFTANumber.from_mapping_file(self._write("empty_list.json", []))

    def test_empty_dict_raises_no_dataset_ranges(self):
        with self.assertRaisesRegex(ValueError, "No dataset ranges"):
            EFTANumber.from_mapping_file(self._write("empty_dict.json", {}))

    def test_dict_with_only_metadata_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "No dataset ranges"):
            EFTANumber.from_mapping_file(
                self._write("meta_only.json",
                            {"_schema_version": "0.1", "_note": "test"})
            )

    def test_list_with_all_entries_missing_range_raises(self):
        with self.assertRaisesRegex(ValueError, "No dataset ranges"):
            EFTANumber.from_mapping_file(
                self._write("no_ranges.json",
                            [{"dataset": "DS1"}, {"dataset": "DS2"}])
            )


# ---------------------------------------------------------------------------
# EFTANumber.gap_is_expected -- range-based DS9 path
# ---------------------------------------------------------------------------

class TestEFTAGapIsExpectedRangeBased(unittest.TestCase):
    """
    The integration suite tests the frozenset path (ds9_gap_numbers constructor).
    We test the range-based path: when dataset_ranges contains DS9,
    gap_is_expected() uses range membership instead of set lookup.
    """

    def setUp(self):
        self.efta = EFTANumber(dataset_ranges={"DS9": (39025, 1262781)})

    def test_default_constructor_no_gaps_expected(self):
        self.assertFalse(EFTANumber().gap_is_expected("700000"))

    def test_first_boundary_included(self):
        self.assertTrue(self.efta.gap_is_expected("39025"))

    def test_last_boundary_included(self):
        self.assertTrue(self.efta.gap_is_expected("1262781"))

    def test_one_below_first_boundary_excluded(self):
        self.assertFalse(self.efta.gap_is_expected("39024"))

    def test_one_above_last_boundary_excluded(self):
        self.assertFalse(self.efta.gap_is_expected("1262782"))

    def test_frozenset_checked_before_range(self):
        # "42" is in the frozenset but outside DS9's range -- still returns True
        efta = EFTANumber(
            ds9_gap_numbers=frozenset({"42"}),
            dataset_ranges={"DS9": (39025, 1262781)},
        )
        self.assertTrue(efta.gap_is_expected("42"))      # frozenset path
        self.assertTrue(efta.gap_is_expected("700000"))  # range path

    def test_non_numeric_string_returns_false(self):
        self.assertFalse(self.efta.gap_is_expected("not-a-number"))

    def test_empty_string_returns_false(self):
        self.assertFalse(self.efta.gap_is_expected(""))

    def test_zero_returns_false(self):
        self.assertFalse(self.efta.gap_is_expected("0"))

    def test_reconcile_routes_ds9_to_expected_not_deletion(self):
        """End-to-end: DS9 gaps -> expected_gap_numbers, not deletion_candidates."""
        result = self.efta.reconcile(["1000"], ["1000", "700000", "700001"])
        self.assertIn("700000", result.expected_gap_numbers)
        self.assertIn("700001", result.expected_gap_numbers)
        self.assertNotIn("700000", result.deletion_candidates)
        self.assertNotIn("700001", result.deletion_candidates)

    def test_reconcile_non_ds9_gap_is_deletion_candidate(self):
        result = self.efta.reconcile(["1000", "1002"], ["1000", "1001", "1002"])
        self.assertIn("1001", result.deletion_candidates)
        self.assertNotIn("1001", result.expected_gap_numbers)


# ---------------------------------------------------------------------------
# EFTANumber -- sort_key invalid input
# ---------------------------------------------------------------------------

class TestEFTASortKeyInvalidInput(unittest.TestCase):

    def test_non_numeric_string_returns_zero(self):
        self.assertEqual(EFTANumber().sort_key("not-a-number"), 0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(EFTANumber().sort_key(""), 0)


# ---------------------------------------------------------------------------
# EFTANumber.describe_number
# ---------------------------------------------------------------------------

class TestEFTADescribeNumber(unittest.TestCase):

    def test_without_mapping_contains_number_and_scheme(self):
        desc = EFTANumber().describe_number("12345")
        self.assertIn("12345", desc)
        self.assertIn("EFTA", desc)

    def test_with_dataset_ranges_contains_number(self):
        efta = EFTANumber(dataset_ranges={"DS1": (1, 3158)})
        self.assertIn("1000", efta.describe_number("1000"))

    def test_non_numeric_value_does_not_raise(self):
        desc = EFTANumber().describe_number("BADVALUE")
        self.assertIn("BADVALUE", desc)


if __name__ == "__main__":
    unittest.main()
