"""
tests/pipeline/test_manifest_loader.py

Unit tests for pipeline/manifest_loader.py.

Coverage targets
----------------
_detect_column()        -- matches known candidate names, case-insensitive,
                           returns None when absent
_normalise_efta()       -- strips EFTA- prefix, strips leading zeros,
                           rejects non-numeric, handles zero
load_manifest_from_csv()-- parses minimal CSV, detects EFTA column,
                           normalises values, skips unparseable rows,
                           counts duplicates, raises on missing EFTA column,
                           optional columns absent handled gracefully
ManifestLoadResult      -- record_count, efta_numbers set populated,
                           to_json round-trip
load_manifest_from_file()-- reads file, delegates to load_manifest_from_csv
load_manifest_from_s3() -- get_object called, decodes correctly,
                           empty bucket raises ValueError,
                           S3 failure raises RuntimeError
save_normalised_manifest_to_s3()
                        -- put_object called, correct key/content-type,
                           empty bucket raises, failure raises RuntimeError
"""

from __future__ import annotations

import io
import json
import textwrap
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from pipeline.manifest_loader import (
    ManifestLoadResult,
    ManifestRecord,
    _detect_column,
    _normalise_efta,
    load_manifest_from_csv,
    load_manifest_from_file,
    load_manifest_from_s3,
    save_normalised_manifest_to_s3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv(*rows: str) -> str:
    return "\n".join(rows)


def _simple_csv() -> str:
    return _csv(
        "efta_number,dataset,title,url",
        "1234567,DS01,FBI 302 Interview,https://example.com/1",
        "1234568,DS01,Court Filing,https://example.com/2",
        "1234569,DS09,Exhibit A,https://example.com/3",
    )


def _mock_s3(csv_text: str) -> MagicMock:
    client = MagicMock()
    client.get_object.return_value = {
        "Body": io.BytesIO(csv_text.encode("utf-8"))
    }
    client.put_object.return_value = {}
    return client


# ===========================================================================
# _detect_column
# ===========================================================================

class TestDetectColumn(unittest.TestCase):

    def test_exact_match(self):
        self.assertEqual(_detect_column(["efta_number", "title"], ["efta_number"]), "efta_number")

    def test_case_insensitive(self):
        self.assertEqual(_detect_column(["EFTA_NUMBER", "title"], ["efta_number"]), "EFTA_NUMBER")

    def test_second_candidate_matched(self):
        self.assertEqual(_detect_column(["efta_num", "title"], ["efta_number", "efta_num"]), "efta_num")

    def test_returns_none_when_absent(self):
        self.assertIsNone(_detect_column(["col1", "col2"], ["efta_number"]))

    def test_empty_fieldnames(self):
        self.assertIsNone(_detect_column([], ["efta_number"]))


# ===========================================================================
# _normalise_efta
# ===========================================================================

class TestNormaliseEfta(unittest.TestCase):

    def test_plain_digits_returned(self):
        self.assertEqual(_normalise_efta("1234567"), "1234567")

    def test_efta_prefix_stripped(self):
        self.assertEqual(_normalise_efta("EFTA-1234567"), "1234567")

    def test_leading_zeros_stripped(self):
        self.assertEqual(_normalise_efta("0001234567"), "1234567")

    def test_zero_returns_zero(self):
        self.assertEqual(_normalise_efta("0"), "0")

    def test_non_numeric_returns_none(self):
        self.assertIsNone(_normalise_efta("DOJ-000042"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_normalise_efta(""))

    def test_whitespace_stripped(self):
        self.assertEqual(_normalise_efta("  1234567  "), "1234567")

    def test_efta_lowercase_prefix_stripped(self):
        self.assertEqual(_normalise_efta("efta-9999"), "9999")


# ===========================================================================
# load_manifest_from_csv
# ===========================================================================

class TestLoadManifestFromCsv(unittest.TestCase):

    def test_basic_parse(self):
        result = load_manifest_from_csv(_simple_csv(), "2026-03-01")
        self.assertEqual(result.record_count, 3)

    def test_efta_numbers_set_populated(self):
        result = load_manifest_from_csv(_simple_csv(), "2026-03-01")
        self.assertIn("1234567", result.efta_numbers)
        self.assertIn("1234569", result.efta_numbers)

    def test_release_version_stored(self):
        result = load_manifest_from_csv(_simple_csv(), "2026-03-01")
        self.assertEqual(result.release_version, "2026-03-01")

    def test_dataset_parsed(self):
        result = load_manifest_from_csv(_simple_csv(), "v1")
        self.assertEqual(result.records[0].dataset, "DS01")

    def test_title_parsed(self):
        result = load_manifest_from_csv(_simple_csv(), "v1")
        self.assertEqual(result.records[0].title, "FBI 302 Interview")

    def test_url_parsed(self):
        result = load_manifest_from_csv(_simple_csv(), "v1")
        self.assertIn("example.com", result.records[0].url)

    def test_unparseable_row_skipped(self):
        csv = _csv(
            "efta_number,dataset",
            "1234567,DS01",
            "NOT_A_NUMBER,DS01",
            "1234568,DS01",
        )
        result = load_manifest_from_csv(csv, "v1")
        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.skipped_rows, 1)

    def test_duplicate_counted(self):
        csv = _csv(
            "efta_number,dataset",
            "1234567,DS01",
            "1234567,DS01",
        )
        result = load_manifest_from_csv(csv, "v1")
        self.assertEqual(result.duplicate_count, 1)

    def test_missing_efta_column_raises(self):
        csv = _csv("title,url", "Some doc,https://example.com")
        with self.assertRaises(ValueError) as ctx:
            load_manifest_from_csv(csv, "v1")
        self.assertIn("EFTA number column", str(ctx.exception))

    def test_missing_optional_columns_handled(self):
        csv = _csv("efta_number", "1234567", "1234568")
        result = load_manifest_from_csv(csv, "v1")
        self.assertEqual(result.records[0].dataset, "")
        self.assertEqual(result.records[0].title, "")

    def test_efta_prefix_in_csv_normalised(self):
        csv = _csv("efta_number,dataset", "EFTA-1234567,DS01")
        result = load_manifest_from_csv(csv, "v1")
        self.assertIn("1234567", result.efta_numbers)

    def test_alternate_column_name_detected(self):
        csv = _csv("number,dataset", "1234567,DS01")
        result = load_manifest_from_csv(csv, "v1")
        self.assertEqual(result.record_count, 1)


# ===========================================================================
# ManifestLoadResult
# ===========================================================================

class TestManifestLoadResult(unittest.TestCase):

    def test_to_json_round_trip(self):
        result = load_manifest_from_csv(_simple_csv(), "2026-03-01")
        json_str = result.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["release_version"], "2026-03-01")
        self.assertEqual(data["record_count"], 3)
        self.assertEqual(len(data["records"]), 3)

    def test_record_count_property(self):
        result = load_manifest_from_csv(_simple_csv(), "v1")
        self.assertEqual(result.record_count, len(result.records))


# ===========================================================================
# load_manifest_from_file
# ===========================================================================

class TestLoadManifestFromFile(unittest.TestCase):

    def test_loads_csv_file(self):
        with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(_simple_csv())
            path = Path(f.name)
        try:
            result = load_manifest_from_file(path, "v1")
            self.assertEqual(result.record_count, 3)
        finally:
            path.unlink()


# ===========================================================================
# load_manifest_from_s3
# ===========================================================================

class TestLoadManifestFromS3(unittest.TestCase):

    def test_get_object_called(self):
        s3 = _mock_s3(_simple_csv())
        load_manifest_from_s3("manifests/v1/manifest.csv", "v1",
                               s3_client=s3, bucket_name="bucket")
        s3.get_object.assert_called_once()

    def test_correct_bucket_and_key(self):
        s3 = _mock_s3(_simple_csv())
        load_manifest_from_s3("manifests/v1/manifest.csv", "v1",
                               s3_client=s3, bucket_name="my-bucket")
        self.assertEqual(s3.get_object.call_args.kwargs["Bucket"], "my-bucket")
        self.assertEqual(s3.get_object.call_args.kwargs["Key"], "manifests/v1/manifest.csv")

    def test_empty_bucket_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_manifest_from_s3("key", "v1", s3_client=_mock_s3(""), bucket_name="")

    def test_s3_failure_raises_runtime_error(self):
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError) as ctx:
            load_manifest_from_s3("key", "v1", s3_client=s3, bucket_name="bucket")
        self.assertIn("Failed to load manifest", str(ctx.exception))


# ===========================================================================
# save_normalised_manifest_to_s3
# ===========================================================================

class TestSaveNormalisedManifestToS3(unittest.TestCase):

    def test_put_object_called(self):
        s3 = _mock_s3("")
        result = load_manifest_from_csv(_simple_csv(), "v1")
        save_normalised_manifest_to_s3(result, "manifests/v1/manifest.json",
                                       s3_client=s3, bucket_name="bucket")
        s3.put_object.assert_called_once()

    def test_content_type_is_json(self):
        s3 = _mock_s3("")
        result = load_manifest_from_csv(_simple_csv(), "v1")
        save_normalised_manifest_to_s3(result, "key.json",
                                       s3_client=s3, bucket_name="bucket")
        self.assertEqual(
            s3.put_object.call_args.kwargs["ContentType"], "application/json"
        )

    def test_empty_bucket_raises(self):
        with self.assertRaises(ValueError):
            save_normalised_manifest_to_s3(
                load_manifest_from_csv(_simple_csv(), "v1"),
                "key", s3_client=_mock_s3(""), bucket_name="",
            )

    def test_s3_failure_raises_runtime_error(self):
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError):
            save_normalised_manifest_to_s3(
                load_manifest_from_csv(_simple_csv(), "v1"),
                "key", s3_client=s3, bucket_name="bucket",
            )


if __name__ == "__main__":
    unittest.main()
