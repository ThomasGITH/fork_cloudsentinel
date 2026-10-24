import tempfile
import unittest
from pathlib import Path

from data_ingestion.catalogue.csv_import import import_csv_dataset
from data_ingestion.catalogue.materialization import materialize_evaluation_bundle
from data_ingestion.catalogue.repository import FileCatalogueRepository
from data_ingestion.catalogue.validation import CatalogueValidationError


CSV = b"""timestamp,cpu,memory,label
2026-10-24T10:00:00Z,0.1,100,0
2026-10-24T10:01:00Z,0.2,101,0
2026-10-24T10:02:00Z,0.3,102,0
2026-10-24T10:03:00Z,0.9,140,1
2026-10-24T10:04:00Z,0.8,135,1
2026-10-24T10:05:00Z,0.2,104,0
"""


def metadata():
    return {
        "display_name": "Uploaded metrics",
        "description": "External evaluation context",
        "purpose": ["training", "evaluation"],
        "application": "external-study",
        "namespace": "benchmark",
        "timestamp_column": "timestamp",
        "label_column": "label",
        "train_split_percent": 50,
        "workload_context": {
            "workload_intensity": "High",
            "dominant_workload_characteristic": "High traffic",
            "anomaly_scenario": "Other",
            "anomaly_scenario_detail": "Traffic burst",
        },
    }


class CsvDatasetImportTests(unittest.TestCase):
    def test_import_creates_checksums_labels_partition_and_evaluation_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = FileCatalogueRepository(directory)
            detail = import_csv_dataset(repository, metadata(), CSV, "metrics.csv")
            version = detail["version"]
            self.assertEqual(version["status"], "available")
            self.assertEqual(version["source"]["type"], "uploaded_csv")
            self.assertEqual(version["technical_schema"]["feature_order"], ["cpu", "memory"])
            self.assertTrue(version["ground_truth"]["labels_available"])
            self.assertEqual(version["partition"]["mode"], "time_range")
            self.assertTrue(detail["usages"]["labeled_evaluation"]["supported"])
            bundle, manifest = materialize_evaluation_bundle(
                repository,
                Path(directory) / "legacy",
                detail["dataset_id"],
                1,
                version["partition"]["partition_id"],
            )
            self.assertTrue(bundle.is_file())
            self.assertEqual(manifest["partition_checksum"], version["partition"]["checksum"])

    def test_import_rejects_non_numeric_features(self):
        invalid = CSV.replace(b"0.9,140,1", b"busy,140,1")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CatalogueValidationError):
                import_csv_dataset(FileCatalogueRepository(directory), metadata(), invalid, "bad.csv")


if __name__ == "__main__":
    unittest.main()
