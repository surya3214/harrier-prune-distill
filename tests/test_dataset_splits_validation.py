from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import load_sts_datasets_config
from scripts.validate_dataset_splits import collect_sts_checks, verify_check, Check


class DatasetSplitValidationTests(unittest.TestCase):
    def test_sts22_sources_only_use_available_subsets(self) -> None:
        sts_cfg = load_sts_datasets_config()
        sts22 = sts_cfg["STS22.v2"]["sources"]
        subsets = {source["hf_subset"] for source in sts22}
        allowed = {"en", "ar", "de", "es", "fr", "it", "pl", "ru", "zh"}
        self.assertTrue(subsets.issubset(allowed), f"Unexpected STS22 subsets: {subsets - allowed}")

    def test_assin2_uses_mteb_assin2sts(self) -> None:
        sts_cfg = load_sts_datasets_config()
        assin = sts_cfg["ASSIN2"]["sources"][0]
        self.assertEqual(assin["hf_path"], "mteb/Assin2STS")

    def test_semrel24_uses_mteb_semrel24sts_codes(self) -> None:
        sts_cfg = load_sts_datasets_config()
        subsets = {source["hf_subset"] for source in sts_cfg["SemRel24"]["sources"]}
        self.assertEqual(subsets, {"arb", "hin", "ind"})

    def test_verify_check_surfaces_load_errors(self) -> None:
        check = Check(
            label="test",
            hf_path="missing/dataset",
            config=None,
            split="train",
        )
        with patch("datasets.load_dataset", side_effect=ValueError("Bad split: train")):
            error = verify_check(check)
        self.assertIsNotNone(error)
        self.assertIn("Bad split", error)


if __name__ == "__main__":
    unittest.main()
