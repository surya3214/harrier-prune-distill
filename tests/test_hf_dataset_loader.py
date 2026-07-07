from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.data import load_hf_source_dataset


class HfDatasetLoaderTests(unittest.TestCase):
    def test_legacy_script_error_includes_jsick_hint(self) -> None:
        source_cfg = {"hf_path": "hpprc/jsick", "split": "train", "streaming": False}
        with patch("datasets.load_dataset", side_effect=ValueError("contains custom code")):
            with self.assertRaises(RuntimeError) as ctx:
                load_hf_source_dataset(source_cfg, split="train")
        message = str(ctx.exception)
        self.assertIn("hpprc/jsick", message)
        self.assertIn("mteb/JSICK", message)

    def test_bad_split_error_is_raised(self) -> None:
        source_cfg = {"hf_path": "example/dataset", "split": "train", "streaming": False}
        with patch("datasets.load_dataset", side_effect=ValueError("Bad split: train")):
            with self.assertRaises(ValueError) as ctx:
                load_hf_source_dataset(source_cfg, split="train")
        self.assertIn("does not have split", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
