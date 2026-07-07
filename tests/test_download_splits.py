from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.data import MULTILINGUAL_NLI_HF_PATH, resolve_hf_source_splits


class DownloadSplitTests(unittest.TestCase):
    def test_multilingual_nli_resolves_lang_subsplits(self) -> None:
        source_cfg = {
            "hf_path": MULTILINGUAL_NLI_HF_PATH,
            "filter_lang": "de",
            "split": "train",
        }
        splits = resolve_hf_source_splits(source_cfg, lang="de")
        self.assertEqual(
            splits,
            ["de_mnli", "de_fever", "de_anli", "de_wanli", "de_ling"],
        )

    def test_regular_dataset_keeps_train_split(self) -> None:
        source_cfg = {
            "hf_path": "allenai/c4",
            "config": "en",
            "split": "train",
        }
        self.assertEqual(resolve_hf_source_splits(source_cfg, lang="en"), ["train"])

    def test_explicit_splits_override(self) -> None:
        source_cfg = {"splits": ["de_mnli", "de_fever"]}
        self.assertEqual(resolve_hf_source_splits(source_cfg, lang="de"), ["de_mnli", "de_fever"])


if __name__ == "__main__":
    unittest.main()
