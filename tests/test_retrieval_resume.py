"""Tests for retrieval pipeline language-level resume helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    canonical_retrieval_checkpoint_dir,
    get_last_completed_retrieval_lang,
    get_next_incomplete_retrieval_lang,
    get_training_order,
    is_retrieval_embedding_complete,
    is_retrieval_lang_complete,
)


def _write_complete_checkpoint(cfg: dict, lang: str) -> Path:
    ckpt = canonical_retrieval_checkpoint_dir(cfg, lang)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "train_metrics.json").write_text(
        json.dumps({"lang": lang, "phase": "retrieval"}),
        encoding="utf-8",
    )
    return ckpt


def _write_embeddings(cfg: dict, lang: str) -> Path:
    from harrier_distill.config import resolve_embedding_path

    path = resolve_embedding_path(cfg, lang, phase="retrieval")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet-bytes")
    return path


class RetrievalResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name)
        self.cfg = {"paths": {"output_dir": str(self.output)}}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_incomplete_when_missing(self) -> None:
        self.assertFalse(is_retrieval_lang_complete(self.cfg, "en"))
        self.assertFalse(is_retrieval_embedding_complete(self.cfg, "en"))
        self.assertIsNone(get_last_completed_retrieval_lang(self.cfg))
        self.assertEqual(get_next_incomplete_retrieval_lang(self.cfg), "en")

    def test_complete_requires_metrics_and_config(self) -> None:
        ckpt = canonical_retrieval_checkpoint_dir(self.cfg, "en")
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "config.json").write_text("{}", encoding="utf-8")
        self.assertFalse(is_retrieval_lang_complete(self.cfg, "en"))
        (ckpt / "train_metrics.json").write_text("{}", encoding="utf-8")
        self.assertTrue(is_retrieval_lang_complete(self.cfg, "en"))

    def test_next_lang_after_prefix_complete(self) -> None:
        order = get_training_order()
        self.assertGreaterEqual(len(order), 3)
        _write_complete_checkpoint(self.cfg, order[0])
        _write_complete_checkpoint(self.cfg, order[1])
        self.assertEqual(get_last_completed_retrieval_lang(self.cfg), order[1])
        self.assertEqual(get_next_incomplete_retrieval_lang(self.cfg), order[2])

    def test_all_complete(self) -> None:
        for lang in get_training_order():
            _write_complete_checkpoint(self.cfg, lang)
        self.assertEqual(get_last_completed_retrieval_lang(self.cfg), get_training_order()[-1])
        self.assertIsNone(get_next_incomplete_retrieval_lang(self.cfg))

    def test_gap_stops_prefix(self) -> None:
        order = get_training_order()
        _write_complete_checkpoint(self.cfg, order[0])
        # leave order[1] incomplete but complete order[2]
        _write_complete_checkpoint(self.cfg, order[2])
        self.assertEqual(get_last_completed_retrieval_lang(self.cfg), order[0])
        self.assertEqual(get_next_incomplete_retrieval_lang(self.cfg), order[1])

    def test_embedding_complete(self) -> None:
        self.assertFalse(is_retrieval_embedding_complete(self.cfg, "ko"))
        _write_embeddings(self.cfg, "ko")
        self.assertTrue(is_retrieval_embedding_complete(self.cfg, "ko"))


if __name__ == "__main__":
    unittest.main()
