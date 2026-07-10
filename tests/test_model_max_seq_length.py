from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.model import apply_max_seq_length, load_sentence_transformer


class MaxSeqLengthTests(unittest.TestCase):
    def test_apply_max_seq_length_sets_attribute(self) -> None:
        model = MagicMock()
        model.max_seq_length = 8192
        apply_max_seq_length(model, 512)
        self.assertEqual(model.max_seq_length, 512)

    def test_apply_max_seq_length_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            apply_max_seq_length(MagicMock(), 0)

    def test_load_sentence_transformer_sets_max_seq_length(self) -> None:
        fake_model = MagicMock()
        fake_model.max_seq_length = 8192
        fake_model.to.return_value = fake_model

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
            loaded = load_sentence_transformer(
                "/tmp/fake-model",
                device="cpu",
                max_seq_length=512,
            )

        self.assertIs(loaded, fake_model)
        self.assertEqual(fake_model.max_seq_length, 512)

    def test_load_without_max_seq_length_leaves_default(self) -> None:
        fake_model = MagicMock()
        fake_model.max_seq_length = 8192
        fake_model.to.return_value = fake_model

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
            load_sentence_transformer("/tmp/fake-model", device="cpu")

        self.assertEqual(fake_model.max_seq_length, 8192)


if __name__ == "__main__":
    unittest.main()
