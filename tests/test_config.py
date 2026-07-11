from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_language_codes,
    get_loss_weights,
    get_num_epochs,
    get_previous_lang,
    get_training_order,
    load_languages_config,
    parse_lang_list,
)


class ConfigTests(unittest.TestCase):
    def test_training_order_has_16_languages(self) -> None:
        order = get_training_order()
        self.assertEqual(len(order), 16)
        self.assertEqual(order[0], "en")
        self.assertEqual(order[-1], "pl")

    def test_get_previous_lang(self) -> None:
        self.assertIsNone(get_previous_lang("en"))
        self.assertEqual(get_previous_lang("ko"), "ja")
        self.assertEqual(get_previous_lang("pl"), "vi")

    def test_parse_lang_list(self) -> None:
        self.assertEqual(len(parse_lang_list("all")), 16)
        self.assertEqual(parse_lang_list("en,ko"), ["en", "ko"])

    def test_get_num_epochs_defaults(self) -> None:
        cfg = {
            "training": {"default_num_epochs": 2, "num_epochs_xx": 2},
            "phases": {"retrieval": {"default_num_epochs": 3, "num_epochs_retrieval_xx": 3}},
        }
        self.assertEqual(get_num_epochs(cfg, "en", "sts"), 1)
        self.assertEqual(get_num_epochs(cfg, "en", "retrieval"), 1)
        cfg_legacy = {
            "training": {"num_epochs_en": 2},
            "phases": {"retrieval": {"num_epochs_retrieval_en": 4}},
        }
        self.assertEqual(get_num_epochs(cfg_legacy, "en", "sts"), 2)
        self.assertEqual(get_num_epochs(cfg_legacy, "en", "retrieval"), 4)

    def test_get_loss_weights_retrieval_phase(self) -> None:
        cfg = {
            "training": {"losses": {"mse": 0.8, "cosine": 0.2, "pairwise_mse": 0.0, "score_kl": 0.0}},
            "phases": {
                "retrieval": {
                    "losses": {"mse": 0.2, "cosine": 0.4, "pairwise_mse": 0.0, "score_kl": 0.4},
                }
            },
        }
        weights = get_loss_weights(cfg, "retrieval")
        self.assertAlmostEqual(weights["mse"], 0.2)
        self.assertAlmostEqual(weights["cosine"], 0.4)
        self.assertAlmostEqual(weights["score_kl"], 0.4)
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
