from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_loss_weights
from harrier_distill.losses import (
    combine_weighted_losses,
    cosine_embedding_loss,
    pairwise_mse_loss,
    pointwise_mse_loss,
    triplet_similarity_mse,
)


class LossTests(unittest.TestCase):
    def test_pointwise_mse_on_normalized_vectors_matches_cosine_distance(self) -> None:
        student = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float32)
        teacher = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float32)

        mse = pointwise_mse_loss(student, teacher)
        cosine = cosine_embedding_loss(student, teacher)

        expected_mse = torch.mean((student - teacher) ** 2)
        self.assertTrue(torch.allclose(mse, expected_mse))
        self.assertGreater(float(cosine.item()), 0.0)

    def test_normalized_mse_is_monotone_with_cosine(self) -> None:
        teacher = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        student_close = torch.tensor([[0.99, 0.1, 0.0]], dtype=torch.float32)
        student_close = student_close / student_close.norm(dim=-1, keepdim=True)
        student_far = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

        mse_close = float(pointwise_mse_loss(student_close, teacher).item())
        mse_far = float(pointwise_mse_loss(student_far, teacher).item())
        self.assertLess(mse_close, mse_far)

    def test_triplet_similarity_mse(self) -> None:
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        student = teacher.clone()
        loss = triplet_similarity_mse(student, teacher)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

        student = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        loss = triplet_similarity_mse(student, teacher)
        self.assertGreater(float(loss.item()), 0.0)

    def test_pairwise_mse_averages_triplets(self) -> None:
        teacher_triplets = [
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]
        student_triplets = [tensor.clone() for tensor in teacher_triplets]
        loss = pairwise_mse_loss(student_triplets, teacher_triplets)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_combine_weighted_losses(self) -> None:
        mse = torch.tensor(0.5)
        cosine = torch.tensor(0.25)
        components = combine_weighted_losses(
            weights={"mse": 1.0, "cosine": 0.5, "pairwise_mse": 0.0},
            mse=mse,
            cosine=cosine,
        )
        self.assertAlmostEqual(float(components.total.item()), 0.625, places=6)

    def test_get_loss_weights_defaults_to_mse(self) -> None:
        weights = get_loss_weights({"training": {}}, "sts")
        self.assertEqual(weights["mse"], 1.0)
        self.assertEqual(weights["cosine"], 0.0)
        self.assertEqual(weights["pairwise_mse"], 0.0)

    def test_get_loss_weights_phase_override(self) -> None:
        cfg = {
            "training": {"losses": {"mse": 1.0, "cosine": 0.0, "pairwise_mse": 0.0}},
            "phases": {
                "retrieval": {
                    "losses": {"mse": 0.5, "cosine": 0.1, "pairwise_mse": 0.4},
                }
            },
        }
        weights = get_loss_weights(cfg, "retrieval")
        self.assertEqual(weights["mse"], 0.5)
        self.assertAlmostEqual(weights["cosine"], 0.1)
        self.assertAlmostEqual(weights["pairwise_mse"], 0.4)
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
