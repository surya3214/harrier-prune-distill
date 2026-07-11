from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_loss_weights, get_score_kl_temperature, should_skip_download, write_download_manifest
from harrier_distill.losses import (
    combine_weighted_losses,
    cosine_embedding_loss,
    pairwise_mse_loss,
    pairwise_score_kl_loss,
    pointwise_mse_loss,
    triplet_score_kl,
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

    def test_triplet_score_kl_zero_when_matched(self) -> None:
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        loss = triplet_score_kl(teacher.clone(), teacher, temperature=0.05)
        self.assertLess(float(loss.item()), 1e-5)

    def test_triplet_score_kl_positive_when_scores_disagree(self) -> None:
        teacher = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        student = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        loss = triplet_score_kl(student, teacher, temperature=0.05)
        self.assertGreater(float(loss.item()), 0.0)

    def test_pairwise_mse_averages_triplets(self) -> None:
        teacher_triplets = [
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]
        student_triplets = [tensor.clone() for tensor in teacher_triplets]
        loss = pairwise_mse_loss(student_triplets, teacher_triplets)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_pairwise_score_kl_averages_triplets(self) -> None:
        teacher_triplets = [
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        ]
        student_triplets = [tensor.clone() for tensor in teacher_triplets]
        loss = pairwise_score_kl_loss(student_triplets, teacher_triplets, temperature=0.05)
        self.assertLess(float(loss.item()), 1e-5)

    def test_combine_weighted_losses(self) -> None:
        mse = torch.tensor(0.5)
        cosine = torch.tensor(0.25)
        score_kl = torch.tensor(0.1)
        components = combine_weighted_losses(
            weights={"mse": 1.0, "cosine": 0.5, "pairwise_mse": 0.0, "score_kl": 0.2},
            mse=mse,
            cosine=cosine,
            score_kl=score_kl,
        )
        self.assertAlmostEqual(float(components.total.item()), 0.645, places=6)

    def test_get_loss_weights_defaults_to_mse(self) -> None:
        weights = get_loss_weights({"training": {}}, "sts")
        self.assertEqual(weights["mse"], 1.0)
        self.assertEqual(weights["cosine"], 0.0)
        self.assertEqual(weights["pairwise_mse"], 0.0)
        self.assertEqual(weights["score_kl"], 0.0)

    def test_get_loss_weights_phase_override(self) -> None:
        cfg = {
            "training": {"losses": {"mse": 1.0, "cosine": 0.0, "pairwise_mse": 0.0, "score_kl": 0.0}},
            "phases": {
                "retrieval": {
                    "losses": {"mse": 0.2, "cosine": 0.4, "pairwise_mse": 0.0, "score_kl": 0.4},
                }
            },
        }
        weights = get_loss_weights(cfg, "retrieval")
        self.assertEqual(weights["mse"], 0.2)
        self.assertAlmostEqual(weights["cosine"], 0.4)
        self.assertAlmostEqual(weights["pairwise_mse"], 0.0)
        self.assertAlmostEqual(weights["score_kl"], 0.4)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_get_score_kl_temperature(self) -> None:
        cfg = {
            "training": {"score_kl_temperature": 0.1},
            "phases": {"retrieval": {"score_kl_temperature": 0.05}},
        }
        self.assertAlmostEqual(get_score_kl_temperature(cfg, "sts"), 0.1)
        self.assertAlmostEqual(get_score_kl_temperature(cfg, "retrieval"), 0.05)


class SkipDownloadTests(unittest.TestCase):
    def test_retrieval_skip_uses_actual_triplet_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.parquet"
            manifest = root / "manifest.json"
            corpus.write_text("placeholder", encoding="utf-8")
            write_download_manifest(
                manifest,
                {
                    "rows": 150_000,  # would false-skip under old rows>=target logic
                    "target_triplets": 150_000,
                    "triplet_count": 30_000,
                    "negatives_per_query": 8,
                },
            )
            self.assertFalse(
                should_skip_download(
                    output_path=corpus,
                    manifest_path=manifest,
                    target_rows=150_000,
                    force=False,
                    skip_existing=True,
                    expected_negatives_per_query=8,
                )
            )

    def test_retrieval_skip_legacy_manifest_without_triplet_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.parquet"
            manifest = root / "manifest.json"
            corpus.write_text("placeholder", encoding="utf-8")
            write_download_manifest(
                manifest,
                {
                    "rows": 1_500_000,
                    "target_triplets": 150_000,
                    "negatives_per_query": 8,
                },
            )
            self.assertFalse(
                should_skip_download(
                    output_path=corpus,
                    manifest_path=manifest,
                    target_rows=150_000,
                    force=False,
                    skip_existing=True,
                    expected_negatives_per_query=8,
                )
            )

    def test_retrieval_skip_invalidates_on_negatives_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.parquet"
            manifest = root / "manifest.json"
            corpus.write_text("placeholder", encoding="utf-8")
            write_download_manifest(
                manifest,
                {
                    "rows": 1_500_000,
                    "target_triplets": 150_000,
                    "triplet_count": 150_000,
                    "negatives_per_query": 3,
                },
            )
            self.assertFalse(
                should_skip_download(
                    output_path=corpus,
                    manifest_path=manifest,
                    target_rows=150_000,
                    force=False,
                    skip_existing=True,
                    expected_negatives_per_query=8,
                )
            )
            self.assertTrue(
                should_skip_download(
                    output_path=corpus,
                    manifest_path=manifest,
                    target_rows=150_000,
                    force=False,
                    skip_existing=True,
                    expected_negatives_per_query=3,
                )
            )


if __name__ == "__main__":
    unittest.main()
