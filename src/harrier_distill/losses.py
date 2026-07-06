from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LossComponents:
    total: torch.Tensor
    mse: torch.Tensor | None = None
    cosine: torch.Tensor | None = None
    pairwise_mse: torch.Tensor | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "total": float(self.total.item()),
            "mse": float(self.mse.item()) if self.mse is not None else None,
            "cosine": float(self.cosine.item()) if self.cosine is not None else None,
            "pairwise_mse": float(self.pairwise_mse.item()) if self.pairwise_mse is not None else None,
        }


def pointwise_mse_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student.float(), teacher.float())


def cosine_embedding_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()


def triplet_similarity_mse(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE between student and teacher query-to-doc dot products for one triplet.

    Args:
        student: (1 + n_docs, dim) L2-normalized embeddings; row 0 is query.
        teacher: same shape as student.
    """
    if student.shape[0] < 2:
        raise ValueError("triplet must contain one query and at least one document")

    student_sims = student[0:1] @ student[1:].T
    teacher_sims = teacher[0:1] @ teacher[1:].T
    return F.mse_loss(student_sims.squeeze(0), teacher_sims.squeeze(0))


def pairwise_mse_loss(student_triplets: list[torch.Tensor], teacher_triplets: list[torch.Tensor]) -> torch.Tensor:
    if not student_triplets:
        raise ValueError("pairwise_mse_loss requires at least one triplet")

    losses = [
        triplet_similarity_mse(student, teacher)
        for student, teacher in zip(student_triplets, teacher_triplets)
    ]
    return torch.stack(losses).mean()


def combine_weighted_losses(
    *,
    weights: dict[str, float],
    mse: torch.Tensor | None = None,
    cosine: torch.Tensor | None = None,
    pairwise_mse: torch.Tensor | None = None,
) -> LossComponents:
    device = _device_from_terms(mse, cosine, pairwise_mse)
    total = torch.zeros((), device=device)

    if weights.get("mse", 0.0) > 0:
        if mse is None:
            raise ValueError("mse weight > 0 but mse loss was not computed")
        total = total + weights["mse"] * mse
    if weights.get("cosine", 0.0) > 0:
        if cosine is None:
            raise ValueError("cosine weight > 0 but cosine loss was not computed")
        total = total + weights["cosine"] * cosine
    if weights.get("pairwise_mse", 0.0) > 0:
        if pairwise_mse is None:
            raise ValueError("pairwise_mse weight > 0 but pairwise_mse loss was not computed")
        total = total + weights["pairwise_mse"] * pairwise_mse

    return LossComponents(total=total, mse=mse, cosine=cosine, pairwise_mse=pairwise_mse)


def _device_from_terms(*tensors: torch.Tensor | None) -> torch.device:
    for tensor in tensors:
        if tensor is not None:
            return tensor.device
    return torch.device("cpu")


def format_loss_postfix(components: LossComponents, weights: dict[str, float]) -> dict[str, Any]:
    postfix: dict[str, Any] = {"loss": f"{components.total.item():.6f}"}
    if components.mse is not None and weights.get("mse", 0.0) > 0:
        postfix["mse"] = f"{components.mse.item():.6f}"
    if components.cosine is not None and weights.get("cosine", 0.0) > 0:
        postfix["cos"] = f"{components.cosine.item():.6f}"
    if components.pairwise_mse is not None and weights.get("pairwise_mse", 0.0) > 0:
        postfix["pw_mse"] = f"{components.pairwise_mse.item():.6f}"
    return postfix
