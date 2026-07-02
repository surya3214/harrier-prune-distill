from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from harrier_distill.model import encode_with_prompt, load_sentence_transformer

CACHE_MSE_THRESHOLD = 1e-4
STUDENT_MSE_THRESHOLD = 1e-3
PAIRWISE_STS_RATIO_THRESHOLD = 0.95
MIN_DIM_CORR_THRESHOLD = 0.5


def sample_embedding_rows(
    parquet_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """Sample texts and cached teacher embeddings from a parquet file."""
    from harrier_distill.data import CachedEmbeddingDataset

    dataset = CachedEmbeddingDataset(parquet_path)
    total = len(dataset)
    if total == 0:
        raise ValueError(f"No rows found in {parquet_path}")

    count = total if sample_size <= 0 else min(sample_size, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False)

    texts: list[str] = []
    embeddings: list[np.ndarray] = []
    for idx in indices:
        item = dataset[int(idx)]
        texts.append(item["text"])
        embeddings.append(item["teacher_embedding"].numpy())

    return texts, np.stack(embeddings, axis=0).astype(np.float32)


def _length_bucket(char_len: int) -> str:
    if char_len < 100:
        return "short"
    if char_len < 300:
        return "medium"
    return "long"


@torch.inference_mode()
def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = encode_with_prompt(
            model,
            batch,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
        )
        outputs.append(emb.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def pointwise_alignment_metrics(
    teacher_emb: np.ndarray,
    student_emb: np.ndarray,
    *,
    texts: list[str] | None = None,
) -> dict[str, Any]:
    """Compute pointwise alignment metrics between teacher and student embeddings."""
    teacher = teacher_emb.astype(np.float32)
    student = student_emb.astype(np.float32)

    mse = float(np.mean((teacher - student) ** 2))
    cosine = np.sum(teacher * student, axis=1)
    cosine = np.clip(cosine, -1.0, 1.0)
    angular_deg = np.degrees(np.arccos(cosine))
    l2 = np.linalg.norm(teacher - student, axis=1)

    dim_corrs: list[float] = []
    for dim in range(teacher.shape[1]):
        corr = np.corrcoef(teacher[:, dim], student[:, dim])[0, 1]
        if np.isfinite(corr):
            dim_corrs.append(float(corr))

    metrics: dict[str, Any] = {
        "count": int(teacher.shape[0]),
        "mse": mse,
        "cosine_mean": float(np.mean(cosine)),
        "cosine_p50": float(np.percentile(cosine, 50)),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "angular_error_deg_mean": float(np.mean(angular_deg)),
        "l2_distance_mean": float(np.mean(l2)),
        "dim_corr_mean": float(np.mean(dim_corrs)) if dim_corrs else None,
        "dim_corr_min": float(np.min(dim_corrs)) if dim_corrs else None,
    }

    if texts:
        buckets: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
        for text, cos_val in zip(texts, cosine):
            buckets[_length_bucket(len(text))].append(float(cos_val))
        metrics["cosine_by_length"] = {
            bucket: {
                "count": len(values),
                "cosine_mean": float(np.mean(values)) if values else None,
            }
            for bucket, values in buckets.items()
        }

    return metrics


def validate_cache_alignment(
    teacher_model: SentenceTransformer,
    texts: list[str],
    cached_emb: np.ndarray,
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Re-encode teacher and compare against cached parquet embeddings."""
    fresh_emb = encode_texts(
        teacher_model,
        texts,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = pointwise_alignment_metrics(cached_emb, fresh_emb)
    return {
        "cache_mse": metrics["mse"],
        "cache_cosine_mean": metrics["cosine_mean"],
        "cache_max_abs_diff": float(np.max(np.abs(cached_emb - fresh_emb))),
        "cache_ok": metrics["mse"] < CACHE_MSE_THRESHOLD,
    }


def _encode_pair_similarities(
    model: SentenceTransformer,
    sentence1: list[str],
    sentence2: list[str],
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    emb1 = encode_texts(
        model,
        sentence1,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    emb2 = encode_texts(
        model,
        sentence2,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    return np.sum(emb1 * emb2, axis=1)


@torch.inference_mode()
def pairwise_sts_proxy(
    teacher_model: SentenceTransformer,
    student_model: SentenceTransformer,
    *,
    dataset_name: str = "mteb/stsbenchmark-sts",
    split: str = "validation",
    prompt_name: str = "sts_query",
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Lightweight pairwise STS proxy without a full MTEB run."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    sentence1 = [row["sentence1"] for row in dataset]
    sentence2 = [row["sentence2"] for row in dataset]
    labels = np.asarray([float(row["score"]) for row in dataset], dtype=np.float32)

    teacher_sims = _encode_pair_similarities(
        teacher_model,
        sentence1,
        sentence2,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    student_sims = _encode_pair_similarities(
        student_model,
        sentence1,
        sentence2,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    teacher_spearman = float(spearmanr(teacher_sims, labels).correlation)
    student_spearman = float(spearmanr(student_sims, labels).correlation)
    rank_corr = float(spearmanr(teacher_sims, student_sims).correlation)

    ratio = student_spearman / teacher_spearman if teacher_spearman != 0 else None
    return {
        "dataset": dataset_name,
        "split": split,
        "pair_count": len(labels),
        "sts_spearman_teacher": teacher_spearman,
        "sts_spearman_student": student_spearman,
        "pairwise_rank_correlation": rank_corr,
        "student_teacher_spearman_ratio": ratio,
        "pairwise_ok": ratio is not None and ratio >= PAIRWISE_STS_RATIO_THRESHOLD,
    }


@torch.inference_mode()
def nli_pair_probe(
    teacher_model: SentenceTransformer,
    student_model: SentenceTransformer,
    *,
    lang: str,
    sample_size: int,
    seed: int,
    prompt_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Probe whether student preserves teacher similarity ordering on NLI pairs."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    if lang == "en":
        ds = load_dataset("sentence-transformers/all-nli", "pair", split="train", streaming=True)
        premise_key, hypothesis_key = "premise", "hypothesis"
    else:
        ds = load_dataset("klue", "nli", split="train", streaming=True)
        premise_key, hypothesis_key = "premise", "hypothesis"

    premises: list[str] = []
    hypotheses: list[str] = []
    for row in ds:
        premises.append(row[premise_key])
        hypotheses.append(row[hypothesis_key])
        if len(premises) >= sample_size:
            break

    if not premises:
        return {"enabled": False, "reason": "no NLI rows loaded"}

    indices = rng.choice(len(premises), size=min(sample_size, len(premises)), replace=False)
    premises = [premises[i] for i in indices]
    hypotheses = [hypotheses[i] for i in indices]

    teacher_sims = _encode_pair_similarities(
        teacher_model,
        premises,
        hypotheses,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    student_sims = _encode_pair_similarities(
        student_model,
        premises,
        hypotheses,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    rank_corr = float(spearmanr(teacher_sims, student_sims).correlation)
    return {
        "enabled": True,
        "lang": lang,
        "pair_count": len(premises),
        "pairwise_rank_correlation": rank_corr,
        "teacher_sim_mean": float(np.mean(teacher_sims)),
        "student_sim_mean": float(np.mean(student_sims)),
    }


def _build_checklist(report: dict[str, Any]) -> list[dict[str, Any]]:
    cache = report.get("cache_alignment", {})
    pointwise = report.get("pointwise_alignment", {})
    pairwise = report.get("pairwise_sts_proxy", {})
    baseline = report.get("pointwise_baseline", {})

    checks: list[dict[str, Any]] = [
        {
            "name": "cache_alignment_ok",
            "passed": bool(cache.get("cache_ok")),
            "detail": f"cache_mse={cache.get('cache_mse')}",
        },
        {
            "name": "student_mse_low",
            "passed": pointwise.get("mse") is not None and pointwise["mse"] < STUDENT_MSE_THRESHOLD,
            "detail": f"student_mse={pointwise.get('mse')}",
        },
        {
            "name": "pairwise_sts_ok",
            "passed": bool(pairwise.get("pairwise_ok")),
            "detail": (
                f"student_spearman={pairwise.get('sts_spearman_student')}, "
                f"teacher_spearman={pairwise.get('sts_spearman_teacher')}"
            ),
        },
        {
            "name": "no_dimension_collapse",
            "passed": (
                pointwise.get("dim_corr_min") is not None
                and pointwise["dim_corr_min"] > MIN_DIM_CORR_THRESHOLD
            ),
            "detail": f"dim_corr_min={pointwise.get('dim_corr_min')}",
        },
    ]

    if baseline:
        improved = (
            pointwise.get("mse") is not None
            and baseline.get("mse") is not None
            and pointwise["mse"] < baseline["mse"]
        )
        checks.append(
            {
                "name": "student_better_than_pruned_baseline",
                "passed": improved,
                "detail": (
                    f"student_mse={pointwise.get('mse')}, "
                    f"baseline_mse={baseline.get('mse')}"
                ),
            }
        )

    return checks


def run_alignment_report(
    *,
    teacher_path: str | Path,
    student_path: str | Path,
    embeddings_path: Path,
    lang: str,
    sample_size: int = 5000,
    seed: int = 42,
    prompt_name: str = "sts_query",
    max_length: int = 512,
    batch_size: int = 64,
    run_sts_proxy: bool = True,
    run_nli_probe: bool = False,
    pruned_baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run cache, pointwise, and pairwise diagnostics for teacher vs student."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, cached_emb = sample_embedding_rows(embeddings_path, sample_size, seed)
    teacher_model = load_sentence_transformer(teacher_path, device=device)
    student_model = load_sentence_transformer(student_path, device=device)

    cache_alignment = validate_cache_alignment(
        teacher_model,
        texts,
        cached_emb,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    student_emb = encode_texts(
        student_model,
        texts,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    pointwise_alignment = pointwise_alignment_metrics(
        cached_emb,
        student_emb,
        texts=texts,
    )

    report: dict[str, Any] = {
        "lang": lang,
        "sample_size": len(texts),
        "seed": seed,
        "teacher_path": str(teacher_path),
        "student_path": str(student_path),
        "embeddings_path": str(embeddings_path),
        "cache_alignment": cache_alignment,
        "pointwise_alignment": pointwise_alignment,
    }

    if pruned_baseline_path is not None:
        baseline_model = load_sentence_transformer(pruned_baseline_path, device=device)
        baseline_emb = encode_texts(
            baseline_model,
            texts,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        report["pointwise_baseline"] = pointwise_alignment_metrics(cached_emb, baseline_emb, texts=texts)
        report["pruned_baseline_path"] = str(pruned_baseline_path)

    if run_sts_proxy:
        report["pairwise_sts_proxy"] = pairwise_sts_proxy(
            teacher_model,
            student_model,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

    if run_nli_probe:
        report["nli_pair_probe"] = nli_pair_probe(
            teacher_model,
            student_model,
            lang=lang,
            sample_size=min(sample_size, 2000),
            seed=seed,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

    report["checklist"] = _build_checklist(report)
    return report


def print_alignment_summary(report: dict[str, Any]) -> None:
    print(f"\nMSE alignment debug (lang={report['lang']}, n={report['sample_size']})")
    print(f"  Teacher: {report['teacher_path']}")
    print(f"  Student: {report['student_path']}")
    print(f"  Embeddings: {report['embeddings_path']}")

    cache = report["cache_alignment"]
    print("\nCache alignment (teacher re-encode vs parquet):")
    print(f"  cache_mse: {cache['cache_mse']:.8f}")
    print(f"  cache_cosine_mean: {cache['cache_cosine_mean']:.6f}")
    print(f"  cache_ok: {cache['cache_ok']}")

    pointwise = report["pointwise_alignment"]
    print("\nPointwise teacher-student alignment:")
    print(f"  mse: {pointwise['mse']:.8f}")
    print(f"  cosine_mean: {pointwise['cosine_mean']:.6f}")
    print(f"  angular_error_deg_mean: {pointwise['angular_error_deg_mean']:.4f}")
    print(f"  dim_corr_min: {pointwise['dim_corr_min']:.4f}")

    if "pairwise_sts_proxy" in report:
        pairwise = report["pairwise_sts_proxy"]
        print("\nPairwise STS proxy (STSBenchmark validation):")
        print(f"  teacher Spearman: {pairwise['sts_spearman_teacher']:.4f}")
        print(f"  student Spearman: {pairwise['sts_spearman_student']:.4f}")
        print(f"  teacher-student rank corr: {pairwise['pairwise_rank_correlation']:.4f}")

    print("\nChecklist:")
    for item in report["checklist"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  [{status}] {item['name']}: {item['detail']}")


def save_alignment_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
