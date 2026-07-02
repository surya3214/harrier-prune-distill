from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

from harrier_distill.data import ensure_dir, write_corpus_parquet
from harrier_distill.model import encode_with_prompt, load_sentence_transformer


@dataclass
class StsPairs:
    sentence1: list[str]
    sentence2: list[str]
    score: np.ndarray
    lang: str
    task: str
    split: str
    source_path: str


def download_sts_split(
    *,
    hf_path: str,
    split: str,
    lang: str,
    task: str,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(hf_path, split=split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(
            {
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": float(row["score"]),
                "lang": lang,
                "task": task,
                "split": split,
            }
        )
    return rows


def write_sts_parquet(rows: list[dict[str, Any]], output_path: Path) -> int:
    write_corpus_parquet(rows, output_path)
    return len(rows)


def load_sts_parquet(path: str | Path) -> StsPairs:
    table = pq.read_table(path, columns=["sentence1", "sentence2", "score", "lang", "task", "split"])
    data = table.to_pydict()
    return StsPairs(
        sentence1=data["sentence1"],
        sentence2=data["sentence2"],
        score=np.asarray(data["score"], dtype=np.float32),
        lang=data["lang"][0] if data["lang"] else "",
        task=data["task"][0] if data["task"] else "",
        split=data["split"][0] if data["split"] else "",
        source_path=str(path),
    )


def parquet_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sts_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": entries}, f, indent=2, ensure_ascii=False)


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


@torch.inference_mode()
def encode_pair_similarities(
    model: SentenceTransformer,
    sentence1: list[str],
    sentence2: list[str],
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
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
def compute_sts_spearman(
    model: SentenceTransformer,
    pairs: StsPairs,
    *,
    prompt_name: str = "sts_query",
    device: torch.device | None = None,
    max_length: int = 512,
    batch_size: int = 64,
) -> float:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sims = encode_pair_similarities(
        model,
        pairs.sentence1,
        pairs.sentence2,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    return float(spearmanr(sims, pairs.score).correlation)


def evaluate_sts_local(
    model_path: str | Path,
    *,
    task_paths: dict[str, Path],
    prompt_name: str = "sts_query",
    batch_size: int = 64,
    max_length: int = 512,
) -> dict[str, Any]:
    """Evaluate STS Spearman scores from local parquet files (offline)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sentence_transformer(model_path, device=device)

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "backend": "local",
        "tasks": {},
    }
    for task_name, parquet_path in task_paths.items():
        if not parquet_path.exists():
            raise FileNotFoundError(f"Local STS dataset not found for {task_name}: {parquet_path}")
        pairs = load_sts_parquet(parquet_path)
        score = compute_sts_spearman(
            model,
            pairs,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )
        summary["tasks"][task_name] = {
            "main_score": score,
            "scores": {
                "test": [
                    {
                        "main_score": score,
                        "pearson": None,
                        "spearman": score,
                    }
                ]
            },
            "source_path": str(parquet_path),
            "pair_count": len(pairs.score),
            "split": pairs.split,
        }
    return summary
