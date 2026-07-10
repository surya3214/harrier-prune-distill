from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

from harrier_distill.data import ensure_dir, write_corpus_parquet
from harrier_distill.eval_progress import StageTimer, log_eval, task_progress
from harrier_distill.model import encode_texts, encode_with_prompt, load_sentence_transformer


@dataclass
class RetrievalEvalData:
    query_ids: list[str]
    query_texts: list[str]
    doc_ids: list[str]
    doc_texts: list[str]
    qrels: dict[str, dict[str, float]]
    task: str
    lang: str
    split: str
    source_dir: str


RetrievalTaskPaths = Path | dict[str, Path]


def _corpus_text_from_row(row: dict[str, Any], *, text_column: str, title_column: str | None) -> str:
    title = row.get(title_column) if title_column else None
    body = row.get(text_column) or ""
    if title:
        return f"{title}\n{body}".strip()
    return str(body)


def _load_hf_table(hf_path: str, config: str, split: str):
    from datasets import load_dataset

    return load_dataset(hf_path, config, split=split, streaming=False)


def download_retrieval_eval_task(
    *,
    task: str,
    lang: str,
    split: str,
    hf_path: str,
    queries_cfg: dict[str, Any],
    corpus_cfg: dict[str, Any],
    qrels_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Download one retrieval benchmark (queries, corpus, qrels) to parquet."""
    ensure_dir(output_dir)

    queries_table = _load_hf_table(hf_path, queries_cfg["config"], queries_cfg["split"])
    corpus_table = _load_hf_table(hf_path, corpus_cfg["config"], corpus_cfg["split"])
    qrels_table = _load_hf_table(hf_path, qrels_cfg["config"], qrels_cfg["split"])

    query_id_col = queries_cfg["id_column"]
    query_text_col = queries_cfg["text_column"]
    doc_id_col = corpus_cfg["id_column"]
    doc_text_col = corpus_cfg["text_column"]
    title_col = corpus_cfg.get("title_column")
    qrels_query_col = qrels_cfg["query_id_column"]
    qrels_doc_col = qrels_cfg["doc_id_column"]
    score_col = qrels_cfg["score_column"]

    query_rows: list[dict[str, Any]] = []
    for row in queries_table:
        query_rows.append(
            {
                "query_id": str(row[query_id_col]),
                "text": str(row[query_text_col]),
                "task": task,
                "lang": lang,
                "split": split,
            }
        )

    qrel_rows: list[dict[str, Any]] = []
    eval_query_ids: set[str] = set()
    for row in qrels_table:
        qid = str(row[qrels_query_col])
        eval_query_ids.add(qid)
        qrel_rows.append(
            {
                "query_id": qid,
                "doc_id": str(row[qrels_doc_col]),
                "score": float(row[score_col]),
                "task": task,
                "lang": lang,
                "split": split,
            }
        )

    query_rows = [row for row in query_rows if row["query_id"] in eval_query_ids]

    corpus_rows: list[dict[str, Any]] = []
    for row in corpus_table:
        text = _corpus_text_from_row(row, text_column=doc_text_col, title_column=title_col)
        corpus_row: dict[str, Any] = {
            "doc_id": str(row[doc_id_col]),
            "text": text,
            "task": task,
            "lang": lang,
            "split": split,
        }
        if title_col and row.get(title_col):
            corpus_row["title"] = str(row[title_col])
        corpus_rows.append(corpus_row)

    paths = {
        "queries": output_dir / "queries.parquet",
        "corpus": output_dir / "corpus.parquet",
        "qrels": output_dir / "qrels.parquet",
    }
    write_corpus_parquet(query_rows, paths["queries"])
    write_corpus_parquet(corpus_rows, paths["corpus"])
    write_corpus_parquet(qrel_rows, paths["qrels"])

    return {
        "task": task,
        "lang": lang,
        "split": split,
        "hf_path": hf_path,
        "output_dir": str(output_dir),
        "query_count": len(query_rows),
        "corpus_count": len(corpus_rows),
        "qrel_count": len(qrel_rows),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha1": {name: parquet_sha1(path) for name, path in paths.items()},
    }


def parquet_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_retrieval_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"datasets": entries}, f, indent=2, ensure_ascii=False)


def load_retrieval_eval_parquet(task_dir: Path) -> RetrievalEvalData:
    queries_path = task_dir / "queries.parquet"
    corpus_path = task_dir / "corpus.parquet"
    qrels_path = task_dir / "qrels.parquet"
    for path in (queries_path, corpus_path, qrels_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing retrieval eval parquet: {path}")

    queries = pq.read_table(queries_path, columns=["query_id", "text", "task", "lang", "split"]).to_pydict()
    corpus = pq.read_table(corpus_path, columns=["doc_id", "text", "task", "lang", "split"]).to_pydict()
    qrels_table = pq.read_table(qrels_path, columns=["query_id", "doc_id", "score"]).to_pydict()

    qrels: dict[str, dict[str, float]] = {}
    for qid, did, score in zip(qrels_table["query_id"], qrels_table["doc_id"], qrels_table["score"]):
        qrels.setdefault(str(qid), {})[str(did)] = float(score)

    return RetrievalEvalData(
        query_ids=[str(q) for q in queries["query_id"]],
        query_texts=[str(t) for t in queries["text"]],
        doc_ids=[str(d) for d in corpus["doc_id"]],
        doc_texts=[str(t) for t in corpus["text"]],
        qrels=qrels,
        task=str(queries["task"][0]) if queries["task"] else "",
        lang=str(queries["lang"][0]) if queries["lang"] else "",
        split=str(queries["split"][0]) if queries["split"] else "",
        source_dir=str(task_dir),
    )


@torch.inference_mode()
def encode_retrieval_queries(
    model: SentenceTransformer,
    texts: list[str],
    *,
    query_prompt: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
    show_progress: bool = False,
    quiet: bool | None = None,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    starts = range(0, len(texts), batch_size)
    n_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
    for start in task_progress(
        starts,
        desc="encode-queries",
        total=n_batches,
        quiet=quiet,
        disable=not show_progress,
    ):
        batch = texts[start : start + batch_size]
        emb = encode_with_prompt(
            model,
            batch,
            prompt_name=query_prompt,
            device=device,
            max_length=max_length,
        )
        outputs.append(emb.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.zeros((0, 0), dtype=np.float32)


@torch.inference_mode()
def encode_retrieval_corpus(
    model: SentenceTransformer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
    show_progress: bool = False,
    quiet: bool | None = None,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    starts = range(0, len(texts), batch_size)
    n_batches = (len(texts) + batch_size - 1) // batch_size if texts else 0
    for start in task_progress(
        starts,
        desc="encode-corpus",
        total=n_batches,
        quiet=quiet,
        disable=not show_progress,
    ):
        batch = texts[start : start + batch_size]
        emb = encode_texts(
            model,
            batch,
            device=device,
            max_length=max_length,
            prompt_name=None,
        )
        outputs.append(emb.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.zeros((0, 0), dtype=np.float32)


def _dcg_at_k(relevances: list[float], k: int) -> float:
    gains = relevances[:k]
    if not gains:
        return 0.0
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(np.asarray(gains, dtype=np.float64) / discounts))


def compute_ndcg_at_k(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, float],
    *,
    k: int = 10,
) -> float:
    relevances = [relevant_docs.get(doc_id, 0.0) for doc_id in ranked_doc_ids]
    ideal = sorted(relevant_docs.values(), reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg_at_k(relevances, k) / idcg


def score_retrieval_subset(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    *,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, float]],
    top_k: int = 10,
    corpus_chunk_size: int = 50_000,
    show_progress: bool = False,
    quiet: bool | None = None,
) -> float:
    """Compute mean nDCG@10 over queries using chunked corpus scoring."""
    if len(query_ids) == 0:
        return 0.0

    scores: list[float] = []
    query_iter = task_progress(
        list(enumerate(query_ids)),
        desc="ndcg@10",
        total=len(query_ids),
        quiet=quiet,
        disable=not show_progress,
    )

    for q_idx, query_id in query_iter:
        relevant = qrels.get(query_id, {})
        if not relevant:
            continue

        q_emb = query_embeddings[q_idx]
        merged: list[tuple[float, str]] = []

        for start in range(0, len(corpus_embeddings), corpus_chunk_size):
            chunk = corpus_embeddings[start : start + corpus_chunk_size]
            sims = chunk @ q_emb
            local_top = min(top_k, len(sims))
            if local_top == 0:
                continue
            top_local_idx = np.argpartition(-sims, local_top - 1)[:local_top]
            for local_idx in top_local_idx:
                global_idx = start + int(local_idx)
                merged.append((float(sims[local_idx]), doc_ids[global_idx]))

        merged.sort(key=lambda item: item[0], reverse=True)
        ranked_doc_ids = [doc_id for _, doc_id in merged[:top_k]]
        scores.append(compute_ndcg_at_k(ranked_doc_ids, relevant, k=top_k))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate_retrieval_task_local(
    model: SentenceTransformer,
    data: RetrievalEvalData,
    *,
    query_prompt: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
    corpus_chunk_size: int = 50_000,
    label: str | None = None,
    gpu: int | None = None,
    quiet: bool | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    show_bars = show_progress and not (quiet if quiet is not None else False)
    log_eval(
        f"Encoding queries ({len(data.query_texts):,})...",
        label=label,
        gpu=gpu,
    )
    query_emb = encode_retrieval_queries(
        model,
        data.query_texts,
        query_prompt=query_prompt,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        show_progress=show_bars,
        quiet=quiet,
    )
    log_eval(
        f"Encoding corpus ({len(data.doc_texts):,} docs)...",
        label=label,
        gpu=gpu,
    )
    corpus_emb = encode_retrieval_corpus(
        model,
        data.doc_texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        show_progress=show_bars,
        quiet=quiet,
    )
    log_eval("Computing nDCG@10...", label=label, gpu=gpu)
    score = score_retrieval_subset(
        query_emb,
        corpus_emb,
        query_ids=data.query_ids,
        doc_ids=data.doc_ids,
        qrels=data.qrels,
        corpus_chunk_size=corpus_chunk_size,
        show_progress=show_bars,
        quiet=quiet,
    )
    return {
        "main_score": score,
        "scores": {
            "dev" if data.split == "dev" else data.split: [
                {
                    "main_score": score,
                    "ndcg_at_10": score,
                }
            ]
        },
        "source_path": data.source_dir,
        "query_count": len(data.query_ids),
        "corpus_count": len(data.doc_ids),
        "qrel_count": sum(len(docs) for docs in data.qrels.values()),
        "split": data.split,
        "lang": data.lang,
        "hf_subset": data.lang,
    }


def _iter_task_dirs(task_name: str, task_paths: RetrievalTaskPaths) -> Iterator[tuple[str | None, Path]]:
    if isinstance(task_paths, dict):
        for subset, path in task_paths.items():
            yield subset, path
    else:
        yield None, task_paths


def evaluate_retrieval_local(
    model_path: str | Path,
    *,
    task_names: list[str],
    local_task_paths: dict[str, RetrievalTaskPaths],
    query_prompt: str = "web_search_query",
    batch_size: int = 64,
    max_length: int = 512,
    corpus_chunk_size: int = 50_000,
    device: torch.device | str | None = None,
    label: str | None = None,
    gpu: int | None = None,
    quiet: bool | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval nDCG@10 from local parquet directories (offline)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    log_eval(f"Loading model: {model_path}", label=label, gpu=gpu)
    model = load_sentence_transformer(model_path, device=device, max_seq_length=max_length)
    log_eval(f"Model loaded on {device} (max_seq_length={max_length})", label=label, gpu=gpu)

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "backend": "local",
        "query_prompt": query_prompt,
        "tasks": {},
    }

    show_progress = not (quiet if quiet is not None else False)
    from harrier_distill.eval_parallel import release_cuda_memory

    try:
        for task_name in task_names:
            if task_name not in local_task_paths:
                raise ValueError(f"No local retrieval data configured for task: {task_name}")

            subset_dirs = list(_iter_task_dirs(task_name, local_task_paths[task_name]))
            subset_results: list[dict[str, Any]] = []
            for subset_idx, (subset, task_dir) in enumerate(subset_dirs, start=1):
                if not task_dir.exists():
                    raise FileNotFoundError(f"Local retrieval dataset not found for {task_name}: {task_dir}")
                data = load_retrieval_eval_parquet(task_dir)
                subset_label = subset or data.lang or "default"
                log_eval(
                    f"Task {task_name} [{subset_label}] ({subset_idx}/{len(subset_dirs)}): "
                    f"{len(data.query_ids):,} queries, {len(data.doc_ids):,} docs",
                    label=label,
                    gpu=gpu,
                )
                timer = StageTimer()
                result = evaluate_retrieval_task_local(
                    model,
                    data,
                    query_prompt=query_prompt,
                    device=device,
                    max_length=max_length,
                    batch_size=batch_size,
                    corpus_chunk_size=corpus_chunk_size,
                    label=label,
                    gpu=gpu,
                    quiet=quiet,
                    show_progress=show_progress,
                )
                log_eval(
                    f"{task_name} [{subset_label}]: nDCG@10={result['main_score']:.4f} ({timer.elapsed_str()})",
                    label=label,
                    gpu=gpu,
                )
                subset_results.append(result)

            if task_name == "MIRACLRetrieval" and len(subset_results) > 1:
                main_score = float(np.mean([r["main_score"] for r in subset_results]))
                summary["tasks"][task_name] = {
                    "main_score": main_score,
                    "scores": subset_results[0]["scores"],
                    "source_path": ", ".join(r["source_path"] for r in subset_results),
                    "query_count": sum(r["query_count"] for r in subset_results),
                    "corpus_count": sum(r["corpus_count"] for r in subset_results),
                    "qrel_count": sum(r["qrel_count"] for r in subset_results),
                    "split": subset_results[0]["split"],
                    "subsets": {r["lang"]: r for r in subset_results},
                }
            else:
                summary["tasks"][task_name] = subset_results[0]

        return summary
    finally:
        release_cuda_memory(model)
        model = None
        log_eval("Released model GPU memory", label=label, gpu=gpu)


def get_local_retrieval_task_paths(
    task_names: list[str],
    local_task_paths: dict[str, RetrievalTaskPaths],
) -> dict[str, RetrievalTaskPaths]:
    missing = [name for name in task_names if name not in local_task_paths]
    if missing:
        raise ValueError(
            f"No local retrieval parquet configured for tasks: {', '.join(missing)}. "
            "Run scripts/01_download_retrieval_eval_local.py and set paths.retrieval_eval_data_root."
        )

    selected = {name: local_task_paths[name] for name in task_names}
    for task_name, paths in selected.items():
        if isinstance(paths, dict):
            missing_files = [f"{task_name}/{subset}" for subset, path in paths.items() if not path.exists()]
        else:
            missing_files = [task_name] if not paths.exists() else []
        if missing_files:
            raise FileNotFoundError(
                f"Local retrieval parquet missing for {', '.join(missing_files)}"
            )
    return selected
