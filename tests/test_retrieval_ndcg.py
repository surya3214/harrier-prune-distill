"""Lightweight numpy vs torch nDCG parity checks for local retrieval scoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harrier_distill.retrieval_eval import (
    _score_retrieval_subset_numpy,
    _score_retrieval_subset_torch,
    score_retrieval_subset,
)


def _synthetic_retrieval(
    *,
    n_docs: int = 20,
    n_queries: int = 5,
    dim: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n_docs, dim)).astype(np.float32)
    # Normalize so similarities are stable
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True).clip(min=1e-8)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-8)

    # Slightly perturb so ties are unlikely
    queries = queries + rng.normal(0.0, 1e-4, size=queries.shape).astype(np.float32)

    doc_ids = [f"d{i}" for i in range(n_docs)]
    query_ids = [f"q{i}" for i in range(n_queries)]
    qrels: dict[str, dict[str, float]] = {
        "q0": {"d0": 1.0, "d3": 1.0},
        "q1": {"d1": 2.0, "d5": 1.0, "d7": 1.0},
        "q2": {"d2": 1.0},
        "q3": {"d4": 1.0, "d8": 1.0},
        "q4": {"d6": 1.0, "d9": 1.0, "d10": 1.0},
    }
    return queries, corpus, query_ids, doc_ids, qrels


def test_numpy_vs_torch_cpu_parity() -> None:
    queries, corpus, query_ids, doc_ids, qrels = _synthetic_retrieval()
    numpy_score = _score_retrieval_subset_numpy(
        queries,
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=10,
        corpus_chunk_size=7,
    )
    torch_score = _score_retrieval_subset_torch(
        queries,
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=10,
        corpus_chunk_size=7,
        query_chunk_size=2,
        device="cpu",
    )
    assert abs(numpy_score - torch_score) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_numpy_vs_torch_cuda_parity() -> None:
    queries, corpus, query_ids, doc_ids, qrels = _synthetic_retrieval()
    numpy_score = _score_retrieval_subset_numpy(
        queries,
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=10,
        corpus_chunk_size=5,
    )
    torch_score = _score_retrieval_subset_torch(
        torch.as_tensor(queries, dtype=torch.float32, device="cuda"),
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=10,
        corpus_chunk_size=5,
        query_chunk_size=3,
        device="cuda",
    )
    assert abs(numpy_score - torch_score) < 1e-5


def test_empty_qrels_and_topk_larger_than_corpus() -> None:
    queries, corpus, query_ids, doc_ids, _ = _synthetic_retrieval(n_docs=8, n_queries=3)
    empty_qrels: dict[str, dict[str, float]] = {qid: {} for qid in query_ids}
    assert (
        score_retrieval_subset(
            queries,
            corpus,
            query_ids=query_ids,
            doc_ids=doc_ids,
            qrels=empty_qrels,
            top_k=50,
            device="numpy",
        )
        == 0.0
    )

    qrels = {"q0": {"d1": 1.0}, "q1": {}, "q2": {"d0": 1.0, "d3": 1.0}}
    numpy_score = _score_retrieval_subset_numpy(
        queries,
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=50,
        corpus_chunk_size=3,
    )
    torch_score = _score_retrieval_subset_torch(
        queries,
        corpus,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_k=50,
        corpus_chunk_size=3,
        query_chunk_size=2,
        device="cpu",
    )
    assert abs(numpy_score - torch_score) < 1e-5
    assert 0.0 <= numpy_score <= 1.0
