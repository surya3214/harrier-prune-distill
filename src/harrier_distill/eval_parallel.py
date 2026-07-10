"""Multi-GPU parallel helpers for compare_sts / compare_retrieval."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable


def parse_gpu_ids(gpus: str | list[int] | None) -> list[int] | None:
    if gpus is None:
        return None
    if isinstance(gpus, list):
        return [int(g) for g in gpus]
    text = str(gpus).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_gpu_ids(
    *,
    parallel: bool,
    n_models: int,
    gpus: list[int] | None = None,
    available: int | None = None,
) -> list[int]:
    """Return GPU IDs to use for parallel model eval (may be empty → sequential)."""
    if not parallel or n_models < 2:
        return []

    if available is None:
        try:
            import torch

            available = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except Exception:
            available = 0

    if available < 1:
        return []

    if gpus:
        ids = [int(g) for g in gpus]
        for gpu_id in ids:
            if gpu_id < 0 or gpu_id >= available:
                raise ValueError(f"GPU id {gpu_id} out of range (available={available})")
        return ids

    return list(range(min(available, n_models)))


def assign_gpus_to_models(
    models: list[tuple[str, str]],
    gpu_ids: list[int],
) -> list[tuple[str, str, int]]:
    """Assign GPUs round-robin to models. Returns (label, path, gpu_id)."""
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty for assignment")
    assigned: list[tuple[str, str, int]] = []
    for idx, (label, path) in enumerate(models):
        assigned.append((label, path, gpu_ids[idx % len(gpu_ids)]))
    return assigned


def _prepare_worker_env(job: dict[str, Any]) -> None:
    """Configure CUDA remapping, quiet mode, and optional per-model log file."""
    gpu_id = int(job["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Parallel workers never share tqdm on stdout; stage lines still print.
    os.environ["EVAL_QUIET"] = "1"
    log_path = job.get("log_path")
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so each run starts a fresh per-model log.
        path.write_text("", encoding="utf-8")
        os.environ["EVAL_LOG_FILE"] = str(path)
    else:
        os.environ.pop("EVAL_LOG_FILE", None)


def _sts_eval_worker(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _prepare_worker_env(job)
    gpu_id = int(job["gpu_id"])

    from harrier_distill.eval import evaluate_sts
    from harrier_distill.eval_progress import log_eval

    label = str(job["label"])
    log_eval(f"Worker starting on physical gpu={gpu_id}", label=label, gpu=gpu_id)
    summary = evaluate_sts(
        job["model_path"],
        tasks=job["tasks"],
        prompt_name=job["prompt_name"],
        batch_size=job["batch_size"],
        output_dir=job.get("output_dir"),
        use_local_sts=job["use_local_sts"],
        local_task_paths={k: Path(v) for k, v in (job.get("local_task_paths") or {}).items()}
        if job.get("local_task_paths")
        else None,
        max_length=job["max_length"],
        device="cuda:0",
        label=label,
        gpu=gpu_id,
        quiet=True,
    )
    return label, summary


def _retrieval_eval_worker(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _prepare_worker_env(job)
    gpu_id = int(job["gpu_id"])

    from harrier_distill.eval import evaluate_retrieval
    from harrier_distill.eval_progress import log_eval

    label = str(job["label"])
    log_eval(f"Worker starting on physical gpu={gpu_id}", label=label, gpu=gpu_id)

    local_task_paths = job.get("local_task_paths")
    resolved_paths = None
    if local_task_paths is not None:
        resolved_paths = {}
        for task_name, value in local_task_paths.items():
            if isinstance(value, dict):
                resolved_paths[task_name] = {k: Path(v) for k, v in value.items()}
            else:
                resolved_paths[task_name] = Path(value)

    summary = evaluate_retrieval(
        job["model_path"],
        tasks=job["tasks"],
        query_prompt=job["query_prompt"],
        batch_size=job["batch_size"],
        output_dir=job.get("output_dir"),
        miracl_subsets=job.get("miracl_subsets"),
        use_local_retrieval=job["use_local_retrieval"],
        local_task_paths=resolved_paths,
        max_length=job["max_length"],
        device="cuda:0",
        label=label,
        gpu=gpu_id,
        quiet=True,
    )
    return label, summary


def run_parallel_jobs(
    jobs: list[dict[str, Any]],
    *,
    worker: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run eval jobs in spawn workers; fail-fast on first error."""
    if not jobs:
        return {}

    workers = max_workers or len(jobs)
    workers = max(1, min(workers, len(jobs)))
    ctx = get_context("spawn")
    summaries: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(worker, job): job["label"] for job in jobs}
        try:
            for future in as_completed(futures):
                label = futures[future]
                try:
                    result_label, summary = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"[eval][{label}] FAILED: {exc}") from exc
                summaries[result_label] = summary
        except Exception:
            for pending in futures:
                pending.cancel()
            raise

    return summaries


def serialize_sts_paths(paths: dict[str, Path] | None) -> dict[str, str] | None:
    if paths is None:
        return None
    return {name: str(path) for name, path in paths.items()}


def serialize_retrieval_paths(paths: dict[str, Any] | None) -> dict[str, Any] | None:
    if paths is None:
        return None
    out: dict[str, Any] = {}
    for name, value in paths.items():
        if isinstance(value, dict):
            out[name] = {k: str(v) for k, v in value.items()}
        else:
            out[name] = str(value)
    return out
