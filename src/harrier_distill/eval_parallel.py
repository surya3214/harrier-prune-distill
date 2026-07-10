"""Multi-GPU parallel helpers for compare_sts / compare_retrieval."""

from __future__ import annotations

import json
import os
import subprocess
import time
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


def probe_visible_cuda_device_count() -> int:
    """Count visible GPUs without initializing a CUDA context in this process.

    Calling ``torch.cuda.device_count()`` in the parent before spawn can leave
    contexts on every device and trigger
    ``CUDA error: CUDA-capable devices are busy`` in workers (especially under
    exclusive-process compute mode).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        parts = [part.strip() for part in cvd.split(",") if part.strip() != ""]
        # Explicit empty CVD means "no GPUs".
        return len(parts)

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return 0

    return sum(1 for line in output.splitlines() if line.strip())


def resolve_physical_cuda_id(logical_id: int, *, visible: str | None = None) -> int:
    """Map a logical GPU index to a physical id under ``CUDA_VISIBLE_DEVICES``."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    if cvd is None or str(cvd).strip() == "":
        return int(logical_id)
    parts = [part.strip() for part in str(cvd).split(",") if part.strip() != ""]
    idx = int(logical_id)
    if idx < 0 or idx >= len(parts):
        raise ValueError(
            f"Logical GPU id {idx} out of range for CUDA_VISIBLE_DEVICES={cvd!r}"
        )
    return int(parts[idx])


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
        available = probe_visible_cuda_device_count()

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


def make_jsonable(obj: Any) -> Any:
    """Convert numpy/path values into plain JSON-safe Python objects for IPC."""

    def _default(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        return str(value)

    return json.loads(json.dumps(obj, default=_default))


def release_cuda_memory(model: Any | None = None) -> None:
    """Move ``model`` to CPU (if given), drop the local ref, and free CUDA cache.

    Callers should also clear their own reference (``model = None`` / ``del model``)
    after this returns so the allocator can reclaim VRAM.
    """
    import gc

    if model is not None:
        try:
            if hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass
        del model

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            # Prefer empty_cache over synchronize: synchronize can raise
            # "devices are busy" during teardown and is not required to free VRAM.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _prepare_worker_env(job: dict[str, Any]) -> None:
    """Configure CUDA remapping, quiet mode, and optional per-model log file.

    Must run before any ``import torch`` / CUDA init in this process.
    """
    # Prefer an explicit physical id captured in the parent before spawn.
    visible = job.get("cuda_visible_devices")
    if visible is None:
        visible = str(job["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)
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
    try:
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
        return label, make_jsonable(summary)
    finally:
        release_cuda_memory()
        log_eval(f"Worker released gpu={gpu_id}", label=label, gpu=gpu_id)


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

    try:
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
        return label, make_jsonable(summary)
    finally:
        release_cuda_memory()
        log_eval(f"Worker released gpu={gpu_id}", label=label, gpu=gpu_id)


def _parallel_process_entry(
    worker: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    job: dict[str, Any],
    conn: Any,
) -> None:
    """Spawn-process entrypoint.

    Sends the result over ``conn``, then ``os._exit`` so PyTorch CUDA atexit
    destructors cannot raise ``CUDA-capable devices are busy`` after a successful
    eval (a common failure mode with ProcessPoolExecutor worker teardown).
    """
    exit_code = 1
    label = str(job.get("label", "?"))
    gpu_id = int(job.get("gpu_id", -1))
    try:
        result_label, summary = worker(job)
        conn.send(("ok", result_label, make_jsonable(summary)))
        exit_code = 0
    except BaseException as exc:  # noqa: BLE001 - must report any worker failure
        try:
            conn.send(("err", label, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        exit_code = 1
    finally:
        try:
            release_cuda_memory()
        except Exception:
            pass
        try:
            from harrier_distill.eval_progress import log_eval

            log_eval(f"Worker process exit gpu={gpu_id} code={exit_code}", label=label, gpu=gpu_id)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        os._exit(exit_code)


def _drain_process(
    process: Any,
    conn: Any,
    label: str,
    *,
    join_timeout: float = 120.0,
) -> tuple[str, dict[str, Any]]:
    try:
        message = conn.recv()
    except EOFError as exc:
        process.join(timeout=join_timeout)
        raise RuntimeError(f"[eval][{label}] FAILED: worker exited before sending a result") from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass

    process.join(timeout=join_timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
        raise RuntimeError(f"[eval][{label}] FAILED: worker hung after sending result")

    status = message[0]
    if status == "ok":
        return message[1], message[2]
    raise RuntimeError(f"[eval][{message[1]}] FAILED: {message[2]}")


def run_parallel_jobs(
    jobs: list[dict[str, Any]],
    *,
    worker: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run eval jobs in spawn processes; fail-fast on first error.

    Uses one process per job (wave-scheduled by unique GPU count). Workers send
    JSON-safe results over a pipe and terminate with ``os._exit`` to avoid CUDA
    atexit crashes after ``Worker released gpu`` logs.
    """
    if not jobs:
        return {}

    unique_gpus = {int(job["gpu_id"]) for job in jobs}
    workers = max_workers or len(unique_gpus)
    workers = max(1, min(workers, len(jobs), len(unique_gpus)))
    ctx = get_context("spawn")
    summaries: dict[str, dict[str, Any]] = {}

    pending = list(jobs)
    active: list[tuple[Any, Any, str]] = []

    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                parent_conn, child_conn = ctx.Pipe(duplex=False)
                process = ctx.Process(
                    target=_parallel_process_entry,
                    args=(worker, job, child_conn),
                    daemon=False,
                )
                process.start()
                child_conn.close()
                active.append((process, parent_conn, str(job["label"])))

            progressed = False
            still_active: list[tuple[Any, Any, str]] = []
            for process, conn, label in active:
                if process.is_alive() and not conn.poll():
                    still_active.append((process, conn, label))
                    continue
                progressed = True
                result_label, summary = _drain_process(process, conn, label)
                summaries[result_label] = summary

            active = still_active
            if not progressed and active:
                # Avoid busy-spin while workers are still running.
                time.sleep(0.05)
    except Exception:
        for process, conn, _label in active:
            try:
                conn.close()
            except Exception:
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
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
