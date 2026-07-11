"""Multi-GPU parallel helpers for compare_sts / compare_retrieval."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
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
    # Prefer an explicit physical id captured in the parent before launch.
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
        emb_cache_root = job.get("emb_cache_root")
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
            emb_cache_root=Path(emb_cache_root) if emb_cache_root else None,
            use_emb_cache=bool(job.get("use_emb_cache", True)),
            refresh_emb_cache=bool(job.get("refresh_emb_cache", False)),
            ndcg_device=job.get("ndcg_device", "auto"),
        )
        return label, make_jsonable(summary)
    finally:
        release_cuda_memory()
        log_eval(f"Worker released gpu={gpu_id}", label=label, gpu=gpu_id)


def _worker_kind(worker: Callable[..., Any]) -> str:
    name = getattr(worker, "__name__", "")
    if name == "_sts_eval_worker":
        return "sts"
    if name == "_retrieval_eval_worker":
        return "retrieval"
    if name == "_probe_eval_worker":
        return "probe"
    raise ValueError(f"Unsupported parallel worker: {worker!r}")


def _probe_eval_worker(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Test-only worker marker; real probe work runs in eval_cuda_worker."""
    return str(job["label"]), {"cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _src_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _launch_worker_subprocess(
    job: dict[str, Any],
    *,
    work_dir: Path,
) -> tuple[subprocess.Popen[Any], Path, str]:
    """Start one eval worker with CVD set before the interpreter starts."""
    from harrier_distill.eval_progress import log_eval

    label = str(job["label"])
    physical = str(job.get("cuda_visible_devices", job["gpu_id"]))
    job_path = work_dir / f"{label}.job.json"
    result_path = work_dir / f"{label}.result.json"
    job_path.write_text(json.dumps(make_jsonable(job)), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = physical
    # Ensure `python -m harrier_distill...` resolves in editable/src layouts.
    src = str(_src_root())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])

    log_eval(
        f"Launching subprocess CUDA_VISIBLE_DEVICES={physical}",
        label=label,
        gpu=int(job["gpu_id"]),
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "harrier_distill.eval_cuda_worker",
            "--job",
            str(job_path),
            "--result",
            str(result_path),
        ],
        env=env,
        stdout=None,
        stderr=None,
    )
    return process, result_path, label


def _read_worker_result(
    result_path: Path,
    label: str,
    returncode: int | None,
) -> tuple[str, dict[str, Any]]:
    if not result_path.exists():
        raise RuntimeError(
            f"[eval][{label}] FAILED: worker exited (code={returncode}) without writing a result"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(
            f"[eval][{payload.get('label', label)}] FAILED: {payload.get('error', 'unknown')}"
        )
    return str(payload["label"]), dict(payload["summary"])


def run_parallel_jobs(
    jobs: list[dict[str, Any]],
    *,
    worker: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Run eval jobs in isolated subprocesses; fail-fast on first error.

    Critical multi-GPU requirement: each child is started with
    ``CUDA_VISIBLE_DEVICES=<one gpu>`` in its environment *before* Python/torch
    import. Spawned multiprocessing children re-import the compare script and
    initialize CUDA on all parent-visible GPUs first, which causes
    ``CUDA-capable devices are busy`` when two workers run concurrently.
    """
    if not jobs:
        return {}

    kind = _worker_kind(worker)
    unique_gpus = {int(job["gpu_id"]) for job in jobs}
    workers = max_workers or len(unique_gpus)
    workers = max(1, min(workers, len(jobs), len(unique_gpus)))
    summaries: dict[str, dict[str, Any]] = {}

    pending = [dict(job, worker_kind=kind) for job in jobs]
    active: list[tuple[subprocess.Popen[Any], Path, str]] = []

    with tempfile.TemporaryDirectory(prefix="harrier-eval-parallel-") as tmp:
        work_dir = Path(tmp)
        try:
            while pending or active:
                while pending and len(active) < workers:
                    job = pending.pop(0)
                    active.append(_launch_worker_subprocess(job, work_dir=work_dir))

                progressed = False
                still_active: list[tuple[subprocess.Popen[Any], Path, str]] = []
                for process, result_path, label in active:
                    code = process.poll()
                    if code is None:
                        still_active.append((process, result_path, label))
                        continue
                    progressed = True
                    result_label, summary = _read_worker_result(result_path, label, code)
                    if code != 0:
                        raise RuntimeError(
                            f"[eval][{result_label}] FAILED: worker exited with code={code}"
                        )
                    summaries[result_label] = summary

                active = still_active
                if not progressed and active:
                    time.sleep(0.05)
        except Exception:
            for process, _result_path, _label in active:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
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
