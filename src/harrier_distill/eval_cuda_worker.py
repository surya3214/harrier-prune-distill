"""Subprocess entrypoint for one-GPU compare eval workers.

Parent must launch this module with ``CUDA_VISIBLE_DEVICES`` already set in the
environment so torch is never imported while multiple GPUs are visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to JSON job payload")
    parser.add_argument("--result", required=True, help="Path to write JSON result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    job_path = Path(args.job)
    result_path = Path(args.result)
    job = json.loads(job_path.read_text(encoding="utf-8"))

    # Parent is responsible for setting this before process start. Re-assert so a
    # misconfigured launcher fails loudly instead of grabbing every GPU.
    expected = str(job.get("cuda_visible_devices", job.get("gpu_id", "")))
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current != expected:
        os.environ["CUDA_VISIBLE_DEVICES"] = expected

    kind = str(job.get("worker_kind", "sts"))
    label = str(job.get("label", "?"))
    gpu_id = int(job.get("gpu_id", -1))

    exit_code = 1
    payload: dict = {"ok": False, "label": label}
    try:
        if kind == "probe":
            # No torch import — used by unit tests to verify CVD isolation.
            payload = {
                "ok": True,
                "label": label,
                "summary": {
                    "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_id": gpu_id,
                },
            }
            exit_code = 0
        else:
            # Import torch-using code only after CVD is correct for this process.
            from harrier_distill.eval_parallel import _retrieval_eval_worker, _sts_eval_worker
            from harrier_distill.eval_progress import log_eval

            log_eval(
                f"Subprocess worker kind={kind} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
                label=label,
                gpu=gpu_id,
            )
            if kind == "sts":
                result_label, summary = _sts_eval_worker(job)
            elif kind == "retrieval":
                result_label, summary = _retrieval_eval_worker(job)
            else:
                raise ValueError(f"Unknown worker_kind={kind!r}")
            payload = {"ok": True, "label": result_label, "summary": summary}
            exit_code = 0
    except BaseException as exc:  # noqa: BLE001 - report any failure to parent
        payload = {
            "ok": False,
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
        }
        exit_code = 1
    finally:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            from harrier_distill.eval_progress import log_eval

            log_eval(f"Subprocess exit code={exit_code}", label=label, gpu=gpu_id)
        except Exception:
            pass
        # Skip CUDA atexit destructors that raise "devices are busy".
        os._exit(exit_code)


if __name__ == "__main__":
    main()
