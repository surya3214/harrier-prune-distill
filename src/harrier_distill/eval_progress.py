"""Progress logging helpers for STS and retrieval evaluation."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, TextIO, TypeVar

T = TypeVar("T")

_LOG_FILE_ENV = "EVAL_LOG_FILE"


def is_quiet(*, quiet: bool | None = None) -> bool:
    if quiet is not None:
        return quiet
    return os.environ.get("EVAL_QUIET", "").strip().lower() in {"1", "true", "yes"}


def eval_log_path() -> str | None:
    path = os.environ.get(_LOG_FILE_ENV, "").strip()
    return path or None


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _format_eval_line(
    message: str,
    *,
    label: str | None = None,
    gpu: int | None = None,
) -> str:
    parts = ["[eval]"]
    if label:
        parts.append(f"[{label}]")
    if gpu is not None:
        parts.append(f"[gpu={gpu}]")
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    return f"{stamp} {' '.join(parts)} {message}"


def log_eval(
    message: str,
    *,
    label: str | None = None,
    gpu: int | None = None,
    file: Any | None = None,
) -> None:
    """Print a flushed timestamped eval progress line.

    When ``EVAL_LOG_FILE`` is set, the same line is also appended there
    (unless ``file`` already targets that path).
    """
    line = _format_eval_line(message, label=label, gpu=gpu)
    out = file or sys.stdout
    print(line, file=out, flush=True)

    log_path = eval_log_path()
    if not log_path:
        return
    # Avoid double-writing when caller already targets the log file.
    if file is not None and getattr(file, "name", None) == log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as log_fp:
            print(line, file=log_fp, flush=True)
    except OSError:
        pass


def task_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    disable: bool | None = None,
    quiet: bool | None = None,
    file: TextIO | None = None,
) -> Iterator[T]:
    """Wrap an iterable with tqdm unless quiet mode is enabled.

    Under quiet/parallel mode, if ``EVAL_LOG_FILE`` is set, tqdm still writes
    full detail into that file (not shared stdout).
    """
    disabled = is_quiet(quiet=quiet) if disable is None else disable
    log_path = eval_log_path()

    if disabled and not log_path:
        yield from iterable
        return

    from tqdm import tqdm

    if disabled and log_path:
        # Parallel + --log-dir: bars go to the per-model log only.
        with open(log_path, "a", encoding="utf-8") as log_fp:
            yield from tqdm(
                iterable,
                desc=desc,
                total=total,
                leave=False,
                file=log_fp,
            )
        return

    yield from tqdm(iterable, desc=desc, total=total, leave=False, file=file)


class StageTimer:
    """Simple wall-clock timer for eval stages."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def elapsed_str(self) -> str:
        return format_elapsed(self.elapsed())


class StageTimer:
    """Simple wall-clock timer for eval stages."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def elapsed_str(self) -> str:
        return format_elapsed(self.elapsed())
