"""Progress logging helpers for STS and retrieval evaluation."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def is_quiet(*, quiet: bool | None = None) -> bool:
    if quiet is not None:
        return quiet
    return os.environ.get("EVAL_QUIET", "").strip().lower() in {"1", "true", "yes"}


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def log_eval(
    message: str,
    *,
    label: str | None = None,
    gpu: int | None = None,
    file: Any | None = None,
) -> None:
    """Print a flushed timestamped eval progress line."""
    parts = ["[eval]"]
    if label:
        parts.append(f"[{label}]")
    if gpu is not None:
        parts.append(f"[gpu={gpu}]")
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"{stamp} {' '.join(parts)} {message}"
    out = file or sys.stdout
    print(line, file=out, flush=True)


def task_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    disable: bool | None = None,
    quiet: bool | None = None,
) -> Iterator[T]:
    """Wrap an iterable with tqdm unless quiet mode is enabled."""
    disabled = is_quiet(quiet=quiet) if disable is None else disable
    if disabled:
        yield from iterable
        return

    from tqdm import tqdm

    yield from tqdm(iterable, desc=desc, total=total, leave=False)


class StageTimer:
    """Simple wall-clock timer for eval stages."""

    def __init__(self) -> None:
        self.started = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def elapsed_str(self) -> str:
        return format_elapsed(self.elapsed())
