#!/usr/bin/env python3
"""Evaluate STS-B (STSBenchmark) and KorSTS for teacher/student checkpoints."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path, resolve_sts_paths
from harrier_distill.eval import evaluate_sts, print_eval_summary, save_eval_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--model", required=True, help="Checkpoint path to evaluate")
    parser.add_argument("--label", default=None, help="Label used in output filename")
    parser.add_argument("--output-dir", default=None, help="Directory for eval JSON results")
    parser.add_argument(
        "--local-sts",
        action="store_true",
        help="Evaluate from local STS parquet (offline; no MTEB/HF download)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    eval_cfg = cfg.get("eval", {})

    output_root = Path(args.output_dir) if args.output_dir else require_path(paths, "output_dir")
    eval_dir = output_root / "eval"
    mteb_dir = eval_dir / "mteb_runs"

    label = args.label or Path(args.model).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_dir / f"{label}_{timestamp}.json"

    summary = evaluate_sts(
        args.model,
        tasks=eval_cfg.get("tasks", ["STSBenchmark", "KorSTS"]),
        prompt_name=eval_cfg.get("prompt_name", "sts_query"),
        batch_size=int(eval_cfg.get("batch_size", 64)),
        output_dir=mteb_dir / label if not args.local_sts else None,
        use_local_sts=args.local_sts,
        local_task_paths=resolve_sts_paths(cfg) if args.local_sts else None,
        max_length=int(cfg.get("data", {}).get("max_length", 512)),
    )
    summary["label"] = label

    print_eval_summary(summary)
    save_eval_summary(summary, summary_path)
    print(f"\nSaved eval summary -> {summary_path}")


if __name__ == "__main__":
    main()
