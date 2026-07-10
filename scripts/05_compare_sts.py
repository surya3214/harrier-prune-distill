#!/usr/bin/env python3
"""Compare teacher and student checkpoints on EN or multilingual STS suites."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path, resolve_sts_paths
from harrier_distill.eval import STS_SUITES, compare_sts, print_compare_summary, save_eval_summary
from harrier_distill.eval_parallel import parse_gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--student", required=True, help="Local distilled checkpoint path")
    parser.add_argument(
        "--teacher",
        default=None,
        help="Teacher model path or HF ID (default: config eval.compare_teacher or teacher_model path)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional pruned baseline checkpoint for 3-way comparison",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(STS_SUITES.keys()),
        default="multilingual",
        help="STS task preset",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit MTEB task names (overrides --suite)",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for comparison JSON and MTEB runs")
    parser.add_argument(
        "--local-sts",
        action="store_true",
        help="Evaluate from local STS parquet (offline; no MTEB/HF download)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Evaluate teacher/student/baseline on separate GPUs in parallel",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for --parallel (order: teacher,student,baseline)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max concurrent GPU workers (default: number of assigned GPUs)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Stage lines only (disable tqdm bars)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional directory for per-model logs when using --parallel",
    )
    return parser.parse_args()


def resolve_teacher_path(args: argparse.Namespace, cfg: dict, paths: dict) -> str:
    if args.teacher:
        return args.teacher
    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("compare_teacher"):
        return str(eval_cfg["compare_teacher"])
    teacher_path = paths.get("teacher_model")
    if teacher_path is not None and str(teacher_path) != "":
        return str(teacher_path)
    return "microsoft/harrier-oss-v1-270m"


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    eval_cfg = cfg.get("eval", {})

    output_root = Path(args.output_dir) if args.output_dir else require_path(paths, "output_dir")
    eval_dir = output_root / "eval"
    teacher_path = resolve_teacher_path(args, cfg, paths)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    comparison = compare_sts(
        teacher_path=teacher_path,
        student_path=args.student,
        baseline_path=args.baseline,
        suite=args.suite,
        tasks=args.tasks,
        prompt_name=eval_cfg.get("prompt_name", "sts_query"),
        batch_size=int(eval_cfg.get("batch_size", 64)),
        output_dir=eval_dir,
        use_local_sts=args.local_sts,
        local_task_paths=resolve_sts_paths(cfg) if args.local_sts else None,
        max_length=int(cfg.get("data", {}).get("max_length", 512)),
        parallel=args.parallel,
        gpu_ids=parse_gpu_ids(args.gpus),
        max_workers=args.max_workers,
        quiet=args.quiet or None,
    )

    print_compare_summary(comparison)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_dir / f"compare_{args.suite}_{timestamp}.json"
    save_eval_summary(comparison, summary_path)
    print(f"\nSaved comparison summary -> {summary_path}")


if __name__ == "__main__":
    main()
