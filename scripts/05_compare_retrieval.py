#!/usr/bin/env python3
"""Compare teacher vs student retrieval scores on MSMARCO and MIRACL."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_resolved_paths,
    load_distill_config,
    require_path,
    resolve_retrieval_eval_paths,
)
from harrier_distill.eval import (
    RETRIEVAL_SUITES,
    compare_retrieval,
    print_retrieval_compare_summary,
    resolve_miracl_subsets_for_suite,
    save_eval_summary,
)
from harrier_distill.eval_parallel import parse_gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--teacher", default=None, help="Teacher checkpoint (defaults to eval.compare_teacher)")
    parser.add_argument("--student", required=True, help="Student checkpoint to compare")
    parser.add_argument("--baseline", default=None, help="Optional baseline checkpoint")
    parser.add_argument("--output-dir", default=None, help="Directory for comparison JSON")
    parser.add_argument(
        "--suite",
        choices=sorted(RETRIEVAL_SUITES.keys()),
        default="en_ko",
        help=(
            "Retrieval task suite. "
            "en_ko=MSMARCO+MIRACL(en,ko); miracl12=MIRACL all configured langs; "
            "all16=MSMARCO+MIRACL×12+BEIR-PL. Note: it/pt/vi have no retrieval eval tasks."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit MTEB task names (overrides --suite)",
    )
    parser.add_argument(
        "--local-retrieval",
        action="store_true",
        help="Evaluate from local retrieval parquet (offline; no MTEB/HF download)",
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


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    eval_cfg = cfg.get("eval", {})
    retrieval_cfg = eval_cfg.get("retrieval", {})

    output_root = Path(args.output_dir) if args.output_dir else require_path(paths, "output_dir")
    eval_dir = output_root / "eval" / "retrieval"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_dir / f"compare_{args.suite}_{timestamp}.json"

    teacher_path = args.teacher or eval_cfg.get("compare_teacher", "microsoft/harrier-oss-v1-270m")

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    miracl_subsets = resolve_miracl_subsets_for_suite(
        args.suite,
        retrieval_cfg.get("languages"),
    )

    comparison = compare_retrieval(
        teacher_path=teacher_path,
        student_path=args.student,
        baseline_path=args.baseline,
        suite=args.suite,
        tasks=args.tasks,
        query_prompt=retrieval_cfg.get("query_prompt", "web_search_query"),
        batch_size=int(retrieval_cfg.get("batch_size", eval_cfg.get("batch_size", 64))),
        output_dir=eval_dir / "mteb_runs",
        miracl_subsets=miracl_subsets,
        use_local_retrieval=args.local_retrieval,
        local_task_paths=resolve_retrieval_eval_paths(cfg) if args.local_retrieval else None,
        max_length=int(cfg.get("data", {}).get("max_length", 512)),
        parallel=args.parallel,
        gpu_ids=parse_gpu_ids(args.gpus),
        max_workers=args.max_workers,
        quiet=args.quiet or None,
    )

    print_retrieval_compare_summary(comparison)
    save_eval_summary(comparison, summary_path)
    print(f"\nSaved retrieval comparison -> {summary_path}")


if __name__ == "__main__":
    main()
