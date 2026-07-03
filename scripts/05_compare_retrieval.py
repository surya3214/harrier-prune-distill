#!/usr/bin/env python3
"""Compare teacher vs student retrieval scores on MSMARCO and MIRACL."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path
from harrier_distill.eval import (
    _miracl_eval_subsets,
    compare_retrieval,
    print_retrieval_compare_summary,
    save_eval_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--teacher", default=None, help="Teacher checkpoint (defaults to eval.compare_teacher)")
    parser.add_argument("--student", required=True, help="Student checkpoint to compare")
    parser.add_argument("--baseline", default=None, help="Optional baseline checkpoint")
    parser.add_argument("--output-dir", default=None, help="Directory for comparison JSON")
    parser.add_argument(
        "--suite",
        choices=["en", "ko", "en_ko"],
        default="en_ko",
        help="Retrieval task suite",
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
    summary_path = eval_dir / f"compare_{timestamp}.json"

    teacher_path = args.teacher or eval_cfg.get("compare_teacher", "microsoft/harrier-oss-v1-270m")

    comparison = compare_retrieval(
        teacher_path=teacher_path,
        student_path=args.student,
        baseline_path=args.baseline,
        suite=args.suite,
        query_prompt=retrieval_cfg.get("query_prompt", "web_search_query"),
        batch_size=int(retrieval_cfg.get("batch_size", eval_cfg.get("batch_size", 64))),
        output_dir=eval_dir / "mteb_runs",
        miracl_subsets=_miracl_eval_subsets(retrieval_cfg.get("languages")),
    )

    print_retrieval_compare_summary(comparison)
    save_eval_summary(comparison, summary_path)
    print(f"\nSaved retrieval comparison -> {summary_path}")


if __name__ == "__main__":
    main()
