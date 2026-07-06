#!/usr/bin/env python3
"""Evaluate retrieval tasks (MSMARCO, MIRACL) for teacher/student checkpoints."""

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
    _miracl_eval_subsets,
    evaluate_retrieval,
    get_retrieval_tasks_for_suite,
    print_retrieval_summary,
    save_eval_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--model", required=True, help="Checkpoint path to evaluate")
    parser.add_argument("--label", default=None, help="Label used in output filename")
    parser.add_argument("--output-dir", default=None, help="Directory for eval JSON results")
    parser.add_argument(
        "--suite",
        choices=["en", "ko", "en_ko"],
        default="en_ko",
        help="Retrieval task suite",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    eval_cfg = cfg.get("eval", {})
    retrieval_cfg = eval_cfg.get("retrieval", {})

    output_root = Path(args.output_dir) if args.output_dir else require_path(paths, "output_dir")
    eval_dir = output_root / "eval" / "retrieval"
    mteb_dir = eval_dir / "mteb_runs"

    label = args.label or Path(args.model).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = eval_dir / f"{label}_{timestamp}.json"

    tasks = get_retrieval_tasks_for_suite(args.suite, tasks=args.tasks)

    summary = evaluate_retrieval(
        args.model,
        tasks=tasks,
        query_prompt=retrieval_cfg.get("query_prompt", "web_search_query"),
        batch_size=int(retrieval_cfg.get("batch_size", eval_cfg.get("batch_size", 64))),
        output_dir=mteb_dir / label if not args.local_retrieval else None,
        miracl_subsets=_miracl_eval_subsets(retrieval_cfg.get("languages")),
        use_local_retrieval=args.local_retrieval,
        local_task_paths=resolve_retrieval_eval_paths(cfg) if args.local_retrieval else None,
        max_length=int(cfg.get("data", {}).get("max_length", 512)),
    )
    summary["label"] = label
    summary["suite"] = args.suite

    print_retrieval_summary(summary)
    save_eval_summary(summary, summary_path)
    print(f"\nSaved retrieval eval summary -> {summary_path}")


if __name__ == "__main__":
    main()
