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
    RETRIEVAL_SUITES,
    evaluate_retrieval,
    get_retrieval_tasks_for_suite,
    print_retrieval_summary,
    resolve_miracl_subsets_for_suite,
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
        choices=sorted(RETRIEVAL_SUITES.keys()),
        default="en_ko",
        help=(
            "Retrieval task suite. "
            "en_ko=MSMARCO+MIRACL(en,ko); miracl12=all configured MIRACL langs; "
            "all16=MSMARCO+MIRACL×12+BEIR-PL"
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
        "--emb-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache query/corpus embeddings on disk (default: on with --local-retrieval)",
    )
    parser.add_argument(
        "--refresh-emb-cache",
        action="store_true",
        help="Force re-encode and overwrite the embedding cache",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Stage lines only (disable tqdm bars)",
    )
    return parser.parse_args()


def _resolve_emb_cache_root(paths: dict, output_root: Path) -> Path | None:
    root = paths.get("retrieval_eval_data_root")
    if root is not None and str(root) != "":
        return Path(root)
    return output_root


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
    miracl_subsets = resolve_miracl_subsets_for_suite(args.suite, retrieval_cfg.get("languages"))

    use_emb_cache = args.emb_cache if args.emb_cache is not None else bool(args.local_retrieval)
    emb_cache_root = _resolve_emb_cache_root(paths, output_root) if use_emb_cache else None

    summary = evaluate_retrieval(
        args.model,
        tasks=tasks,
        query_prompt=retrieval_cfg.get("query_prompt", "web_search_query"),
        batch_size=int(retrieval_cfg.get("batch_size", eval_cfg.get("batch_size", 192))),
        output_dir=mteb_dir / label if not args.local_retrieval else None,
        miracl_subsets=miracl_subsets,
        use_local_retrieval=args.local_retrieval,
        local_task_paths=resolve_retrieval_eval_paths(cfg) if args.local_retrieval else None,
        max_length=int(cfg.get("data", {}).get("max_length", 512)),
        label=label,
        quiet=args.quiet or None,
        emb_cache_root=emb_cache_root if args.local_retrieval else None,
        use_emb_cache=use_emb_cache and args.local_retrieval,
        refresh_emb_cache=args.refresh_emb_cache,
    )
    summary["label"] = label
    summary["suite"] = args.suite

    print_retrieval_summary(summary)
    save_eval_summary(summary, summary_path)
    print(f"\nSaved retrieval eval summary -> {summary_path}")


if __name__ == "__main__":
    main()
