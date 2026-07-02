#!/usr/bin/env python3
"""Debug whether training MSE translates to embedding alignment and STS behavior."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path
from harrier_distill.debug import (
    print_alignment_summary,
    run_alignment_report,
    save_alignment_report,
)


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
        "--embeddings",
        default=None,
        help="Cached teacher embeddings parquet (default: en_embeddings or ko_embeddings from config)",
    )
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--sample-size", type=int, default=None, help="Rows to sample (default from config)")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed (default from config)")
    parser.add_argument(
        "--pruned-baseline",
        default=None,
        help="Optional pruned init checkpoint for baseline comparison",
    )
    parser.add_argument(
        "--no-sts-proxy",
        action="store_true",
        help="Skip lightweight pairwise STS proxy on STSBenchmark validation",
    )
    parser.add_argument(
        "--nli-probe",
        action="store_true",
        help="Run NLI pair ordering probe from training distribution",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for debug JSON report")
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


def resolve_embeddings_path(args: argparse.Namespace, paths: dict) -> Path:
    if args.embeddings:
        return Path(args.embeddings)
    key = f"{args.lang}_embeddings"
    return require_path(paths, key)


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    debug_cfg = cfg.get("debug", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})

    output_root = Path(args.output_dir) if args.output_dir else require_path(paths, "output_dir")
    debug_dir = output_root / "debug"
    embeddings_path = resolve_embeddings_path(args, paths)
    teacher_path = resolve_teacher_path(args, cfg, paths)
    pruned_baseline = args.pruned_baseline

    report = run_alignment_report(
        teacher_path=teacher_path,
        student_path=args.student,
        embeddings_path=embeddings_path,
        lang=args.lang,
        sample_size=int(
            args.sample_size if args.sample_size is not None else debug_cfg.get("sample_size", 5000)
        ),
        seed=int(args.seed if args.seed is not None else debug_cfg.get("seed", 42)),
        prompt_name=eval_cfg.get("prompt_name", "sts_query"),
        max_length=int(data_cfg.get("max_length", 512)),
        batch_size=int(eval_cfg.get("batch_size", 64)),
        run_sts_proxy=not args.no_sts_proxy,
        run_nli_probe=args.nli_probe,
        pruned_baseline_path=pruned_baseline,
    )

    print_alignment_summary(report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = debug_dir / f"mse_debug_{args.lang}_{timestamp}.json"
    save_alignment_report(report, summary_path)
    print(f"\nSaved debug report -> {summary_path}")


if __name__ == "__main__":
    main()
