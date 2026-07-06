#!/usr/bin/env python3
"""Download retrieval eval benchmarks (MSMARCO, MIRACL) for offline GPU evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_resolved_paths,
    load_distill_config,
    load_retrieval_eval_datasets_config,
    require_path,
)
from harrier_distill.retrieval_eval import download_retrieval_eval_task, write_retrieval_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--retrieval-eval-config",
        default=str(PROJECT_ROOT / "configs" / "retrieval_eval_datasets.yaml"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["MSMARCO", "MIRACLRetrieval"],
        default=None,
        help="Tasks to download (default: both)",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ko", "both"],
        default="both",
        help="MIRACL language subsets to download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    eval_cfg = load_retrieval_eval_datasets_config(args.retrieval_eval_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    tasks = args.tasks or ["MSMARCO", "MIRACLRetrieval"]
    eval_root = local_root / "retrieval_eval"
    manifest_entries: list[dict] = []

    if "MSMARCO" in tasks:
        msmarco_cfg = eval_cfg["MSMARCO"]
        output_dir = eval_root / msmarco_cfg["output_dir"]
        print(f"Downloading MSMARCO [{msmarco_cfg['split']}] -> {output_dir} ...")
        entry = download_retrieval_eval_task(
            task=msmarco_cfg["task"],
            lang=msmarco_cfg["lang"],
            split=msmarco_cfg["split"],
            hf_path=msmarco_cfg["hf_path"],
            queries_cfg=msmarco_cfg["queries"],
            corpus_cfg=msmarco_cfg["corpus"],
            qrels_cfg=msmarco_cfg["qrels"],
            output_dir=output_dir,
        )
        manifest_entries.append(entry)
        print(
            f"  wrote {entry['query_count']:,} queries, "
            f"{entry['corpus_count']:,} docs, {entry['qrel_count']:,} qrels"
        )

    if "MIRACLRetrieval" in tasks:
        miracl_cfg = eval_cfg["MIRACLRetrieval"]
        langs = ["en", "ko"] if args.lang == "both" else [args.lang]
        for lang in langs:
            if lang not in miracl_cfg["languages"]:
                raise KeyError(f"Language '{lang}' missing from MIRACLRetrieval config")
            lang_cfg = miracl_cfg["languages"][lang]
            output_dir = eval_root / lang_cfg["output_dir"]
            print(f"Downloading MIRACLRetrieval/{lang} [{miracl_cfg['split']}] -> {output_dir} ...")
            entry = download_retrieval_eval_task(
                task=miracl_cfg["task"],
                lang=lang_cfg["lang"],
                split=miracl_cfg["split"],
                hf_path=miracl_cfg["hf_path"],
                queries_cfg=lang_cfg["queries"],
                corpus_cfg=lang_cfg["corpus"],
                qrels_cfg=lang_cfg["qrels"],
                output_dir=output_dir,
            )
            manifest_entries.append(entry)
            print(
                f"  wrote {entry['query_count']:,} queries, "
                f"{entry['corpus_count']:,} docs, {entry['qrel_count']:,} qrels"
            )

    manifest_path = eval_root / "manifest.json"
    write_retrieval_manifest(manifest_path, manifest_entries)
    print(f"\nWrote manifest -> {manifest_path}")
    print("\nRetrieval eval download complete.")
    print(f"  root: {eval_root}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run eval with --local-retrieval.")


if __name__ == "__main__":
    main()
