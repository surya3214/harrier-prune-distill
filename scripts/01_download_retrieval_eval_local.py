#!/usr/bin/env python3
"""Download retrieval eval benchmarks (MSMARCO, MIRACL, BEIR-PL) for offline GPU evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_resolved_paths,
    load_distill_config,
    load_retrieval_eval_datasets_config,
    parse_lang_list,
    require_path,
    write_download_manifest,
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
        default=None,
        help="Tasks to download (default: MSMARCO, MIRACLRetrieval, BEIR-PL)",
    )
    parser.add_argument(
        "--lang",
        default="all",
        help="MIRACL language subsets to download, or 'all' for all MIRACL langs in config",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks whose output manifest exists (default: on)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even when outputs exist")
    return parser.parse_args()


def _manifest_satisfied(manifest_path: Path, output_dir: Path) -> bool:
    if not manifest_path.exists() or not output_dir.exists():
        return False
    required = ["queries.parquet", "corpus.parquet", "qrels.parquet"]
    return all((output_dir / name).exists() for name in required)


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    eval_cfg = load_retrieval_eval_datasets_config(args.retrieval_eval_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    tasks = args.tasks or ["MSMARCO", "MIRACLRetrieval", "BEIR-PL"]
    eval_root = local_root / "retrieval_eval"
    manifest_entries: list[dict] = []

    if "MSMARCO" in tasks:
        msmarco_cfg = eval_cfg["MSMARCO"]
        output_dir = eval_root / msmarco_cfg["output_dir"]
        manifest_path = output_dir / "manifest.json"
        if not args.force and args.skip_existing and _manifest_satisfied(manifest_path, output_dir):
            entry = json.load(open(manifest_path, encoding="utf-8"))
            manifest_entries.append(entry)
            print(f"[skip] MSMARCO -> {output_dir}")
        else:
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
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            write_download_manifest(manifest_path, entry)
            manifest_entries.append(entry)
            print(
                f"  wrote {entry['query_count']:,} queries, "
                f"{entry['corpus_count']:,} docs, {entry['qrel_count']:,} qrels"
            )

    if "MIRACLRetrieval" in tasks:
        miracl_cfg = eval_cfg["MIRACLRetrieval"]
        miracl_langs = list(miracl_cfg["languages"].keys())
        if args.lang != "all":
            miracl_langs = [lang for lang in parse_lang_list(args.lang) if lang in miracl_langs]

        for lang in miracl_langs:
            lang_cfg = miracl_cfg["languages"][lang]
            output_dir = eval_root / lang_cfg["output_dir"]
            manifest_path = output_dir / "manifest.json"
            if not args.force and args.skip_existing and _manifest_satisfied(manifest_path, output_dir):
                entry = json.load(open(manifest_path, encoding="utf-8"))
                manifest_entries.append(entry)
                print(f"[skip] MIRACLRetrieval/{lang} -> {output_dir}")
                continue

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
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            write_download_manifest(manifest_path, entry)
            manifest_entries.append(entry)
            print(
                f"  wrote {entry['query_count']:,} queries, "
                f"{entry['corpus_count']:,} docs, {entry['qrel_count']:,} qrels"
            )

    if "BEIR-PL" in tasks and "BEIR-PL" in eval_cfg:
        beir_cfg = eval_cfg["BEIR-PL"]
        output_dir = eval_root / beir_cfg["output_dir"]
        manifest_path = output_dir / "manifest.json"
        if not args.force and args.skip_existing and _manifest_satisfied(manifest_path, output_dir):
            entry = json.load(open(manifest_path, encoding="utf-8"))
            manifest_entries.append(entry)
            print(f"[skip] BEIR-PL -> {output_dir}")
        else:
            print(f"Downloading BEIR-PL [{beir_cfg['split']}] -> {output_dir} ...")
            entry = download_retrieval_eval_task(
                task=beir_cfg["task"],
                lang=beir_cfg["lang"],
                split=beir_cfg["split"],
                hf_path=beir_cfg["hf_path"],
                queries_cfg=beir_cfg["queries"],
                corpus_cfg=beir_cfg["corpus"],
                qrels_cfg=beir_cfg["qrels"],
                output_dir=output_dir,
            )
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            write_download_manifest(manifest_path, entry)
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
