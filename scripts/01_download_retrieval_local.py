#!/usr/bin/env python3
"""Download retrieval distillation corpora on local infra (internet required)."""

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
    load_retrieval_datasets_config,
    parse_lang_list,
    parquet_sha1,
    require_path,
    should_skip_download,
    write_download_manifest,
)
from harrier_distill.retrieval import build_retrieval_corpus, get_retrieval_lang_configs


def _max_negatives_per_query(lang_cfg: dict) -> int | None:
    """Return MIRACL negatives_per_query when present; else None for non-MIRACL langs."""
    values: list[int] = []
    for source in lang_cfg.get("sources", []):
        if "negatives_per_query" in source:
            values.append(int(source["negatives_per_query"]))
    if not values:
        return None
    return max(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--retrieval-config",
        default=str(PROJECT_ROOT / "configs" / "retrieval_datasets.yaml"),
    )
    parser.add_argument(
        "--lang",
        default="all",
        help="Language code, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use pilot_sources / scaled triplet targets from retrieval_datasets.yaml",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip languages whose retrieval corpus manifest meets target (default: on)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even when manifest is satisfied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    retrieval_cfg = load_retrieval_datasets_config(args.retrieval_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    if args.pilot:
        retrieval_cfg.setdefault("pilot", {})["enabled"] = True

    data_cfg = distill_cfg.get("data", {})
    min_chars = int(data_cfg.get("min_text_chars", 20))
    shard_size = int(retrieval_cfg.get("output", {}).get("shard_size", 50_000))
    dedupe_cfg = retrieval_cfg.get("dedupe", {})

    requested = parse_lang_list(args.lang)
    languages, lang_cfgs = get_retrieval_lang_configs(retrieval_cfg)
    languages = [lang for lang in requested if lang in languages]

    totals: dict[str, int] = {}
    skipped: list[str] = []

    for lang in languages:
        if lang not in lang_cfgs:
            raise KeyError(f"Language '{lang}' missing from retrieval config")
        output_path = local_root / "retrieval" / lang / "corpus.parquet"
        manifest_path = local_root / "retrieval" / lang / "manifest.json"
        target_triplets = int(lang_cfgs[lang].get("target_triplets", 0))
        negatives_per_query = _max_negatives_per_query(lang_cfgs[lang])

        if should_skip_download(
            output_path=output_path,
            manifest_path=manifest_path,
            target_rows=target_triplets,
            force=args.force,
            skip_existing=args.skip_existing,
            expected_negatives_per_query=negatives_per_query,
        ):
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            totals[lang] = int(manifest.get("rows", 0))
            skipped.append(lang)
            print(f"[skip] {lang}: {totals[lang]:,} rows already at {output_path}")
            continue

        totals[lang] = build_retrieval_corpus(
            lang=lang,
            lang_cfg=lang_cfgs[lang],
            dedupe_cfg=dedupe_cfg,
            min_chars=min_chars,
            output_path=output_path,
            shard_size=shard_size,
        )
        manifest_entry = {
            "lang": lang,
            "rows": totals[lang],
            "target_triplets": target_triplets,
            "path": str(output_path),
            "sha1": parquet_sha1(output_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if negatives_per_query is not None:
            manifest_entry["negatives_per_query"] = negatives_per_query
        write_download_manifest(manifest_path, manifest_entry)
        print(f"Wrote {totals[lang]:,} rows -> {output_path}")

    print("\nRetrieval download complete.")
    for lang, count in totals.items():
        status = " (skipped)" if lang in skipped else ""
        print(f"  {lang}: {count:,} rows at {local_root / 'retrieval' / lang / 'corpus.parquet'}{status}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run retrieval GPU scripts.")


if __name__ == "__main__":
    main()
