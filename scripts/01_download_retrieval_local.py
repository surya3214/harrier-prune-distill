#!/usr/bin/env python3
"""Download EN/KO retrieval distillation corpora on local infra (internet required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, load_retrieval_datasets_config, require_path
from harrier_distill.retrieval import build_retrieval_corpus, get_retrieval_lang_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--retrieval-config",
        default=str(PROJECT_ROOT / "configs" / "retrieval_datasets.yaml"),
    )
    parser.add_argument("--lang", choices=["en", "ko", "both"], default="both")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use pilot_sources from retrieval_datasets.yaml (mini MIRACL + 50k MS MARCO)",
    )
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

    languages, lang_cfgs = get_retrieval_lang_configs(retrieval_cfg)
    if args.lang != "both":
        languages = [args.lang]
        lang_cfgs = {args.lang: lang_cfgs[args.lang]}

    totals: dict[str, int] = {}
    for lang in languages:
        if lang not in lang_cfgs:
            raise KeyError(f"Language '{lang}' missing from retrieval config")
        output_path = local_root / "retrieval" / lang / "corpus.parquet"
        totals[lang] = build_retrieval_corpus(
            lang=lang,
            lang_cfg=lang_cfgs[lang],
            dedupe_cfg=dedupe_cfg,
            min_chars=min_chars,
            output_path=output_path,
            shard_size=shard_size,
        )
        print(f"Wrote {totals[lang]:,} rows -> {output_path}")

    print("\nRetrieval download complete.")
    for lang, count in totals.items():
        print(f"  {lang}: {count:,} rows at {local_root / 'retrieval' / lang / 'corpus.parquet'}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run retrieval GPU scripts.")


if __name__ == "__main__":
    main()
