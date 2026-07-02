#!/usr/bin/env python3
"""Download EN/KO STS benchmarks locally for offline GPU evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_resolved_paths,
    load_distill_config,
    load_sts_datasets_config,
    require_path,
)
from harrier_distill.sts import (
    download_sts_split,
    parquet_sha1,
    write_sts_manifest,
    write_sts_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--sts-config",
        default=str(PROJECT_ROOT / "configs" / "sts_datasets.yaml"),
    )
    parser.add_argument("--lang", choices=["en", "ko", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    sts_cfg = load_sts_datasets_config(args.sts_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    sts_root = local_root / "sts"
    langs = ["en", "ko"] if args.lang == "both" else [args.lang]
    manifest_entries: list[dict] = []

    for lang in langs:
        if lang not in sts_cfg:
            raise KeyError(f"Language '{lang}' missing from STS datasets config")
        lang_cfg = sts_cfg[lang]
        task = lang_cfg["task"]
        out_dir = sts_root / lang

        for source in lang_cfg.get("sources", []):
            name = source["name"]
            hf_path = source["hf_path"]
            split = source["split"]
            output_path = out_dir / f"{name}.parquet"

            print(f"Downloading {lang}/{name} from {hf_path} [{split}] ...")
            rows = download_sts_split(
                hf_path=hf_path,
                split=split,
                lang=lang,
                task=task,
            )
            count = write_sts_parquet(rows, output_path)
            entry = {
                "lang": lang,
                "task": task,
                "name": name,
                "hf_path": hf_path,
                "split": split,
                "rows": count,
                "path": str(output_path),
                "sha1": parquet_sha1(output_path),
            }
            manifest_entries.append(entry)
            print(f"  wrote {count:,} rows -> {output_path}")

    manifest_path = sts_root / "manifest.json"
    write_sts_manifest(manifest_path, manifest_entries)
    print(f"\nWrote manifest -> {manifest_path}")
    print("\nSTS download complete.")
    print(f"  root: {sts_root}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run eval with --local-sts.")


if __name__ == "__main__":
    main()
