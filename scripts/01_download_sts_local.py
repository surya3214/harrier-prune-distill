#!/usr/bin/env python3
"""Download STS benchmarks locally for offline GPU evaluation."""

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
    load_sts_datasets_config,
    parse_lang_list,
    require_path,
    should_skip_download,
    write_download_manifest,
)
from harrier_distill.mteb_sts import mteb_eng_v2_sts_task_names
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
    parser.add_argument(
        "--lang",
        default="all",
        help="Language code, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit MTEB task names to download (overrides --lang filter)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip sources whose parquet + manifest already exist (default: on)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even when outputs exist")
    return parser.parse_args()


def select_tasks(sts_cfg: dict, *, langs: set[str], tasks: list[str] | None) -> list[tuple[str, dict]]:
    if tasks:
        selected = []
        for task_name in tasks:
            if task_name not in sts_cfg:
                raise KeyError(f"Task '{task_name}' missing from STS datasets config")
            selected.append((task_name, sts_cfg[task_name]))
        return selected

    selected: list[tuple[str, dict]] = []
    for task_name, task_cfg in sts_cfg.items():
        if not isinstance(task_cfg, dict) or "sources" not in task_cfg:
            continue
        if task_cfg.get("lang") in langs:
            selected.append((task_name, task_cfg))
    return selected


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    sts_cfg = load_sts_datasets_config(args.sts_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    if args.tasks:
        langs = set()
    else:
        langs = set(parse_lang_list(args.lang))

    sts_root = local_root / "sts"
    manifest_entries: list[dict] = []

    for task_name, task_cfg in select_tasks(sts_cfg, langs=langs, tasks=args.tasks):
        default_lang = task_cfg.get("lang", "en")
        out_dir = sts_root / default_lang

        for source in task_cfg.get("sources", []):
            name = source["name"]
            lang = source.get("lang", default_lang)
            out_dir = sts_root / lang
            hf_path = source["hf_path"]
            split = source["split"]
            hf_subset = source.get("hf_subset")
            output_path = out_dir / f"{name}.parquet"
            manifest_path = out_dir / f"{name}.manifest.json"

            if (
                not args.force
                and args.skip_existing
                and output_path.exists()
                and manifest_path.exists()
            ):
                manifest = json.load(open(manifest_path, encoding="utf-8"))
                entry = dict(manifest)
                manifest_entries.append(entry)
                print(f"[skip] {task_name}/{name}: {entry.get('rows', 0):,} rows at {output_path}")
                continue

            subset_label = f" subset={hf_subset}" if hf_subset else ""
            print(f"Downloading {task_name}/{name} from {hf_path} [{split}]{subset_label} ...")
            rows = download_sts_split(
                hf_path=hf_path,
                split=split,
                lang=lang,
                task=task_name,
                hf_subset=hf_subset,
            )
            count = write_sts_parquet(rows, output_path)
            entry = {
                "lang": lang,
                "task": task_name,
                "name": name,
                "hf_path": hf_path,
                "hf_subset": hf_subset,
                "split": split,
                "rows": count,
                "path": str(output_path),
                "sha1": parquet_sha1(output_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            manifest_entries.append(entry)
            write_download_manifest(manifest_path, entry)
            print(f"  wrote {count:,} rows -> {output_path}")

    manifest_path = sts_root / "manifest.json"
    write_sts_manifest(manifest_path, manifest_entries)
    print(f"\nWrote manifest -> {manifest_path}")
    print("\nSTS download complete.")
    print(f"  root: {sts_root}")
    print(f"  MTEB(eng, v2) STS tasks: {', '.join(mteb_eng_v2_sts_task_names())}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run eval with --local-sts.")


if __name__ == "__main__":
    main()
