#!/usr/bin/env python3
"""Download and prepare distillation corpora on local infra (internet required)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    apply_sample_overrides,
    get_resolved_paths,
    load_datasets_config,
    load_distill_config,
    parse_lang_list,
    parquet_sha1,
    require_path,
    should_skip_download,
    write_download_manifest,
)
from harrier_distill.data import (
    MULTILINGUAL_NLI_HF_PATH,
    append_corpus_shard,
    corpus_row,
    ensure_dir,
    load_hf_source_dataset,
    merge_parquet_shards,
    resolve_hf_source_splits,
)
from harrier_distill.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--datasets-config", default=str(PROJECT_ROOT / "configs" / "datasets.yaml"))
    parser.add_argument(
        "--lang",
        default="all",
        help="Language code, comma-separated list, or 'all' (from languages.yaml)",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip languages whose corpus.parquet manifest meets target_samples (default: on)",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even when manifest is satisfied")
    return parser.parse_args()


class DedupeState:
    def __init__(self, *, lowercase: bool = False, normalize_whitespace: bool = True):
        self.seen: set[str] = set()
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace

    def key(self, text: str) -> str:
        cleaned = normalize_text(text, normalize_whitespace=self.normalize_whitespace)
        return cleaned.lower() if self.lowercase else cleaned

    def add_if_new(self, text: str) -> bool:
        key = self.key(text)
        if not key or key in self.seen:
            return False
        self.seen.add(key)
        return True


def text_from_row(row: dict, text_column: str | None, text_columns: list[str] | None) -> list[str]:
    if text_columns:
        return [row[col] for col in text_columns if row.get(col)]
    if text_column:
        value = row.get(text_column)
        return [value] if value else []
    return []


def passes_lang_filter(row: dict, source_cfg: dict) -> bool:
    filter_lang = source_cfg.get("filter_lang")
    if not filter_lang:
        return True
    hf_path = source_cfg.get("hf_path", "")
    if hf_path == MULTILINGUAL_NLI_HF_PATH or source_cfg.get("split_resolver") == "multilingual_nli":
        return True
    filter_column = source_cfg.get("filter_column", "language")
    value = row.get(filter_column)
    if value is None:
        return False
    return str(value).lower() == str(filter_lang).lower()


def reservoir_id(lang: str, source: str, text: str) -> str:
    digest = hashlib.sha1(f"{lang}:{source}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{lang}_{source}_{digest}"


def stream_hf_dataset(source_cfg: dict, *, split: str):
    return load_hf_source_dataset(source_cfg, split=split)


def iter_hf_rows(source_cfg: dict, lang: str):
    from itertools import chain

    splits = resolve_hf_source_splits(source_cfg, lang)
    return chain.from_iterable(stream_hf_dataset(source_cfg, split=split_name) for split_name in splits)


def collect_from_source(
    *,
    lang: str,
    source_cfg: dict,
    dedupe: DedupeState,
    min_chars: int,
    lang_target: int,
    collected_rows: list[dict],
) -> int:
    name = source_cfg["name"]
    source_target = min(int(source_cfg.get("target_samples", lang_target)), lang_target)
    text_column = source_cfg.get("text_column")
    text_columns = source_cfg.get("text_columns")
    optional = bool(source_cfg.get("optional", False))

    if len(collected_rows) >= lang_target:
        return 0

    split_names = resolve_hf_source_splits(source_cfg, lang)
    split_label = split_names[0] if len(split_names) == 1 else f"{len(split_names)} splits"
    progress = tqdm(total=source_target, desc=f"{lang}/{name} ({split_label})")
    added = 0

    try:
        dataset = iter_hf_rows(source_cfg, lang)
        for row in dataset:
            if len(collected_rows) >= lang_target or added >= source_target:
                break
            if not passes_lang_filter(row, source_cfg):
                continue

            texts = text_from_row(row, text_column, text_columns)
            for text in texts:
                if len(collected_rows) >= lang_target or added >= source_target:
                    break
                if not dedupe.add_if_new(text):
                    continue
                record = corpus_row(
                    row_id=reservoir_id(lang, name, text),
                    text=text,
                    lang=lang,
                    source=name,
                    min_chars=min_chars,
                    normalize_whitespace=True,
                )
                if record is None:
                    continue
                collected_rows.append(record)
                added += 1
                progress.update(1)
    except Exception as exc:
        progress.close()
        if optional:
            print(f"[WARN] Skipping optional source {name}: {exc}")
            return 0
        raise

    progress.close()
    print(f"  {lang}/{name}: added {added:,} rows")
    return added


def flush_shard(rows: list[dict], shard_dir: Path, shard_idx: int, shard_size: int) -> tuple[list[dict], int]:
    while len(rows) >= shard_size:
        shard = rows[:shard_size]
        rows = rows[shard_size:]
        shard_path = shard_dir / f"part_{shard_idx:05d}.parquet"
        append_corpus_shard(shard, shard_path)
        shard_idx += 1
    return rows, shard_idx


def build_language_corpus(
    *,
    lang: str,
    lang_cfg: dict,
    dedupe_cfg: dict,
    min_chars: int,
    output_path: Path,
    shard_size: int,
) -> int:
    dedupe = DedupeState(
        lowercase=bool(dedupe_cfg.get("lowercase_for_dedupe", False)),
        normalize_whitespace=bool(dedupe_cfg.get("normalize_whitespace", True)),
    )

    lang_target = int(lang_cfg["target_samples"])
    rows: list[dict] = []
    shard_dir = output_path.parent / f".{lang}_shards"
    if shard_dir.exists():
        for old_shard in shard_dir.glob("part_*.parquet"):
            old_shard.unlink()
    ensure_dir(shard_dir)

    shard_idx = 0
    for source_cfg in lang_cfg.get("sources", []):
        collect_from_source(
            lang=lang,
            source_cfg=source_cfg,
            dedupe=dedupe,
            min_chars=min_chars,
            lang_target=lang_target,
            collected_rows=rows,
        )
        rows, shard_idx = flush_shard(rows, shard_dir, shard_idx, shard_size)
        if len(rows) >= lang_target:
            break

    if rows:
        append_corpus_shard(rows, shard_dir / f"part_{shard_idx:05d}.parquet")

    shard_paths = sorted(shard_dir.glob("part_*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No rows collected for language '{lang}'")

    total_rows = merge_parquet_shards(shard_dir, output_path)
    print(f"Wrote {total_rows:,} rows -> {output_path}")
    return total_rows


def main() -> None:
    args = parse_args()
    distill_cfg = load_distill_config(args.config)
    datasets_cfg = load_datasets_config(args.datasets_config)
    paths = get_resolved_paths(distill_cfg)
    local_root = require_path(paths, "local_data_root")

    langs = parse_lang_list(args.lang)
    apply_sample_overrides(datasets_cfg, distill_cfg, langs=langs)

    data_cfg = distill_cfg.get("data", {})
    min_chars = int(data_cfg.get("min_text_chars", 20))
    shard_size = int(datasets_cfg.get("output", {}).get("shard_size", 100_000))
    dedupe_cfg = datasets_cfg.get("dedupe", {})

    totals: dict[str, int] = {}
    skipped: list[str] = []

    for lang in langs:
        if lang not in datasets_cfg:
            raise KeyError(f"Language '{lang}' missing from datasets config")
        output_path = local_root / lang / "corpus.parquet"
        manifest_path = local_root / lang / "manifest.json"
        target_samples = int(datasets_cfg[lang]["target_samples"])

        if should_skip_download(
            output_path=output_path,
            manifest_path=manifest_path,
            target_rows=target_samples,
            force=args.force,
            skip_existing=args.skip_existing,
        ):
            manifest = __import__("json").load(open(manifest_path, encoding="utf-8"))
            totals[lang] = int(manifest.get("rows", 0))
            skipped.append(lang)
            print(f"[skip] {lang}: {totals[lang]:,} rows already at {output_path}")
            continue

        totals[lang] = build_language_corpus(
            lang=lang,
            lang_cfg=datasets_cfg[lang],
            dedupe_cfg=dedupe_cfg,
            min_chars=min_chars,
            output_path=output_path,
            shard_size=shard_size,
        )
        write_download_manifest(
            manifest_path,
            {
                "lang": lang,
                "rows": totals[lang],
                "target_samples": target_samples,
                "path": str(output_path),
                "sha1": parquet_sha1(output_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    print("\nDownload complete.")
    for lang, count in totals.items():
        status = " (skipped)" if lang in skipped else ""
        print(f"  {lang}: {count:,} rows at {local_root / lang / 'corpus.parquet'}{status}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run GPU scripts.")


if __name__ == "__main__":
    main()
