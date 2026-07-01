#!/usr/bin/env python3
"""Download and prepare EN/KO distillation corpora on local infra (internet required)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import load_datasets_config, load_distill_config, get_resolved_paths, require_path
from harrier_distill.data import append_corpus_shard, corpus_row, ensure_dir, merge_parquet_shards
from harrier_distill.text import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--datasets-config", default=str(PROJECT_ROOT / "configs" / "datasets.yaml"))
    parser.add_argument("--lang", choices=["en", "ko", "both"], default="both")
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


def reservoir_id(lang: str, source: str, text: str) -> str:
    digest = hashlib.sha1(f"{lang}:{source}:{text}".encode("utf-8")).hexdigest()[:16]
    return f"{lang}_{source}_{digest}"


def stream_hf_dataset(source_cfg: dict):
    from datasets import load_dataset

    hf_path = source_cfg["hf_path"]
    config = source_cfg.get("config")
    split = source_cfg.get("split", "train")
    streaming = source_cfg.get("streaming", False)

    kwargs = {"path": hf_path, "split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config

    try:
        return load_dataset(**kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message:
            hints = {
                "mc4": "Use hf_path: allenai/c4 with the same language config (e.g. config: ko).",
            }
            hint = hints.get(hf_path, "Pick a Parquet-native dataset on Hugging Face Hub.")
            raise RuntimeError(f"{message} Dataset '{hf_path}' uses a legacy loading script. {hint}") from exc
        raise


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

    try:
        dataset = stream_hf_dataset(source_cfg)
    except Exception as exc:
        if optional:
            print(f"[WARN] Skipping optional source {name}: {exc}")
            return 0
        raise

    added = 0
    progress = tqdm(total=source_target, desc=f"{lang}/{name}")

    for row in dataset:
        if len(collected_rows) >= lang_target or added >= source_target:
            break

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

    data_cfg = distill_cfg.get("data", {})
    min_chars = int(data_cfg.get("min_text_chars", 20))
    shard_size = int(datasets_cfg.get("output", {}).get("shard_size", 100_000))
    dedupe_cfg = datasets_cfg.get("dedupe", {})

    # Full run only: pilot_samples_per_lang=0 uses datasets.yaml target_samples.
    pilot_per_lang = int(data_cfg.get("pilot_samples_per_lang", 0))
    if pilot_per_lang > 0:
        for lang_key in ("en", "ko"):
            if lang_key in datasets_cfg:
                datasets_cfg[lang_key]["target_samples"] = pilot_per_lang

    langs = ["en", "ko"] if args.lang == "both" else [args.lang]
    totals: dict[str, int] = {}

    for lang in langs:
        if lang not in datasets_cfg:
            raise KeyError(f"Language '{lang}' missing from datasets config")
        output_path = local_root / lang / "corpus.parquet"
        totals[lang] = build_language_corpus(
            lang=lang,
            lang_cfg=datasets_cfg[lang],
            dedupe_cfg=dedupe_cfg,
            min_chars=min_chars,
            output_path=output_path,
            shard_size=shard_size,
        )

    print("\nDownload complete.")
    for lang, count in totals.items():
        print(f"  {lang}: {count:,} rows at {local_root / lang / 'corpus.parquet'}")
    print("\nNext: rsync local_data_root to gpu_data_root, then run GPU scripts.")


if __name__ == "__main__":
    main()
