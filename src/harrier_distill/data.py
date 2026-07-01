from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from harrier_distill.text import normalize_text


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(rows: Iterator[dict[str, Any]], output_path: Path) -> int:
    ensure_dir(output_path.parent)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_corpus_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="zstd")


def append_corpus_shard(rows: list[dict[str, Any]], shard_path: Path) -> None:
    ensure_dir(shard_path.parent)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, shard_path, compression="zstd")


def merge_parquet_shards(shard_dir: Path, output_path: Path, pattern: str = "part_*.parquet") -> int:
    shard_paths = sorted(shard_dir.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found in {shard_dir} matching {pattern}")

    tables = [pq.read_table(path) for path in shard_paths]
    merged = pa.concat_tables(tables, promote_options="default")
    ensure_dir(output_path.parent)
    pq.write_table(merged, output_path, compression="zstd")
    return merged.num_rows


def load_corpus_table(path: Path) -> pa.Table:
    return pq.read_table(path)


class CachedEmbeddingDataset(Dataset):
    """Dataset of text + precomputed teacher embeddings stored in Parquet."""

    def __init__(self, parquet_path: Path):
        table = pq.read_table(parquet_path, columns=["text", "embedding"])
        self.texts = table.column("text").to_pylist()

        embedding_col = table.column("embedding")
        if pa.types.is_fixed_size_list(embedding_col.type):
            np_emb = embedding_col.combine_chunks().to_numpy(zero_copy_only=False)
            if np_emb.dtype == object:
                self.embeddings = np.stack(
                    [np.asarray(x, dtype=np.float32) for x in np_emb], axis=0
                )
            else:
                self.embeddings = np.asarray(np_emb, dtype=np.float32)
        else:
            self.embeddings = np.stack(
                [np.asarray(x, dtype=np.float32) for x in embedding_col.to_pylist()],
                axis=0,
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "text": self.texts[idx],
            "teacher_embedding": torch.from_numpy(self.embeddings[idx]),
        }


def corpus_row(
    *,
    row_id: str,
    text: str,
    lang: str,
    source: str,
    min_chars: int,
    normalize_whitespace: bool = True,
) -> dict[str, Any] | None:
    cleaned = normalize_text(text, normalize_whitespace=normalize_whitespace)
    if len(cleaned) < min_chars:
        return None
    return {
        "id": row_id,
        "text": cleaned,
        "lang": lang,
        "source": source,
    }
