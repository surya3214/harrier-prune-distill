from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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

    ensure_dir(output_path.parent)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for path in shard_paths:
            table = pq.read_table(path)
            total_rows += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def write_rank_done_marker(shard_dir: Path, rank: int) -> Path:
    ensure_dir(shard_dir)
    marker = shard_dir / f"rank{rank}.done"
    marker.write_text("done\n", encoding="utf-8")
    return marker


def wait_for_rank_done_markers(shard_dir: Path, world_size: int, *, timeout_sec: int = 3600) -> None:
    import time

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        done_count = len(list(shard_dir.glob("rank*.done")))
        if done_count >= world_size:
            return
        time.sleep(1)
    raise TimeoutError(
        f"Timed out waiting for embedding shards in {shard_dir}. "
        f"Found {len(list(shard_dir.glob('rank*.done')))}/{world_size} done markers."
    )


def clear_rank_done_markers(shard_dir: Path) -> None:
    for marker in shard_dir.glob("rank*.done"):
        marker.unlink(missing_ok=True)


def load_corpus_table(path: Path) -> pa.Table:
    return pq.read_table(path)


class CachedEmbeddingDataset:
    """Dataset of text + precomputed teacher embeddings stored in Parquet."""

    def __init__(self, parquet_path: Path, *, role_column: str | None = "role"):
        columns = ["text", "embedding"]
        table = pq.read_table(parquet_path)
        available = set(table.column_names)
        if role_column and role_column in available:
            columns.append(role_column)
        table = table.select(columns)
        self.texts = table.column("text").to_pylist()
        self.roles = table.column(role_column).to_pylist() if role_column and role_column in columns else None

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
        import torch

        item: dict[str, Any] = {
            "text": self.texts[idx],
            "teacher_embedding": torch.from_numpy(self.embeddings[idx]),
        }
        if self.roles is not None:
            item["role"] = self.roles[idx]
        return item


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
