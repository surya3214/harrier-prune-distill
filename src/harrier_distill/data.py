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


def _stack_embedding_batch(embedding_values: list[Any]) -> np.ndarray:
    """Convert one parquet batch of embedding values to a float32 ndarray."""
    if not embedding_values:
        return np.zeros((0, 0), dtype=np.float32)

    first = embedding_values[0]
    if isinstance(first, np.ndarray) and first.ndim == 1:
        return np.stack([np.asarray(x, dtype=np.float32) for x in embedding_values], axis=0)

    if isinstance(first, (list, tuple)):
        return np.stack([np.asarray(x, dtype=np.float32) for x in embedding_values], axis=0)

    arr = np.asarray(embedding_values, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


class CachedEmbeddingDataset:
    """Dataset of text + precomputed teacher embeddings stored in Parquet."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        role_column: str | None = "role",
        batch_size: int = 50_000,
        show_progress: bool = True,
    ):
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        columns = ["text", "embedding"]
        load_role = role_column is not None and role_column in available
        if load_role:
            columns.append(role_column)

        total_rows = pf.metadata.num_rows
        texts: list[str] = []
        roles: list[str] = []
        emb_chunks: list[np.ndarray] = []

        batch_iter = pf.iter_batches(batch_size=batch_size, columns=columns)
        progress = None
        if show_progress:
            from tqdm import tqdm

            progress = tqdm(total=total_rows, desc="Loading cached embeddings", unit="rows")

        try:
            for batch in batch_iter:
                chunk = batch.to_pydict()
                texts.extend(chunk["text"])
                if load_role:
                    roles.extend(chunk[role_column])
                emb_chunks.append(_stack_embedding_batch(chunk["embedding"]))
                if progress is not None:
                    progress.update(batch.num_rows)
        finally:
            if progress is not None:
                progress.close()

        self.texts = texts
        self.roles = roles if load_role else None
        self.embeddings = np.vstack(emb_chunks) if emb_chunks else np.zeros((0, 0), dtype=np.float32)

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
