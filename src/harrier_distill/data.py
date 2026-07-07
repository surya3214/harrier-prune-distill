from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harrier_distill.text import normalize_text

MULTILINGUAL_NLI_HF_PATH = "MoritzLaurer/multilingual-NLI-26lang-2mil7"
MULTILINGUAL_NLI_SUBSOURCES = ("mnli", "fever", "anli", "wanli", "ling")

LEGACY_DATASET_HINTS: dict[str, str] = {
    "hpprc/jsick": "Use hf_path: mteb/JSICK (Parquet-native MTEB mirror).",
    "mc4": "Use hf_path: allenai/c4 with the same language config (e.g. config: ja).",
}


def load_hf_source_dataset(source_cfg: dict[str, Any], *, split: str):
    """Load a Hugging Face dataset for a ``datasets.yaml`` source entry."""
    from datasets import load_dataset

    hf_path = source_cfg["hf_path"]
    config = source_cfg.get("config")
    streaming = source_cfg.get("streaming", False)
    trust_remote_code = bool(source_cfg.get("trust_remote_code", False))

    kwargs: dict[str, Any] = {"path": hf_path, "split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        return load_dataset(**kwargs)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        legacy_script = (
            "Dataset scripts are no longer supported" in message
            or "contains custom code" in message
        )
        if legacy_script:
            hint = LEGACY_DATASET_HINTS.get(hf_path, "Pick a Parquet-native dataset on Hugging Face Hub.")
            raise RuntimeError(
                f"{message} Dataset '{hf_path}' uses a legacy loading script. {hint}"
            ) from exc
        if "Bad split" in message or 'Unknown split "train"' in message:
            raise ValueError(
                f"Dataset '{hf_path}' does not have split '{split}'. "
                f"Set source.splits explicitly or use filter_lang with multilingual-NLI. "
                f"Original error: {message}"
            ) from exc
        raise


def resolve_hf_source_splits(source_cfg: dict[str, Any], lang: str) -> list[str]:
    """Resolve HF split name(s) for a dataset source config.

    MoritzLaurer/multilingual-NLI-26lang-2mil7 has no ``train`` split; it exposes
    per-language subsets such as ``de_mnli``, ``ko_fever``, etc.
    """
    explicit = source_cfg.get("splits")
    if explicit:
        return [str(split) for split in explicit]

    hf_path = source_cfg.get("hf_path", "")
    filter_lang = source_cfg.get("filter_lang", lang)
    if hf_path == MULTILINGUAL_NLI_HF_PATH or source_cfg.get("split_resolver") == "multilingual_nli":
        subsources = source_cfg.get("nli_subsources", MULTILINGUAL_NLI_SUBSOURCES)
        return [f"{filter_lang}_{sub}" for sub in subsources]

    return [str(source_cfg.get("split", "train"))]


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


def _build_triplet_groups(
    triplet_ids: list[str],
    roles: list[str],
) -> tuple[dict[str, dict[str, list[int]]], list[str]]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"query": [], "doc": []})
    for idx, (triplet_id, role) in enumerate(zip(triplet_ids, roles)):
        if role == "query":
            groups[triplet_id]["query"].append(idx)
        else:
            groups[triplet_id]["doc"].append(idx)

    valid_triplets: dict[str, dict[str, list[int]]] = {}
    triplet_id_list: list[str] = []
    for triplet_id, group in groups.items():
        if len(group["query"]) == 1 and len(group["doc"]) >= 1:
            valid_triplets[triplet_id] = {
                "query": group["query"],
                "doc": group["doc"],
            }
            triplet_id_list.append(triplet_id)
    return valid_triplets, triplet_id_list


class CachedEmbeddingDataset:
    """Dataset of text + precomputed teacher embeddings stored in Parquet."""

    def __init__(
        self,
        parquet_path: Path,
        *,
        role_column: str | None = "role",
        triplet_id_column: str | None = "triplet_id",
        batch_size: int = 50_000,
        show_progress: bool = True,
    ):
        pf = pq.ParquetFile(parquet_path)
        available = set(pf.schema_arrow.names)
        columns = ["text", "embedding"]
        load_role = role_column is not None and role_column in available
        load_triplet_id = triplet_id_column is not None and triplet_id_column in available
        if load_role:
            columns.append(role_column)
        if load_triplet_id:
            columns.append(triplet_id_column)

        total_rows = pf.metadata.num_rows
        texts: list[str] = []
        roles: list[str] = []
        triplet_ids: list[str] = []
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
                if load_triplet_id:
                    triplet_ids.extend(chunk[triplet_id_column])
                emb_chunks.append(_stack_embedding_batch(chunk["embedding"]))
                if progress is not None:
                    progress.update(batch.num_rows)
        finally:
            if progress is not None:
                progress.close()

        self.texts = texts
        self.roles = roles if load_role else None
        self.triplet_ids = triplet_ids if load_triplet_id else None
        self.embeddings = np.vstack(emb_chunks) if emb_chunks else np.zeros((0, 0), dtype=np.float32)
        self.triplets: dict[str, dict[str, list[int]]] = {}
        self.triplet_id_list: list[str] = []
        if self.triplet_ids is not None and self.roles is not None:
            self.triplets, self.triplet_id_list = _build_triplet_groups(self.triplet_ids, self.roles)

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

    @property
    def has_triplets(self) -> bool:
        return bool(self.triplet_id_list)

    def sample_triplets(self, count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Sample retrieval triplets with texts, roles, and teacher embeddings."""
        import torch

        if not self.triplet_id_list:
            return []

        size = min(count, len(self.triplet_id_list))
        chosen = rng.choice(len(self.triplet_id_list), size=size, replace=size > len(self.triplet_id_list))
        samples: list[dict[str, Any]] = []
        for choice in chosen:
            triplet_id = self.triplet_id_list[int(choice)]
            group = self.triplets[triplet_id]
            indices = [group["query"][0], *group["doc"]]
            samples.append(
                {
                    "triplet_id": triplet_id,
                    "texts": [self.texts[idx] for idx in indices],
                    "roles": [self.roles[idx] for idx in indices] if self.roles is not None else None,
                    "teacher_embedding": torch.from_numpy(self.embeddings[indices].astype(np.float32)),
                }
            )
        return samples


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
