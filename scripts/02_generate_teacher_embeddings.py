#!/usr/bin/env python3
"""Generate teacher embeddings with 4-GPU torchrun (offline GPU infra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path
from harrier_distill.data import ensure_dir, merge_parquet_shards
from harrier_distill.distributed import barrier, cleanup_distributed, init_distributed, is_main_process


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--corpus", default=None, help="Override corpus parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    return parser.parse_args()


class CorpusDataset(Dataset):
    def __init__(self, texts: list[str], ids: list[str], langs: list[str]):
        self.texts = texts
        self.ids = ids
        self.langs = langs

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        return {"id": self.ids[idx], "text": self.texts[idx], "lang": self.langs[idx]}


def collate_rows(batch: list[dict]) -> dict:
    return {
        "id": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "lang": [item["lang"] for item in batch],
    }


@torch.inference_mode()
def encode_batch(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prompt_name: str,
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        prompt_name=prompt_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def write_shard(rows: list[dict], shard_path: Path) -> None:
    pq.write_table(pa.Table.from_pylist(rows), shard_path, compression="zstd")


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    corpus_key = f"{args.lang}_corpus"
    embeddings_key = f"{args.lang}_embeddings"
    corpus_path = Path(args.corpus) if args.corpus else require_path(paths, corpus_key)
    output_path = Path(args.output) if args.output else require_path(paths, embeddings_key)
    teacher_path = require_path(paths, "teacher_model")

    prompt_name = train_cfg.get("prompt_name", "sts_query")
    batch_size = int(train_cfg.get("embed_batch_size_per_gpu", 192))
    max_length = int(data_cfg.get("max_length", 512))

    if is_main_process(rank):
        print(f"Loading corpus: {corpus_path}")
    table = pq.read_table(corpus_path, columns=["id", "text", "lang"])
    texts = table.column("text").to_pylist()
    ids = table.column("id").to_pylist()
    langs = table.column("lang").to_pylist()

    dataset = CorpusDataset(texts, ids, langs)
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_rows,
    )

    if is_main_process(rank):
        print(f"Loading teacher model: {teacher_path}")
    model = SentenceTransformer(
        str(teacher_path),
        model_kwargs={"dtype": torch.bfloat16},
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length

    shard_dir = output_path.parent / f".{args.lang}_embedding_shards"
    ensure_dir(shard_dir)
    for old in shard_dir.glob(f"rank{rank}_part_*.parquet"):
        old.unlink()
    barrier()

    rows: list[dict] = []
    shard_idx = 0
    iterator = tqdm(loader, desc=f"embed-{args.lang}-rank{rank}", disable=not is_main_process(rank))

    for batch in iterator:
        embeddings = encode_batch(
            model,
            batch["text"],
            prompt_name=prompt_name,
            batch_size=batch_size,
        )
        for row_id, text, lang, emb in zip(batch["id"], batch["text"], batch["lang"], embeddings):
            rows.append(
                {
                    "id": row_id,
                    "text": text,
                    "lang": lang,
                    "embedding": emb.tolist(),
                }
            )

        if len(rows) >= 50_000:
            shard_path = shard_dir / f"rank{rank}_part_{shard_idx:05d}.parquet"
            write_shard(rows, shard_path)
            rows = []
            shard_idx += 1

    if rows:
        shard_path = shard_dir / f"rank{rank}_part_{shard_idx:05d}.parquet"
        write_shard(rows, shard_path)

    barrier()

    if is_main_process(rank):
        all_shards = sorted(shard_dir.glob("rank*_part_*.parquet"))
        if not all_shards:
            raise RuntimeError(f"No embedding shards written under {shard_dir}")
        merged_rows = merge_parquet_shards(shard_dir, output_path, pattern="rank*_part_*.parquet")
        print(f"Wrote {merged_rows:,} embeddings -> {output_path}")

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
