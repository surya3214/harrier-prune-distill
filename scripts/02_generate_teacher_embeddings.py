#!/usr/bin/env python3
"""Generate teacher embeddings with 4-GPU torchrun (offline GPU infra)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_phase_config,
    get_resolved_paths,
    load_distill_config,
    require_path,
    resolve_retrieval_corpus_paths,
    resolve_retrieval_embedding_paths,
)
from harrier_distill.data import (
    append_corpus_shard,
    clear_rank_done_markers,
    ensure_dir,
    merge_parquet_shards,
    wait_for_rank_done_markers,
    write_rank_done_marker,
)
from harrier_distill.distributed import cleanup_distributed, init_distributed, is_main_process, release_gpu_resources
from harrier_distill.model import encode_batch_by_role, get_model_dtype_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--phase", choices=["sts", "retrieval"], default="sts")
    parser.add_argument("--corpus", default=None, help="Override corpus parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    return parser.parse_args()


class CorpusDataset(Dataset):
    def __init__(self, texts: list[str], ids: list[str], langs: list[str], roles: list[str] | None = None):
        self.texts = texts
        self.ids = ids
        self.langs = langs
        self.roles = roles

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        item = {"id": self.ids[idx], "text": self.texts[idx], "lang": self.langs[idx]}
        if self.roles is not None:
            item["role"] = self.roles[idx]
        return item


def collate_rows(batch: list[dict]) -> dict:
    collated = {
        "id": [item["id"] for item in batch],
        "text": [item["text"] for item in batch],
        "lang": [item["lang"] for item in batch],
    }
    if "role" in batch[0]:
        collated["role"] = [item["role"] for item in batch]
    return collated


def write_shard(rows: list[dict], shard_path: Path) -> None:
    append_corpus_shard(rows, shard_path)


def finalize_shards(
    *,
    rank: int,
    world_size: int,
    shard_dir: Path,
    output_path: Path,
) -> None:
    """CPU-only merge on rank 0; avoid NCCL barriers after GPU work."""
    write_rank_done_marker(shard_dir, rank)
    release_gpu_resources()

    if not is_main_process(rank):
        cleanup_distributed()
        return

    wait_for_rank_done_markers(shard_dir, world_size)
    shard_paths = sorted(shard_dir.glob("rank*_part_*.parquet"))
    if not shard_paths:
        raise RuntimeError(f"No embedding shards written under {shard_dir}")

    merged_rows = merge_parquet_shards(shard_dir, output_path, pattern="rank*_part_*.parquet")
    print(f"Wrote {merged_rows:,} embeddings -> {output_path}")
    clear_rank_done_markers(shard_dir)
    cleanup_distributed()


def resolve_paths_for_phase(args: argparse.Namespace, cfg: dict, paths: dict) -> tuple[Path, Path]:
    if args.phase == "retrieval":
        corpus_paths = resolve_retrieval_corpus_paths(cfg)
        embedding_paths = resolve_retrieval_embedding_paths(cfg)
        corpus_path = Path(args.corpus) if args.corpus else corpus_paths[args.lang]
        output_path = Path(args.output) if args.output else embedding_paths[args.lang]
        return corpus_path, output_path

    corpus_key = f"{args.lang}_corpus"
    embeddings_key = f"{args.lang}_embeddings"
    corpus_path = Path(args.corpus) if args.corpus else require_path(paths, corpus_key)
    output_path = Path(args.output) if args.output else require_path(paths, embeddings_key)
    return corpus_path, output_path


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    corpus_path, output_path = resolve_paths_for_phase(args, cfg, paths)
    teacher_path = require_path(paths, "teacher_model")

    if args.phase == "retrieval":
        phase_cfg = get_phase_config(cfg, "retrieval")
        query_prompt = phase_cfg.get("query_prompt", "web_search_query")
        doc_prompt = phase_cfg.get("doc_prompt")
        prompt_name = query_prompt
    else:
        query_prompt = train_cfg.get("prompt_name", "sts_query")
        doc_prompt = None
        prompt_name = query_prompt

    batch_size = int(train_cfg.get("embed_batch_size_per_gpu", 192))
    max_length = int(data_cfg.get("max_length", 512))

    if is_main_process(rank):
        print(f"Phase: {args.phase}")
        print(f"Loading corpus: {corpus_path}")

    columns = ["id", "text", "lang"]
    schema_names = pq.read_schema(corpus_path).names
    if args.phase == "retrieval" and "role" in schema_names:
        columns.append("role")
    table = pq.read_table(corpus_path, columns=columns)
    texts = table.column("text").to_pylist()
    ids = table.column("id").to_pylist()
    langs = table.column("lang").to_pylist()
    roles = table.column("role").to_pylist() if "role" in columns else None

    dataset = CorpusDataset(texts, ids, langs, roles)
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rows,
    )

    if is_main_process(rank):
        print(f"Loading teacher model: {teacher_path}")
        if args.phase == "retrieval":
            print(f"Query prompt: {query_prompt}; doc prompt: {doc_prompt or '(none)'}")
        else:
            print(f"Prompt: {prompt_name}")

    model = SentenceTransformer(
        str(teacher_path),
        model_kwargs=get_model_dtype_kwargs(prefer_bf16=True),
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length

    shard_dir = output_path.parent / f".{args.lang}_{args.phase}_embedding_shards"
    ensure_dir(shard_dir)
    clear_rank_done_markers(shard_dir)
    for old in shard_dir.glob(f"rank{rank}_part_*.parquet"):
        old.unlink()

    rows: list[dict] = []
    shard_idx = 0
    iterator = tqdm(
        loader,
        desc=f"embed-{args.phase}-{args.lang}-rank{rank}",
        disable=not is_main_process(rank),
    )

    for batch in iterator:
        if args.phase == "retrieval" and batch.get("role"):
            embeddings = encode_batch_by_role(
                model,
                batch["text"],
                batch["role"],
                query_prompt=query_prompt,
                doc_prompt=doc_prompt,
                device=device,
                max_length=max_length,
                batch_size=batch_size,
            )
        else:
            embeddings = model.encode(
                batch["text"],
                prompt_name=prompt_name,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        for idx, (row_id, text, lang, emb) in enumerate(
            zip(batch["id"], batch["text"], batch["lang"], embeddings)
        ):
            row = {
                "id": row_id,
                "text": text,
                "lang": lang,
                "embedding": emb.tolist(),
            }
            if batch.get("role"):
                row["role"] = batch["role"][idx]
            rows.append(row)

        if len(rows) >= 50_000:
            shard_path = shard_dir / f"rank{rank}_part_{shard_idx:05d}.parquet"
            write_shard(rows, shard_path)
            rows = []
            shard_idx += 1

    if rows:
        shard_path = shard_dir / f"rank{rank}_part_{shard_idx:05d}.parquet"
        write_shard(rows, shard_path)

    del loader, dataset, table, texts, ids, langs, model
    release_gpu_resources()
    finalize_shards(
        rank=rank,
        world_size=world_size,
        shard_dir=shard_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
