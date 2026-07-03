#!/usr/bin/env python3
"""MSE distillation training on cached teacher embeddings (DDP, 4-GPU)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_phase_config,
    get_resolved_paths,
    load_distill_config,
    require_path,
    resolve_retrieval_checkpoint_paths,
    resolve_retrieval_embedding_paths,
)
from harrier_distill.data import CachedEmbeddingDataset, ensure_dir
from harrier_distill.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from harrier_distill.model import encode_training_batch_by_role, encode_with_prompt, get_model_dtype_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--phase", choices=["sts", "retrieval"], default="sts")
    parser.add_argument("--embeddings", default=None, help="Override cached embeddings parquet")
    parser.add_argument("--init-model", default=None, help="Override student init checkpoint")
    parser.add_argument("--output", default=None, help="Override output checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume checkpoint (e.g. checkpoint_en for KO step)")
    return parser.parse_args()


def collate_batch(batch: list[dict]) -> dict:
    collated = {
        "text": [item["text"] for item in batch],
        "teacher_embedding": torch.stack([item["teacher_embedding"] for item in batch], dim=0),
    }
    if "role" in batch[0]:
        collated["role"] = [item["role"] for item in batch]
    return collated


def save_checkpoint(model: SentenceTransformer, output_dir: Path, rank: int) -> None:
    if not is_main_process(rank):
        return
    ensure_dir(output_dir)
    model.save(str(output_dir))


def resolve_training_paths(args: argparse.Namespace, cfg: dict, paths: dict) -> tuple[Path, Path, Path]:
    output_root = require_path(paths, "output_dir")

    if args.phase == "retrieval":
        embedding_paths = resolve_retrieval_embedding_paths(cfg)
        checkpoint_paths = resolve_retrieval_checkpoint_paths(cfg)
        embeddings_path = Path(args.embeddings) if args.embeddings else embedding_paths[args.lang]
        if args.output:
            checkpoint_dir = Path(args.output)
        else:
            checkpoint_dir = checkpoint_paths["en"] if args.lang == "en" else checkpoint_paths["final"]

        if args.init_model:
            init_path = Path(args.init_model)
        elif args.lang == "ko":
            resume_path = Path(args.resume) if args.resume else checkpoint_paths["en"]
            init_path = resume_path if resume_path.exists() else (output_root / "checkpoint_final")
            if not init_path.exists():
                init_path = require_path(paths, "student_model")
        else:
            sts_final = output_root / "checkpoint_final"
            init_path = sts_final if sts_final.exists() else require_path(paths, "student_model")

        return embeddings_path, checkpoint_dir, init_path

    embeddings_key = f"{args.lang}_embeddings"
    embeddings_path = Path(args.embeddings) if args.embeddings else require_path(paths, embeddings_key)
    if args.output:
        checkpoint_dir = Path(args.output)
    else:
        checkpoint_name = "checkpoint_en" if args.lang == "en" else "checkpoint_final"
        checkpoint_dir = output_root / checkpoint_name

    if args.init_model:
        init_path = Path(args.init_model)
    elif args.lang == "ko":
        resume_path = Path(args.resume) if args.resume else (output_root / "checkpoint_en")
        init_path = resume_path if resume_path.exists() else require_path(paths, "student_model")
    else:
        init_path = require_path(paths, "student_model")

    return embeddings_path, checkpoint_dir, init_path


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    embeddings_path, checkpoint_dir, init_path = resolve_training_paths(args, cfg, paths)

    if args.phase == "retrieval":
        phase_cfg = get_phase_config(cfg, "retrieval")
        query_prompt = phase_cfg.get("query_prompt", "web_search_query")
        doc_prompt = phase_cfg.get("doc_prompt")
        prompt_name = query_prompt
        epoch_key = "num_epochs_retrieval_en" if args.lang == "en" else "num_epochs_retrieval_ko"
        num_epochs = int(phase_cfg.get(epoch_key, train_cfg.get("num_epochs_en" if args.lang == "en" else "num_epochs_ko", 1)))
    else:
        query_prompt = train_cfg.get("prompt_name", "sts_query")
        doc_prompt = None
        prompt_name = query_prompt
        num_epochs = int(train_cfg.get("num_epochs_en" if args.lang == "en" else "num_epochs_ko", 1))

    batch_size = int(train_cfg.get("train_batch_size_per_gpu", 256))
    learning_rate = float(train_cfg.get("learning_rate", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    max_length = int(data_cfg.get("max_length", 512))
    seed = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed + rank)

    if is_main_process(rank):
        print(f"Phase: {args.phase}")
        print(f"Loading cached embeddings: {embeddings_path}")

    role_column = "role" if args.phase == "retrieval" else None
    dataset = CachedEmbeddingDataset(embeddings_path, role_column=role_column)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    if is_main_process(rank):
        print(f"Loading student from: {init_path}")

    model = SentenceTransformer(
        str(init_path),
        model_kwargs=get_model_dtype_kwargs(),
        trust_remote_code=True,
    )
    model = model.to(device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
    model.train()

    if world_size > 1 and device.type == "cuda":
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(dataset) / (batch_size * world_size))
    total_steps = max(steps_per_epoch * num_epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    running_loss = 0.0

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        progress = tqdm(
            loader,
            desc=f"train-{args.phase}-{args.lang}-epoch{epoch + 1}",
            disable=not is_main_process(rank),
        )
        for batch in progress:
            texts = batch["text"]
            roles = batch.get("role")
            teacher_emb = batch["teacher_embedding"].to(device, dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if args.phase == "retrieval" and roles:
                    student_emb = encode_training_batch_by_role(
                        raw_model,
                        texts,
                        roles,
                        query_prompt=query_prompt,
                        doc_prompt=doc_prompt,
                        default_prompt=prompt_name,
                        device=device,
                        max_length=max_length,
                    )
                else:
                    student_emb = encode_with_prompt(
                        raw_model,
                        texts,
                        prompt_name=prompt_name,
                        device=device,
                        max_length=max_length,
                    )
            loss = F.mse_loss(student_emb.float(), teacher_emb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            if is_main_process(rank):
                progress.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    barrier()
    save_checkpoint(raw_model, checkpoint_dir, rank)

    if is_main_process(rank):
        metrics = {
            "phase": args.phase,
            "lang": args.lang,
            "epochs": num_epochs,
            "global_steps": global_step,
            "avg_loss": running_loss / max(global_step, 1),
            "checkpoint_dir": str(checkpoint_dir),
            "init_model": str(init_path),
            "embeddings": str(embeddings_path),
        }
        metrics_path = checkpoint_dir / "train_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved checkpoint -> {checkpoint_dir}")
        print(f"Metrics -> {metrics_path}")

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
