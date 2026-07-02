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

from harrier_distill.config import get_resolved_paths, load_distill_config, require_path
from harrier_distill.data import CachedEmbeddingDataset, ensure_dir
from harrier_distill.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from harrier_distill.model import encode_with_prompt, get_model_dtype_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--lang", choices=["en", "ko"], required=True)
    parser.add_argument("--embeddings", default=None, help="Override cached embeddings parquet")
    parser.add_argument("--init-model", default=None, help="Override student init checkpoint")
    parser.add_argument("--output", default=None, help="Override output checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume checkpoint (e.g. checkpoint_en for KO step)")
    return parser.parse_args()


def collate_batch(batch: list[dict]) -> dict:
    return {
        "text": [item["text"] for item in batch],
        "teacher_embedding": torch.stack([item["teacher_embedding"] for item in batch], dim=0),
    }


def save_checkpoint(model: SentenceTransformer, output_dir: Path, rank: int) -> None:
    if not is_main_process(rank):
        return
    ensure_dir(output_dir)
    model.save(str(output_dir))


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    embeddings_key = f"{args.lang}_embeddings"
    embeddings_path = Path(args.embeddings) if args.embeddings else require_path(paths, embeddings_key)
    output_root = require_path(paths, "output_dir")
    student_init = Path(args.init_model) if args.init_model else require_path(paths, "student_model")

    if args.output:
        checkpoint_dir = Path(args.output)
    else:
        checkpoint_name = "checkpoint_en" if args.lang == "en" else "checkpoint_final"
        checkpoint_dir = output_root / checkpoint_name

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.lang == "ko":
        en_ckpt = output_root / "checkpoint_en"
        if en_ckpt.exists():
            resume_path = en_ckpt

    prompt_name = train_cfg.get("prompt_name", "sts_query")
    batch_size = int(train_cfg.get("train_batch_size_per_gpu", 256))
    learning_rate = float(train_cfg.get("learning_rate", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    num_epochs = int(
        train_cfg.get("num_epochs_en" if args.lang == "en" else "num_epochs_ko", 1)
    )
    max_length = int(data_cfg.get("max_length", 512))
    seed = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed + rank)

    if is_main_process(rank):
        print(f"Loading cached embeddings: {embeddings_path}")

    dataset = CachedEmbeddingDataset(embeddings_path)
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

    init_path = resume_path if resume_path and resume_path.exists() else student_init
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
            desc=f"train-{args.lang}-epoch{epoch + 1}",
            disable=not is_main_process(rank),
        )
        for batch in progress:
            texts = batch["text"]
            teacher_emb = batch["teacher_embedding"].to(device, dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
