#!/usr/bin/env python3
"""Distillation training on cached teacher embeddings (DDP, multi-loss)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_loss_weights,
    get_num_epochs,
    get_phase_config,
    get_previous_lang,
    get_resolved_paths,
    get_training_order,
    load_distill_config,
    parse_lang_list,
    require_path,
    resolve_embedding_path,
    resolve_output_root,
    resolve_retrieval_checkpoint_path,
    resolve_sts_checkpoint_path,
)
from harrier_distill.data import CachedEmbeddingDataset, ensure_dir
from harrier_distill.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from harrier_distill.losses import (
    combine_weighted_losses,
    cosine_embedding_loss,
    format_loss_postfix,
    pairwise_mse_loss,
    pointwise_mse_loss,
)
from harrier_distill.model import encode_training_batch_by_role, encode_with_prompt, get_model_dtype_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument("--lang", required=True, help="Language code (one per torchrun invocation)")
    parser.add_argument("--phase", choices=["sts", "retrieval"], default="sts")
    parser.add_argument("--embeddings", default=None, help="Override cached embeddings parquet")
    parser.add_argument("--init-model", default=None, help="Override student init checkpoint")
    parser.add_argument("--output", default=None, help="Override output checkpoint directory")
    parser.add_argument("--resume", default=None, help="Resume checkpoint path override")
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


def save_legacy_checkpoints(
    model: SentenceTransformer,
    *,
    rank: int,
    cfg: dict,
    lang: str,
    phase: str,
    checkpoint_dir: Path,
) -> None:
    if not is_main_process(rank):
        return
    output_root = resolve_output_root(cfg)
    order = get_training_order()
    save_checkpoint(model, checkpoint_dir, rank)

    if phase == "sts":
        if lang == "en":
            legacy_en = output_root / "checkpoint_en"
            if legacy_en.resolve() != checkpoint_dir.resolve():
                if legacy_en.exists():
                    shutil.rmtree(legacy_en)
                shutil.copytree(checkpoint_dir, legacy_en)
        if lang == order[-1]:
            legacy_final = output_root / "checkpoint_final"
            if legacy_final.resolve() != checkpoint_dir.resolve():
                if legacy_final.exists():
                    shutil.rmtree(legacy_final)
                shutil.copytree(checkpoint_dir, legacy_final)
    else:
        retrieval_root = output_root / "retrieval"
        if lang == "en":
            legacy_en = retrieval_root / "checkpoint_en"
            if legacy_en.resolve() != checkpoint_dir.resolve():
                if legacy_en.exists():
                    shutil.rmtree(legacy_en)
                shutil.copytree(checkpoint_dir, legacy_en)
        if lang == order[-1]:
            legacy_final = retrieval_root / "checkpoint_final"
            if legacy_final.resolve() != checkpoint_dir.resolve():
                if legacy_final.exists():
                    shutil.rmtree(legacy_final)
                shutil.copytree(checkpoint_dir, legacy_final)


def resolve_training_paths(args: argparse.Namespace, cfg: dict, paths: dict) -> tuple[Path, Path, Path]:
    output_root = require_path(paths, "output_dir")
    langs = parse_lang_list(args.lang)
    if len(langs) != 1:
        raise ValueError("Specify exactly one --lang per torchrun invocation")
    lang = langs[0]

    if args.phase == "retrieval":
        embeddings_path = (
            Path(args.embeddings) if args.embeddings else resolve_embedding_path(cfg, lang, phase="retrieval")
        )
        checkpoint_dir = (
            Path(args.output) if args.output else resolve_retrieval_checkpoint_path(cfg, lang)
        )
    else:
        embeddings_path = Path(args.embeddings) if args.embeddings else resolve_embedding_path(cfg, lang, phase="sts")
        checkpoint_dir = Path(args.output) if args.output else resolve_sts_checkpoint_path(cfg, lang)

    if args.init_model:
        init_path = Path(args.init_model)
    elif args.resume:
        init_path = Path(args.resume)
    else:
        prev = get_previous_lang(lang)
        if args.phase == "retrieval":
            if prev:
                prev_ckpt = resolve_retrieval_checkpoint_path(cfg, prev)
                init_path = prev_ckpt if prev_ckpt.exists() else resolve_sts_checkpoint_path(cfg, prev)
            else:
                legacy = output_root / "checkpoint_final"
                sts_last = resolve_sts_checkpoint_path(cfg, get_training_order()[-1])
                if legacy.exists():
                    init_path = legacy
                elif sts_last.exists():
                    init_path = sts_last
                else:
                    init_path = require_path(paths, "student_model")
        elif prev:
            prev_ckpt = resolve_sts_checkpoint_path(cfg, prev)
            init_path = prev_ckpt if prev_ckpt.exists() else require_path(paths, "student_model")
        else:
            init_path = require_path(paths, "student_model")

    return embeddings_path, checkpoint_dir, init_path


def encode_student_batch(
    raw_model: SentenceTransformer,
    texts: list[str],
    roles: list[str] | None,
    *,
    phase: str,
    query_prompt: str,
    doc_prompt: str | None,
    prompt_name: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    if phase == "retrieval" and roles:
        return encode_training_batch_by_role(
            raw_model,
            texts,
            roles,
            query_prompt=query_prompt,
            doc_prompt=doc_prompt,
            default_prompt=prompt_name,
            device=device,
            max_length=max_length,
        )
    return encode_with_prompt(
        raw_model,
        texts,
        prompt_name=prompt_name,
        device=device,
        max_length=max_length,
    )


def encode_triplet_batch(
    raw_model: SentenceTransformer,
    triplets: list[dict],
    *,
    query_prompt: str,
    doc_prompt: str | None,
    prompt_name: str,
    device: torch.device,
    max_length: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if not triplets:
        return [], []

    all_texts: list[str] = []
    all_roles: list[str] = []
    sizes: list[int] = []
    teacher_chunks: list[torch.Tensor] = []
    for triplet in triplets:
        all_texts.extend(triplet["texts"])
        all_roles.extend(triplet["roles"])
        sizes.append(len(triplet["texts"]))
        teacher_chunks.append(triplet["teacher_embedding"])

    student_all = encode_training_batch_by_role(
        raw_model,
        all_texts,
        all_roles,
        query_prompt=query_prompt,
        doc_prompt=doc_prompt,
        default_prompt=prompt_name,
        device=device,
        max_length=max_length,
    ).float()

    student_triplets: list[torch.Tensor] = []
    teacher_triplets: list[torch.Tensor] = []
    offset = 0
    for size, teacher in zip(sizes, teacher_chunks):
        student_triplets.append(student_all[offset : offset + size])
        teacher_triplets.append(teacher.to(device=device, dtype=torch.float32))
        offset += size
    return student_triplets, teacher_triplets


def main() -> None:
    args = parse_args()
    rank, _local_rank, world_size, device = init_distributed()

    cfg = load_distill_config(args.config)
    paths = get_resolved_paths(cfg)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    lang = parse_lang_list(args.lang)[0]

    embeddings_path, checkpoint_dir, init_path = resolve_training_paths(args, cfg, paths)
    loss_weights = get_loss_weights(cfg, args.phase)
    num_epochs = get_num_epochs(cfg, lang, args.phase)

    if args.phase == "retrieval":
        phase_cfg = get_phase_config(cfg, "retrieval")
        query_prompt = phase_cfg.get("query_prompt", "web_search_query")
        doc_prompt = phase_cfg.get("doc_prompt")
        prompt_name = query_prompt
    else:
        query_prompt = train_cfg.get("prompt_name", "sts_query")
        doc_prompt = None
        prompt_name = query_prompt

    batch_size = int(train_cfg.get("train_batch_size_per_gpu", 256))
    pairwise_triplets_per_batch = int(train_cfg.get("pairwise_triplets_per_batch", 64))
    learning_rate = float(train_cfg.get("learning_rate", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    max_length = int(data_cfg.get("max_length", 512))
    seed = int(train_cfg.get("seed", 42))

    torch.manual_seed(seed + rank)
    triplet_rng = np.random.default_rng(seed + rank)

    if is_main_process(rank):
        print(f"Phase: {args.phase}")
        print(f"Language: {lang}")
        print(f"Loss weights: {loss_weights}")
        print(f"Epochs: {num_epochs}")
        print(f"Loading cached embeddings: {embeddings_path}")

    role_column = "role" if args.phase == "retrieval" else None
    triplet_id_column = "triplet_id" if args.phase == "retrieval" else None
    dataset = CachedEmbeddingDataset(
        embeddings_path,
        role_column=role_column,
        triplet_id_column=triplet_id_column,
        show_progress=is_main_process(rank),
    )

    use_pairwise = loss_weights.get("pairwise_mse", 0.0) > 0
    if use_pairwise and not dataset.has_triplets:
        if is_main_process(rank):
            print(
                "WARNING: pairwise_mse weight > 0 but embedding parquet has no usable triplets. "
                "Re-run 02_generate_teacher_embeddings.py --phase retrieval after triplet_id support. "
                "Disabling pairwise_mse for this run."
            )
        loss_weights = dict(loss_weights)
        loss_weights["pairwise_mse"] = 0.0
        use_pairwise = False
        if sum(loss_weights.values()) <= 0:
            loss_weights["mse"] = 1.0

    use_pointwise = loss_weights.get("mse", 0.0) > 0 or loss_weights.get("cosine", 0.0) > 0
    steps_per_epoch = max(math.ceil(len(dataset) / (batch_size * world_size)), 1)

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 and use_pointwise else None
    loader = None
    if use_pointwise:
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
        if dataset.has_triplets:
            print(f"Loaded {len(dataset.triplet_id_list):,} retrieval triplets for pairwise loss")

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

    total_steps = max(steps_per_epoch * num_epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    running_totals = {"total": 0.0, "mse": 0.0, "cosine": 0.0, "pairwise_mse": 0.0}
    running_counts = {"mse": 0, "cosine": 0, "pairwise_mse": 0}

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if use_pointwise:
            step_iterable = loader
        else:
            step_iterable = range(steps_per_epoch)

        progress = tqdm(
            step_iterable,
            desc=f"train-{args.phase}-{lang}-epoch{epoch + 1}",
            disable=not is_main_process(rank),
        )
        for step_idx, batch in enumerate(progress):
            mse_loss = None
            cosine_loss = None
            pairwise_loss = None

            if use_pointwise:
                texts = batch["text"]
                roles = batch.get("role")
                teacher_emb = batch["teacher_embedding"].to(device, dtype=torch.float32)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    student_emb = encode_student_batch(
                        raw_model,
                        texts,
                        roles,
                        phase=args.phase,
                        query_prompt=query_prompt,
                        doc_prompt=doc_prompt,
                        prompt_name=prompt_name,
                        device=device,
                        max_length=max_length,
                    )

                if loss_weights.get("mse", 0.0) > 0:
                    mse_loss = pointwise_mse_loss(student_emb, teacher_emb)
                if loss_weights.get("cosine", 0.0) > 0:
                    cosine_loss = cosine_embedding_loss(student_emb, teacher_emb)

            if use_pairwise:
                triplet_seed = seed + rank + global_step + step_idx
                triplet_rng_step = np.random.default_rng(triplet_seed)
                triplets = dataset.sample_triplets(pairwise_triplets_per_batch, triplet_rng_step)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    student_triplets, teacher_triplets = encode_triplet_batch(
                        raw_model,
                        triplets,
                        query_prompt=query_prompt,
                        doc_prompt=doc_prompt,
                        prompt_name=prompt_name,
                        device=device,
                        max_length=max_length,
                    )
                if student_triplets:
                    pairwise_loss = pairwise_mse_loss(student_triplets, teacher_triplets)

            components = combine_weighted_losses(
                weights=loss_weights,
                mse=mse_loss,
                cosine=cosine_loss,
                pairwise_mse=pairwise_loss,
            )
            loss = components.total

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_totals["total"] += components.total.item()
            if components.mse is not None:
                running_totals["mse"] += components.mse.item()
                running_counts["mse"] += 1
            if components.cosine is not None:
                running_totals["cosine"] += components.cosine.item()
                running_counts["cosine"] += 1
            if components.pairwise_mse is not None:
                running_totals["pairwise_mse"] += components.pairwise_mse.item()
                running_counts["pairwise_mse"] += 1

            if is_main_process(rank):
                postfix = format_loss_postfix(components, loss_weights)
                postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                progress.set_postfix(postfix)

    barrier()
    save_legacy_checkpoints(
        raw_model,
        rank=rank,
        cfg=cfg,
        lang=lang,
        phase=args.phase,
        checkpoint_dir=checkpoint_dir,
    )

    if is_main_process(rank):
        metrics = {
            "phase": args.phase,
            "lang": lang,
            "epochs": num_epochs,
            "global_steps": global_step,
            "avg_loss": running_totals["total"] / max(global_step, 1),
            "avg_mse": running_totals["mse"] / max(running_counts["mse"], 1),
            "avg_cosine": running_totals["cosine"] / max(running_counts["cosine"], 1),
            "avg_pairwise_mse": running_totals["pairwise_mse"] / max(running_counts["pairwise_mse"], 1),
            "loss_weights": loss_weights,
            "checkpoint_dir": str(checkpoint_dir),
            "init_model": str(init_path),
            "embeddings": str(embeddings_path),
            "triplet_count": len(dataset.triplet_id_list),
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
