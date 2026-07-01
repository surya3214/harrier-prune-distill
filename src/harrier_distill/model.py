from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def load_sentence_transformer(
    model_path: str | Path,
    *,
    device: torch.device | str | None = None,
    trust_remote_code: bool = True,
) -> SentenceTransformer:
    model_kwargs = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    model = SentenceTransformer(
        str(model_path),
        model_kwargs=model_kwargs,
        trust_remote_code=trust_remote_code,
    )
    if device is not None:
        model = model.to(device)
    return model


def get_prompt(model: SentenceTransformer, prompt_name: str) -> str:
    prompts = getattr(model, "prompts", None) or {}
    if prompt_name not in prompts:
        available = ", ".join(sorted(prompts.keys())) if prompts else "(none)"
        raise KeyError(f"Prompt '{prompt_name}' not found. Available prompts: {available}")
    return prompts[prompt_name]


def encode_with_prompt(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Differentiable forward pass with prompt prefix."""
    prompt = get_prompt(model, prompt_name)
    prompted = [f"{prompt}{text}" for text in texts]
    features = model.tokenize(prompted)
    features = {key: value.to(device) for key, value in features.items()}
    if "attention_mask" in features:
        # Respect max_length from config during training.
        for key, value in features.items():
            if value.shape[1] > max_length:
                features[key] = value[:, :max_length]
    outputs = model(features)
    embeddings = outputs["sentence_embedding"]
    return F.normalize(embeddings, p=2, dim=1)
