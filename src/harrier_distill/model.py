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


def _features_to_device(
    features: dict,
    device: torch.device,
    max_length: int,
) -> dict:
    """Move token tensors to device; leave metadata fields (e.g. str) untouched."""
    try:
        from sentence_transformers.util import batch_to_device

        features = batch_to_device(features, device)
    except Exception:
        features = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }

    for key, value in list(features.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] > max_length:
            features[key] = value[:, :max_length]
    return features


def encode_with_prompt(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prompt_name: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Differentiable forward pass with prompt prefix."""
    try:
        features = model.tokenize(texts, prompt_name=prompt_name)
    except TypeError:
        prompt = get_prompt(model, prompt_name)
        prompted = [f"{prompt}{text}" for text in texts]
        features = model.tokenize(prompted)

    features = _features_to_device(features, device, max_length)
    outputs = model(features)
    embeddings = outputs["sentence_embedding"]
    return F.normalize(embeddings, p=2, dim=1)
