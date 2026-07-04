from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def get_model_dtype_kwargs(*, prefer_bf16: bool | None = None) -> dict[str, torch.dtype]:
    """Return transformers-safe dtype kwargs for SentenceTransformer model_kwargs."""
    import transformers

    use_bf16 = torch.cuda.is_available() if prefer_bf16 is None else prefer_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    major = int(transformers.__version__.split(".", maxsplit=1)[0])
    # transformers 4.x only accepts torch_dtype in from_pretrained; passing dtype
    # leaks into Gemma3TextModel.__init__ and raises TypeError.
    if major >= 5:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_sentence_transformer(
    model_path: str | Path,
    *,
    device: torch.device | str | None = None,
    trust_remote_code: bool = True,
    prefer_bf16: bool | None = None,
) -> SentenceTransformer:
    model = SentenceTransformer(
        str(model_path),
        model_kwargs=get_model_dtype_kwargs(prefer_bf16=prefer_bf16),
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


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
    prompt_name: str | None = None,
) -> torch.Tensor:
    """Differentiable encode with optional prompt (None = no prompt on documents)."""
    if prompt_name:
        return encode_with_prompt(
            model,
            texts,
            prompt_name=prompt_name,
            device=device,
            max_length=max_length,
        )

    features = model.tokenize(texts)
    features = _features_to_device(features, device, max_length)
    outputs = model(features)
    embeddings = outputs["sentence_embedding"]
    return F.normalize(embeddings, p=2, dim=1)


def encode_batch_by_role(
    model: SentenceTransformer,
    texts: list[str],
    roles: list[str],
    *,
    query_prompt: str,
    doc_prompt: str | None,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> "np.ndarray":
    """Inference-mode batch encode with role-specific prompts."""
    import numpy as np

    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_roles = roles[start : start + batch_size]
        query_indices = [idx for idx, role in enumerate(batch_roles) if role == "query"]
        doc_indices = [idx for idx, role in enumerate(batch_roles) if role != "query"]

        batch_emb = np.zeros((len(batch_texts), model.get_sentence_embedding_dimension()), dtype=np.float32)
        if query_indices:
            query_texts = [batch_texts[idx] for idx in query_indices]
            query_emb = model.encode(
                query_texts,
                prompt_name=query_prompt,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            for local_idx, emb in zip(query_indices, np.asarray(query_emb)):
                batch_emb[local_idx] = emb

        if doc_indices:
            doc_texts = [batch_texts[idx] for idx in doc_indices]
            encode_kwargs = {
                "batch_size": batch_size,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            if doc_prompt:
                encode_kwargs["prompt_name"] = doc_prompt
            doc_emb = model.encode(doc_texts, **encode_kwargs)
            for local_idx, emb in zip(doc_indices, np.asarray(doc_emb)):
                batch_emb[local_idx] = emb

        embeddings.append(batch_emb)

    return np.vstack(embeddings) if embeddings else np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)


def encode_training_batch_by_role(
    model: SentenceTransformer,
    texts: list[str],
    roles: list[str] | None,
    *,
    query_prompt: str,
    doc_prompt: str | None,
    default_prompt: str,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    """Differentiable encode for mixed query/doc batches during distillation."""
    if not roles:
        return encode_with_prompt(
            model,
            texts,
            prompt_name=default_prompt,
            device=device,
            max_length=max_length,
        )

    if len(texts) != len(roles):
        raise ValueError(f"texts/roles length mismatch: {len(texts)} vs {len(roles)}")

    batch_size = len(texts)
    query_indices = [idx for idx, role in enumerate(roles) if role == "query"]
    doc_indices = [idx for idx, role in enumerate(roles) if role != "query"]

    emb_dim = model.get_sentence_embedding_dimension()
    embeddings = torch.empty(batch_size, emb_dim, device=device)

    if query_indices:
        query_texts = [texts[idx] for idx in query_indices]
        query_emb = encode_with_prompt(
            model,
            query_texts,
            prompt_name=query_prompt,
            device=device,
            max_length=max_length,
        )
        index = torch.tensor(query_indices, device=query_emb.device, dtype=torch.long)
        embeddings.index_copy_(0, index, query_emb)

    if doc_indices:
        doc_texts = [texts[idx] for idx in doc_indices]
        if doc_prompt:
            doc_emb = encode_with_prompt(
                model,
                doc_texts,
                prompt_name=doc_prompt,
                device=device,
                max_length=max_length,
            )
        else:
            doc_emb = encode_texts(
                model,
                doc_texts,
                device=device,
                max_length=max_length,
                prompt_name=None,
            )
        index = torch.tensor(doc_indices, device=doc_emb.device, dtype=torch.long)
        embeddings.index_copy_(0, index, doc_emb)

    return embeddings
