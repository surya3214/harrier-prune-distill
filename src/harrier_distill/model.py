from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

ATTN_IMPLEMENTATION_ALIASES = {
    "sdpa": "sdpa",
    "flash_attention_2": "flash_attention_2",
    "none": "eager",
    "eager": "eager",
}


def resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    """Map config attn_implementation to HuggingFace attn_implementation.

    Accepted config values: sdpa, flash_attention_2, none (maps to eager).
    None / empty leaves HF defaults unchanged.
    """
    if attn_implementation is None:
        return None
    key = str(attn_implementation).strip().lower()
    if not key:
        return None
    if key not in ATTN_IMPLEMENTATION_ALIASES:
        allowed = ", ".join(sorted(ATTN_IMPLEMENTATION_ALIASES))
        raise ValueError(f"Unsupported attn_implementation={attn_implementation!r}. Allowed: {allowed}")
    return ATTN_IMPLEMENTATION_ALIASES[key]


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


def get_model_kwargs(
    *,
    prefer_bf16: bool | None = None,
    attn_implementation: str | None = None,
) -> dict[str, Any]:
    """Build SentenceTransformer model_kwargs (dtype + optional attention backend)."""
    kwargs: dict[str, Any] = dict(get_model_dtype_kwargs(prefer_bf16=prefer_bf16))
    resolved = resolve_attn_implementation(attn_implementation)
    if resolved is not None:
        kwargs["attn_implementation"] = resolved
    return kwargs


def get_transformer_auto_model(model: "SentenceTransformer") -> torch.nn.Module:
    """Return the HuggingFace backbone under a SentenceTransformer wrapper."""
    first = model[0]
    auto_model = getattr(first, "auto_model", None)
    if auto_model is None:
        raise AttributeError(
            "SentenceTransformer module[0] has no auto_model; cannot enable gradient checkpointing"
        )
    return auto_model


def enable_gradient_checkpointing(model: "SentenceTransformer") -> None:
    """Enable non-reentrant gradient checkpointing on the HF backbone."""
    auto_model = get_transformer_auto_model(model)
    if hasattr(auto_model, "gradient_checkpointing_enable"):
        auto_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        raise AttributeError("Backbone does not support gradient_checkpointing_enable()")
    config = getattr(auto_model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


def maybe_enable_tf32(*, enabled: bool = True) -> bool:
    """Enable TF32 matmul on Ampere+ GPUs (capability >= 8.0). No-op on V100/CPU.

    Returns True if TF32 was enabled.
    """
    if not enabled or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        return False
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return True


def build_adamw(
    parameters,
    *,
    lr: float,
    weight_decay: float,
    fused: bool = True,
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Construct AdamW, preferring fused=True on CUDA when requested."""
    use_fused = bool(fused) and device_type == "cuda"
    if use_fused:
        try:
            return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, fused=True)
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def load_sentence_transformer(
    model_path: str | Path,
    *,
    device: torch.device | str | None = None,
    trust_remote_code: bool = True,
    prefer_bf16: bool | None = None,
    attn_implementation: str | None = None,
) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(model_path),
        model_kwargs=get_model_kwargs(
            prefer_bf16=prefer_bf16,
            attn_implementation=attn_implementation,
        ),
        trust_remote_code=trust_remote_code,
    )
    if device is not None:
        model = model.to(device)
    return model


def get_prompt(model: "SentenceTransformer", prompt_name: str) -> str:
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
    model: "SentenceTransformer",
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
    model: "SentenceTransformer",
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
    model: "SentenceTransformer",
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
    model: "SentenceTransformer",
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
