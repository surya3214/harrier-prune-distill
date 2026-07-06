from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | None, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    root = base or PROJECT_ROOT
    return (root / path).resolve()


def load_distill_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "distill.yaml"
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def load_datasets_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "datasets.yaml"
    return load_yaml(config_path)


def load_sts_datasets_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "sts_datasets.yaml"
    return load_yaml(config_path)


def load_retrieval_datasets_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "retrieval_datasets.yaml"
    return load_yaml(config_path)


def load_retrieval_eval_datasets_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "retrieval_eval_datasets.yaml"
    return load_yaml(config_path)


def get_phase_config(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    phases = cfg.get("phases", {})
    if phase not in phases:
        raise KeyError(f"Unknown phase '{phase}'. Available: {', '.join(sorted(phases)) or '(none)'}")
    return phases[phase]


LOSS_NAMES = ("mse", "cosine", "pairwise_mse")


def get_loss_weights(cfg: dict[str, Any], phase: str) -> dict[str, float]:
    """Merge training.losses with optional phases.<phase>.losses overrides."""
    train_cfg = cfg.get("training", {})
    base = dict(train_cfg.get("losses", {}))
    if phase != "sts":
        phase_cfg = cfg.get("phases", {}).get(phase, {})
        base.update(phase_cfg.get("losses", {}))

    weights = {name: float(base.get(name, 0.0)) for name in LOSS_NAMES}
    if "mse" not in base and not any(base.get(name) for name in LOSS_NAMES):
        weights["mse"] = 1.0

    for name, value in weights.items():
        if value < 0:
            raise ValueError(f"loss weight training.losses.{name} must be non-negative, got {value}")

    if sum(weights.values()) <= 0:
        raise ValueError("at least one loss weight must be > 0")

    return weights


def resolve_retrieval_corpus_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = get_resolved_paths(cfg)
    root = paths.get("gpu_data_root") or paths.get("local_data_root")
    if root is None or str(root) == "":
        raise ValueError("Missing data root for retrieval paths")

    explicit_en = paths.get("en_retrieval_corpus")
    explicit_ko = paths.get("ko_retrieval_corpus")
    resolved = {
        "en": explicit_en if explicit_en is not None and str(explicit_en) != "" else (root / "retrieval" / "en" / "corpus.parquet"),
        "ko": explicit_ko if explicit_ko is not None and str(explicit_ko) != "" else (root / "retrieval" / "ko" / "corpus.parquet"),
    }
    return {lang: Path(path).resolve() for lang, path in resolved.items()}


def resolve_retrieval_embedding_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = get_resolved_paths(cfg)
    output_root = paths.get("output_dir")
    if output_root is None or str(output_root) == "":
        raise ValueError("Missing paths.output_dir for retrieval embedding paths")

    explicit_en = paths.get("en_retrieval_embeddings")
    explicit_ko = paths.get("ko_retrieval_embeddings")
    resolved = {
        "en": explicit_en
        if explicit_en is not None and str(explicit_en) != ""
        else (output_root / "retrieval" / "embeddings" / "en_embeddings.parquet"),
        "ko": explicit_ko
        if explicit_ko is not None and str(explicit_ko) != ""
        else (output_root / "retrieval" / "embeddings" / "ko_embeddings.parquet"),
    }
    return {lang: Path(path).resolve() for lang, path in resolved.items()}


def resolve_retrieval_checkpoint_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    paths = get_resolved_paths(cfg)
    output_root = paths.get("output_dir")
    if output_root is None or str(output_root) == "":
        raise ValueError("Missing paths.output_dir for retrieval checkpoint paths")

    explicit_en = paths.get("retrieval_checkpoint_en")
    explicit_final = paths.get("retrieval_checkpoint_final")
    return {
        "en": Path(
            explicit_en
            if explicit_en is not None and str(explicit_en) != ""
            else (output_root / "retrieval" / "checkpoint_en")
        ).resolve(),
        "final": Path(
            explicit_final
            if explicit_final is not None and str(explicit_final) != ""
            else (output_root / "retrieval" / "checkpoint_final")
        ).resolve(),
    }


def get_resolved_paths(cfg: dict[str, Any]) -> dict[str, Path | None]:
    paths = cfg.get("paths", {})
    return {key: resolve_path(value) for key, value in paths.items()}


def require_path(paths: dict[str, Path | None], key: str) -> Path:
    value = paths.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Missing required path in configs/distill.yaml: paths.{key}")
    return value


def resolve_sts_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve local STS parquet paths for each MTEB task name."""
    paths = get_resolved_paths(cfg)
    eval_cfg = cfg.get("eval", {})
    local_sts = eval_cfg.get("local_sts", {})
    task_files: dict[str, str] = local_sts.get("tasks", {})

    sts_root = paths.get("sts_data_root")
    if sts_root is None or str(sts_root) == "":
        sts_root = paths.get("gpu_data_root") or paths.get("local_data_root")
    if sts_root is None or str(sts_root) == "":
        raise ValueError("Missing STS data root: set paths.sts_data_root or local/gpu_data_root")

    resolved: dict[str, Path] = {}
    for task_name, rel_path in task_files.items():
        path = Path(rel_path)
        if not path.is_absolute():
            path = (sts_root / path).resolve()
        resolved[task_name] = path

    explicit = {
        "STSBenchmark": paths.get("en_sts_test"),
        "KorSTS": paths.get("ko_sts_test"),
    }
    for task_name, path in explicit.items():
        if path is not None and str(path) != "":
            resolved[task_name] = path

    return resolved


def resolve_sts_dev_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve optional dev/validation STS parquet paths for debug proxies."""
    paths = get_resolved_paths(cfg)
    local_sts = cfg.get("eval", {}).get("local_sts", {})
    dev_files: dict[str, str] = local_sts.get("dev_splits", {})

    sts_root = paths.get("sts_data_root") or paths.get("gpu_data_root") or paths.get("local_data_root")
    if sts_root is None or str(sts_root) == "":
        return {}

    resolved: dict[str, Path] = {}
    for task_name, rel_path in dev_files.items():
        path = Path(rel_path)
        if not path.is_absolute():
            path = (sts_root / path).resolve()
        resolved[task_name] = path
    return resolved


def resolve_retrieval_eval_paths(cfg: dict[str, Any]) -> dict[str, Path | dict[str, Path]]:
    """Resolve local retrieval eval parquet directories for each MTEB task name."""
    paths = get_resolved_paths(cfg)
    local_retrieval = cfg.get("eval", {}).get("local_retrieval", {})
    task_dirs: dict[str, Any] = local_retrieval.get("tasks", {})

    root = paths.get("retrieval_eval_data_root")
    if root is None or str(root) == "":
        root = paths.get("gpu_data_root") or paths.get("local_data_root")
    if root is None or str(root) == "":
        raise ValueError(
            "Missing retrieval eval data root: set paths.retrieval_eval_data_root or local/gpu_data_root"
        )

    resolved: dict[str, Path | dict[str, Path]] = {}
    for task_name, spec in task_dirs.items():
        if isinstance(spec, dict):
            resolved[task_name] = {
                subset: (root / rel_path).resolve() if not Path(rel_path).is_absolute() else Path(rel_path).resolve()
                for subset, rel_path in spec.items()
            }
        else:
            path = Path(spec)
            if not path.is_absolute():
                path = (root / path).resolve()
            resolved[task_name] = path
    return resolved
