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
