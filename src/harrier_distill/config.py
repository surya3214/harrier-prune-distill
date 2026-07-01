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


def get_resolved_paths(cfg: dict[str, Any]) -> dict[str, Path | None]:
    paths = cfg.get("paths", {})
    return {key: resolve_path(value) for key, value in paths.items()}


def require_path(paths: dict[str, Path | None], key: str) -> Path:
    value = paths.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Missing required path in configs/distill.yaml: paths.{key}")
    return value
