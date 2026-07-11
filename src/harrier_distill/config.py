from __future__ import annotations

import hashlib
import json
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


def load_languages_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "languages.yaml"
    return load_yaml(config_path)


def get_training_order(languages_cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = languages_cfg or load_languages_config()
    order = cfg.get("training_order")
    if not order:
        raise ValueError("languages.yaml missing training_order")
    return list(order)


def get_language_codes(languages_cfg: dict[str, Any] | None = None) -> list[str]:
    return get_training_order(languages_cfg)


def get_language_meta(lang: str, languages_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = languages_cfg or load_languages_config()
    languages = cfg.get("languages", {})
    if lang not in languages:
        available = ", ".join(sorted(languages))
        raise KeyError(f"Unknown language '{lang}'. Available: {available}")
    return languages[lang]


def get_previous_lang(lang: str, languages_cfg: dict[str, Any] | None = None) -> str | None:
    order = get_training_order(languages_cfg)
    if lang not in order:
        raise KeyError(f"Language '{lang}' not in training_order")
    idx = order.index(lang)
    return order[idx - 1] if idx > 0 else None


def parse_lang_list(
    lang_arg: str,
    *,
    languages_cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Parse --lang all | en | en,ko,ar into validated language codes."""
    order = get_training_order(languages_cfg)
    if lang_arg == "all":
        return list(order)
    codes = [part.strip() for part in lang_arg.split(",") if part.strip()]
    unknown = [code for code in codes if code not in order]
    if unknown:
        raise ValueError(f"Unknown language(s): {', '.join(unknown)}. Available: {', '.join(order)}")
    return codes


def get_phase_config(cfg: dict[str, Any], phase: str) -> dict[str, Any]:
    phases = cfg.get("phases", {})
    if phase not in phases:
        raise KeyError(f"Unknown phase '{phase}'. Available: {', '.join(sorted(phases)) or '(none)'}")
    return phases[phase]


LOSS_NAMES = ("mse", "cosine", "pairwise_mse", "score_kl")


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


def get_score_kl_temperature(cfg: dict[str, Any], phase: str) -> float:
    """Merge training.score_kl_temperature with phases.<phase> override."""
    train_cfg = cfg.get("training", {})
    temperature = float(train_cfg.get("score_kl_temperature", 0.05))
    if phase != "sts":
        phase_cfg = cfg.get("phases", {}).get(phase, {})
        if "score_kl_temperature" in phase_cfg:
            temperature = float(phase_cfg["score_kl_temperature"])
    if temperature <= 0:
        raise ValueError(f"score_kl_temperature must be > 0, got {temperature}")
    return temperature


def get_resolved_paths(cfg: dict[str, Any]) -> dict[str, Path | None]:
    paths = cfg.get("paths", {})
    return {key: resolve_path(value) for key, value in paths.items()}


def require_path(paths: dict[str, Path | None], key: str) -> Path:
    value = paths.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Missing required path in configs/distill.yaml: paths.{key}")
    return value


def resolve_data_root(cfg: dict[str, Any], *, prefer_gpu: bool = True) -> Path:
    paths = get_resolved_paths(cfg)
    if prefer_gpu:
        root = paths.get("gpu_data_root") or paths.get("local_data_root")
    else:
        root = paths.get("local_data_root") or paths.get("gpu_data_root")
    if root is None or str(root) == "":
        raise ValueError("Missing data root: set paths.local_data_root or paths.gpu_data_root")
    return Path(root).resolve()


def resolve_output_root(cfg: dict[str, Any]) -> Path:
    return require_path(get_resolved_paths(cfg), "output_dir")


def _explicit_lang_path(paths: dict[str, Path | None], key: str) -> Path | None:
    value = paths.get(key)
    if value is not None and str(value) != "":
        return Path(value).resolve()
    return None


def resolve_corpus_path(cfg: dict[str, Any], lang: str, *, phase: str = "sts") -> Path:
    paths = get_resolved_paths(cfg)
    if phase == "retrieval":
        explicit = _explicit_lang_path(paths, f"{lang}_retrieval_corpus")
        if explicit is not None:
            return explicit
        return resolve_data_root(cfg) / "retrieval" / lang / "corpus.parquet"

    explicit = _explicit_lang_path(paths, f"{lang}_corpus")
    if explicit is not None:
        return explicit
    return resolve_data_root(cfg) / lang / "corpus.parquet"


def resolve_embedding_path(cfg: dict[str, Any], lang: str, *, phase: str = "sts") -> Path:
    paths = get_resolved_paths(cfg)
    output_root = resolve_output_root(cfg)
    if phase == "retrieval":
        explicit = _explicit_lang_path(paths, f"{lang}_retrieval_embeddings")
        if explicit is not None:
            return explicit
        return output_root / "retrieval" / "embeddings" / f"{lang}_embeddings.parquet"

    explicit = _explicit_lang_path(paths, f"{lang}_embeddings")
    if explicit is not None:
        return explicit
    return output_root / "embeddings" / f"{lang}_embeddings.parquet"


def resolve_sts_checkpoint_path(cfg: dict[str, Any], lang: str) -> Path:
    paths = get_resolved_paths(cfg)
    output_root = resolve_output_root(cfg)
    explicit = _explicit_lang_path(paths, f"sts_checkpoint_{lang}")
    if explicit is not None:
        return explicit

    default = output_root / "checkpoints" / "sts" / lang
    order = get_training_order()
    if lang == "en":
        legacy = output_root / "checkpoint_en"
        if legacy.exists() and not default.exists():
            return legacy.resolve()
    if lang in {order[-1], "ko"}:
        legacy = output_root / "checkpoint_final"
        if legacy.exists() and not default.exists():
            return legacy.resolve()
    return default


def resolve_retrieval_checkpoint_path(cfg: dict[str, Any], lang: str) -> Path:
    paths = get_resolved_paths(cfg)
    output_root = resolve_output_root(cfg)
    explicit = _explicit_lang_path(paths, f"retrieval_checkpoint_{lang}")
    if explicit is not None:
        return explicit
    order = get_training_order()
    if lang == order[-1]:
        legacy = paths.get("retrieval_checkpoint_final")
        if legacy is not None and str(legacy) != "":
            return Path(legacy).resolve()
        legacy_path = output_root / "retrieval" / "checkpoint_final"
        if legacy_path.exists():
            return legacy_path.resolve()
    if lang == "en":
        legacy = paths.get("retrieval_checkpoint_en")
        if legacy is not None and str(legacy) != "":
            return Path(legacy).resolve()
        legacy_path = output_root / "retrieval" / "checkpoint_en"
        if legacy_path.exists():
            return legacy_path.resolve()
    return output_root / "retrieval" / "checkpoints" / lang


def get_num_epochs(cfg: dict[str, Any], lang: str, phase: str) -> int:
    train_cfg = cfg.get("training", {})
    languages_cfg = load_languages_config()
    lang_meta = get_language_meta(lang, languages_cfg)

    if phase == "retrieval":
        phase_cfg = get_phase_config(cfg, "retrieval")
        legacy_key = f"num_epochs_retrieval_{lang}"
        if legacy_key in phase_cfg:
            return int(phase_cfg[legacy_key])
        if "num_epochs_retrieval" in lang_meta:
            return int(lang_meta["num_epochs_retrieval"])
        return int(phase_cfg.get("default_num_epochs", train_cfg.get("default_num_epochs", 1)))

    legacy_key = f"num_epochs_{lang}"
    if legacy_key in train_cfg:
        return int(train_cfg[legacy_key])
    if "num_epochs_sts" in lang_meta:
        return int(lang_meta["num_epochs_sts"])
    return int(train_cfg.get("default_num_epochs", 1))


def resolve_retrieval_corpus_path(cfg: dict[str, Any], lang: str) -> Path:
    return resolve_corpus_path(cfg, lang, phase="retrieval")


def resolve_retrieval_embedding_path(cfg: dict[str, Any], lang: str) -> Path:
    return resolve_embedding_path(cfg, lang, phase="retrieval")


def resolve_retrieval_corpus_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {lang: resolve_retrieval_corpus_path(cfg, lang) for lang in get_language_codes()}


def resolve_retrieval_embedding_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {lang: resolve_retrieval_embedding_path(cfg, lang) for lang in get_language_codes()}


def resolve_retrieval_checkpoint_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    order = get_training_order()
    paths = {lang: resolve_retrieval_checkpoint_path(cfg, lang) for lang in order}
    paths["final"] = resolve_retrieval_checkpoint_path(cfg, order[-1])
    return paths


def apply_sample_overrides(
    datasets_cfg: dict[str, Any],
    distill_cfg: dict[str, Any],
    *,
    langs: list[str],
) -> None:
    """Apply pilot_samples_per_lang or full_samples_per_lang overrides in-place."""
    data_cfg = distill_cfg.get("data", {})
    pilot = int(data_cfg.get("pilot_samples_per_lang", 0))
    full = int(data_cfg.get("full_samples_per_lang", 0))

    if pilot > 0:
        cap = pilot
    elif full > 0:
        cap = full
    else:
        return

    for lang in langs:
        if lang not in datasets_cfg or not isinstance(datasets_cfg[lang], dict):
            continue
        lang_cfg = datasets_cfg[lang]
        lang_cfg["target_samples"] = cap
        sources = lang_cfg.get("sources", [])
        if not sources:
            continue
        total_source = sum(int(s.get("target_samples", 0)) for s in sources)
        if total_source <= 0:
            continue
        for source in sources:
            frac = int(source.get("target_samples", 0)) / total_source
            source["target_samples"] = max(int(cap * frac), 1)


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


def parquet_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_download_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def write_download_manifest(manifest_path: Path, entry: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)


def should_skip_download(
    *,
    output_path: Path,
    manifest_path: Path,
    target_rows: int,
    force: bool,
    skip_existing: bool,
    expected_negatives_per_query: int | None = None,
    expected_negatives_per_triplet: int | None = None,
) -> bool:
    """Return True when an existing download can be reused.

    STS manifests use ``target_samples`` + ``rows``. Retrieval manifests use
    ``triplet_count`` (actual triplets written) plus ``target_triplets`` (configured
    target). Changing the configured target, MIRACL ``negatives_per_query``, or
    bulk ``negatives_per_triplet`` invalidates a stale corpus.

    Legacy retrieval manifests that only store ``target_triplets`` without
    ``triplet_count`` are treated as stale and are not skipped.
    """
    if force or not skip_existing:
        return False
    if not output_path.exists():
        return False
    manifest = read_download_manifest(manifest_path)
    if manifest is None:
        return False

    if "target_triplets" in manifest or "triplet_count" in manifest:
        if "triplet_count" not in manifest:
            return False
        # Require the configured target to match so lowering 1M → 350k rebuilds.
        if int(manifest.get("target_triplets", -1)) != int(target_rows):
            return False
        actual_triplets = int(manifest["triplet_count"])
        if actual_triplets < int(target_rows):
            return False
        if expected_negatives_per_query is not None:
            stored = manifest.get("negatives_per_query")
            if stored is None or int(stored) != int(expected_negatives_per_query):
                return False
        if expected_negatives_per_triplet is not None:
            stored_nt = manifest.get("negatives_per_triplet")
            if stored_nt is None or int(stored_nt) != int(expected_negatives_per_triplet):
                return False
        return True

    rows = int(manifest.get("rows", 0))
    target = int(manifest.get("target_samples", target_rows))
    return rows >= target
