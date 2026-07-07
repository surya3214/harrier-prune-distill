#!/usr/bin/env python3
"""Validate HF dataset paths/configs/splits referenced in YAML configs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import load_datasets_config, load_sts_datasets_config
from harrier_distill.data import resolve_hf_source_splits


@dataclass(frozen=True)
class Check:
    label: str
    hf_path: str
    config: str | None
    split: str
    loader: str | None = None
    extra: dict[str, Any] | None = None


def _add(checks: list[Check], *, label: str, hf_path: str, config: str | None, split: str, **extra: Any) -> None:
    loader = extra.pop("loader", None)
    checks.append(Check(label=label, hf_path=hf_path, config=config, split=split, loader=loader, extra=extra or None))


def collect_distillation_checks(datasets_cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for lang, lang_cfg in datasets_cfg.items():
        if not isinstance(lang_cfg, dict) or "sources" not in lang_cfg:
            continue
        for source in lang_cfg["sources"]:
            hf_path = source["hf_path"]
            config = source.get("config")
            for split in resolve_hf_source_splits(source, lang):
                _add(
                    checks,
                    label=f"datasets.yaml:{lang}/{source['name']}",
                    hf_path=hf_path,
                    config=config,
                    split=split,
                    streaming=source.get("streaming", False),
                )
    return checks


def collect_sts_checks(sts_cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for task_name, task_cfg in sts_cfg.items():
        if not isinstance(task_cfg, dict):
            continue
        for source in task_cfg.get("sources", []):
            _add(
                checks,
                label=f"sts_datasets.yaml:{task_name}/{source['name']}",
                hf_path=source["hf_path"],
                config=source.get("hf_subset"),
                split=source["split"],
            )
    return checks


def collect_retrieval_checks(retrieval_cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []

    def add_source(label: str, source: dict[str, Any]) -> None:
        name = source.get("name", label)
        loader = source.get("loader")
        if loader == "unicamp_mmarco":
            _add(
                checks,
                label=f"{label}/{name}",
                hf_path=source["hf_path"],
                config=source["config"],
                split=source.get("split", "train"),
                loader=loader,
                translation=source.get("translation", "google"),
                queries_relpath=source.get(
                    "queries_relpath",
                    f"queries/dev/{source['config']}_queries.dev.small.tsv",
                ),
            )
            return
        if loader == "maupqa_csv":
            _add(
                checks,
                label=f"{label}/{name}",
                hf_path=source["hf_path"],
                config=None,
                split=source.get("split", "train"),
                loader=loader,
                csv_subsets=source.get("csv_subsets", ["msmarco"]),
            )
            return
        if "qrels_config" in source:
            hf_path = source["hf_path"]
            _add(checks, label=f"{label}/{name}:qrels", hf_path=hf_path, config=source["qrels_config"], split=source["qrels_split"])
            _add(checks, label=f"{label}/{name}:queries", hf_path=hf_path, config=source["queries_config"], split=source["queries_split"])
            _add(checks, label=f"{label}/{name}:corpus", hf_path=hf_path, config=source["corpus_config"], split=source["corpus_split"])
        else:
            _add(
                checks,
                label=f"{label}/{name}",
                hf_path=source["hf_path"],
                config=source.get("config"),
                split=source.get("split", "train"),
                streaming=source.get("streaming", True),
            )

    for lang in retrieval_cfg.get("languages", []):
        if lang in retrieval_cfg and isinstance(retrieval_cfg[lang], dict):
            for source in retrieval_cfg[lang].get("sources", []):
                add_source(f"retrieval_datasets.yaml:{lang}", source)

    for pilot_lang, pilot_cfg in retrieval_cfg.get("pilot_sources", {}).items():
        for source in pilot_cfg.get("sources", []):
            add_source(f"retrieval_datasets.yaml:pilot:{pilot_lang}", source)

    return checks


def collect_retrieval_eval_checks(eval_cfg: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    for task_name, task_cfg in eval_cfg.items():
        if not isinstance(task_cfg, dict) or "hf_path" not in task_cfg:
            continue
        hf_path = task_cfg["hf_path"]
        if "languages" in task_cfg:
            for lang, lang_cfg in task_cfg["languages"].items():
                for part in ("queries", "corpus", "qrels"):
                    part_cfg = lang_cfg[part]
                    _add(
                        checks,
                        label=f"retrieval_eval_datasets.yaml:{task_name}/{lang}:{part}",
                        hf_path=hf_path,
                        config=part_cfg["config"],
                        split=part_cfg["split"],
                    )
        else:
            for part in ("queries", "corpus", "qrels"):
                part_cfg = task_cfg[part]
                _add(
                    checks,
                    label=f"retrieval_eval_datasets.yaml:{task_name}:{part}",
                    hf_path=hf_path,
                    config=part_cfg["config"],
                    split=part_cfg["split"],
                )
    return checks


def verify_check(check: Check) -> str | None:
    if check.loader == "unicamp_mmarco":
        return _verify_unicamp_mmarco(check)
    if check.loader == "maupqa_csv":
        return _verify_maupqa_csv(check)
    return _verify_hf_dataset(check)


def _verify_hf_dataset(check: Check) -> str | None:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": check.hf_path, "split": check.split, "streaming": True}
    if check.config is not None:
        kwargs["name"] = check.config
    try:
        dataset = load_dataset(**kwargs)
        next(iter(dataset))
        return None
    except StopIteration:
        return None
    except Exception as exc:
        return str(exc)


def _verify_unicamp_mmarco(check: Check) -> str | None:
    from huggingface_hub import hf_hub_download

    extra = check.extra or {}
    lang = check.config
    translation = extra.get("translation", "google")
    repo = check.hf_path
    queries_relpath = extra.get("queries_relpath", f"queries/dev/{lang}_queries.dev.small.tsv")
    try:
        hf_hub_download(repo, f"data/{translation}/{queries_relpath}", repo_type="dataset")
        hf_hub_download(
            repo,
            f"data/{translation}/collections/{lang}_collection.tsv",
            repo_type="dataset",
        )
        hf_hub_download(
            repo,
            f"data/{translation}/runs/run.bm25_{lang}-msmarco.txt",
            repo_type="dataset",
        )
        return None
    except Exception as exc:
        return str(exc)


def _verify_maupqa_csv(check: Check) -> str | None:
    from datasets import load_dataset

    extra = check.extra or {}
    subsets = extra.get("csv_subsets", ["msmarco"])
    repo = check.hf_path
    data_files = [f"hf://datasets/{repo}/data/{subset}/train-v2.0.0.csv" for subset in subsets[:1]]
    try:
        dataset = load_dataset("csv", data_files=data_files, split="train", streaming=True)
        next(iter(dataset))
        return None
    except StopIteration:
        return None
    except Exception as exc:
        return str(exc)


def dedupe_checks(checks: list[Check]) -> list[Check]:
    seen: set[tuple[str, str | None, str, str | None]] = set()
    unique: list[Check] = []
    for check in checks:
        key = (check.hf_path, check.config, check.split, check.loader)
        if key in seen:
            continue
        seen.add(key)
        unique.append(check)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-config", default=str(PROJECT_ROOT / "configs" / "datasets.yaml"))
    parser.add_argument("--sts-config", default=str(PROJECT_ROOT / "configs" / "sts_datasets.yaml"))
    parser.add_argument("--retrieval-config", default=str(PROJECT_ROOT / "configs" / "retrieval_datasets.yaml"))
    parser.add_argument("--retrieval-eval-config", default=str(PROJECT_ROOT / "configs" / "retrieval_eval_datasets.yaml"))
    parser.add_argument("--quiet", action="store_true", help="Only print failures and summary")
    args = parser.parse_args()

    checks: list[Check] = []
    checks.extend(collect_distillation_checks(load_datasets_config(args.datasets_config)))
    checks.extend(collect_sts_checks(load_sts_datasets_config(args.sts_config)))
    with open(args.retrieval_config, encoding="utf-8") as f:
        checks.extend(collect_retrieval_checks(yaml.safe_load(f)))
    with open(args.retrieval_eval_config, encoding="utf-8") as f:
        checks.extend(collect_retrieval_eval_checks(yaml.safe_load(f)))

    checks = dedupe_checks(checks)
    failures: list[tuple[Check, str]] = []
    passed = 0

    if not args.quiet:
        print(f"Validating {len(checks)} unique dataset references...\n")

    for check in checks:
        error = verify_check(check)
        if error:
            failures.append((check, error))
            print(f"FAIL  {check.label}")
            print(f"      {check.hf_path} config={check.config!r} split={check.split!r} loader={check.loader!r}")
            print(f"      -> {error}\n")
        else:
            passed += 1
            if not args.quiet:
                print(f"OK    {check.label}")

    print(f"\nSummary: {passed} passed, {len(failures)} failed, {len(checks)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
