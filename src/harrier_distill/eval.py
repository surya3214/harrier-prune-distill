from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mteb

from harrier_distill.model import load_sentence_transformer
from harrier_distill.mteb_sts import mteb_eng_v2_sts_task_names, resolve_mteb_sts_task_objects
from harrier_distill.retrieval_eval import (
    RetrievalTaskPaths,
    evaluate_retrieval_local,
    get_local_retrieval_task_paths,
)
from harrier_distill.sts import evaluate_sts_local

MTEB_ENG_V2_STS = mteb_eng_v2_sts_task_names()

STS_SUITES: dict[str, list[str]] = {
    "en": list(MTEB_ENG_V2_STS),
    "ko": ["KorSTS"],
    "multilingual": [*MTEB_ENG_V2_STS, "KorSTS"],
    "extended": [*MTEB_ENG_V2_STS, "KorSTS"],
}

RETRIEVAL_SUITES: dict[str, list[str]] = {
    "en": ["MSMARCO"],
    "ko": ["MIRACLRetrieval"],
    "en_ko": ["MSMARCO", "MIRACLRetrieval"],
}


def get_retrieval_tasks_for_suite(suite: str, *, tasks: list[str] | None = None) -> list[str]:
    if tasks:
        return tasks
    if suite not in RETRIEVAL_SUITES:
        available = ", ".join(sorted(RETRIEVAL_SUITES))
        raise ValueError(f"Unknown retrieval suite '{suite}'. Available: {available}")
    return list(RETRIEVAL_SUITES[suite])


def _apply_retrieval_prompts(model, task_names: list[str], prompt_name: str) -> None:
    if not prompt_name or not getattr(model, "prompts", None):
        return
    prompts = dict(model.prompts)
    if prompt_name not in prompts:
        return
    retrieval_instruction = prompts[prompt_name]
    for task_name in task_names:
        prompts[task_name] = retrieval_instruction
    prompts["Retrieval"] = retrieval_instruction
    prompts["Query"] = retrieval_instruction
    model.prompts = prompts


def _miracl_eval_subsets(languages_cfg: dict[str, Any] | list[str] | None) -> list[str] | None:
    if languages_cfg is None:
        return ["en", "ko"]
    if isinstance(languages_cfg, list):
        mapping = {
            "eng-Latn": "en",
            "kor-Kore": "ko",
            "en": "en",
            "ko": "ko",
        }
        return [mapping.get(lang, lang) for lang in languages_cfg]
    miracl_langs = languages_cfg.get("MIRACLRetrieval")
    if miracl_langs is None:
        return ["en", "ko"]
    mapping = {"eng-Latn": "en", "kor-Kore": "ko", "en": "en", "ko": "ko"}
    return [mapping.get(lang, lang) for lang in miracl_langs]


def _filter_local_retrieval_paths(
    local_task_paths: dict[str, RetrievalTaskPaths],
    *,
    miracl_subsets: list[str] | None,
) -> dict[str, RetrievalTaskPaths]:
    if miracl_subsets is None or "MIRACLRetrieval" not in local_task_paths:
        return local_task_paths

    miracl_paths = local_task_paths["MIRACLRetrieval"]
    if not isinstance(miracl_paths, dict):
        return local_task_paths

    filtered = dict(local_task_paths)
    filtered["MIRACLRetrieval"] = {
        subset: path for subset, path in miracl_paths.items() if subset in miracl_subsets
    }
    if not filtered["MIRACLRetrieval"]:
        raise ValueError(
            f"No MIRACL local subsets match miracl_subsets={miracl_subsets}. "
            f"Available: {', '.join(sorted(miracl_paths))}"
        )
    return filtered


def evaluate_retrieval(
    model_path: str | Path,
    *,
    tasks: list[str] | None = None,
    query_prompt: str = "web_search_query",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    miracl_subsets: list[str] | None = None,
    use_local_retrieval: bool = False,
    local_task_paths: dict[str, RetrievalTaskPaths] | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    """Run retrieval tasks via MTEB or local parquet and return per-task nDCG@10."""
    task_names = tasks or ["MSMARCO", "MIRACLRetrieval"]
    if use_local_retrieval:
        if local_task_paths is None:
            raise ValueError("local_task_paths is required when use_local_retrieval=True")
        filtered_paths = _filter_local_retrieval_paths(
            get_local_retrieval_task_paths(task_names, local_task_paths),
            miracl_subsets=miracl_subsets,
        )
        return evaluate_retrieval_local(
            model_path,
            task_names=task_names,
            local_task_paths=filtered_paths,
            query_prompt=query_prompt,
            batch_size=batch_size,
            max_length=max_length,
        )

    mteb_tasks = mteb.get_tasks(tasks=task_names)

    model = load_sentence_transformer(model_path)
    _apply_retrieval_prompts(model, task_names, query_prompt)

    evaluation = mteb.MTEB(tasks=mteb_tasks)
    encode_kwargs = {"batch_size": batch_size, "show_progress_bar": True, "prompt_name": query_prompt}
    eval_subsets = miracl_subsets if miracl_subsets is not None else ["en", "ko"]

    results = evaluation.run(
        model,
        output_folder=str(output_dir) if output_dir else None,
        encode_kwargs=encode_kwargs,
        eval_subsets=eval_subsets if "MIRACLRetrieval" in task_names else None,
    )

    summary: dict[str, Any] = {
        "model_path": str(model_path),
        "backend": "mteb",
        "query_prompt": query_prompt,
        "tasks": {},
    }
    for result in results:
        task_name = result.task_name
        summary["tasks"][task_name] = {
            "main_score": _extract_main_score(result),
            "scores": result.scores,
            "hf_subset": getattr(result, "hf_subset", None),
            "languages": getattr(result, "languages", None),
        }
    return summary


def compare_retrieval(
    *,
    teacher_path: str | Path,
    student_path: str | Path,
    baseline_path: str | Path | None = None,
    suite: str = "en_ko",
    tasks: list[str] | None = None,
    query_prompt: str = "web_search_query",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    miracl_subsets: list[str] | None = None,
    use_local_retrieval: bool = False,
    local_task_paths: dict[str, RetrievalTaskPaths] | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    """Evaluate teacher and student (and optional baseline) on retrieval tasks."""
    task_names = get_retrieval_tasks_for_suite(suite, tasks=tasks)
    if use_local_retrieval:
        unsupported = [name for name in task_names if name not in (local_task_paths or {})]
        if unsupported:
            raise ValueError(
                f"Local retrieval mode does not support tasks: {', '.join(unsupported)}. "
                "Run scripts/01_download_retrieval_eval_local.py and configure eval.local_retrieval."
            )

    output_root = Path(output_dir) if output_dir else None
    mteb_root = output_root / "mteb_runs" if output_root and not use_local_retrieval else None

    models: list[tuple[str, str | Path]] = [
        ("teacher", teacher_path),
        ("student", student_path),
    ]
    if baseline_path is not None:
        models.append(("baseline", baseline_path))

    summaries: dict[str, dict[str, Any]] = {}
    for label, model_path in models:
        model_mteb_dir = mteb_root / label if mteb_root else None
        summaries[label] = evaluate_retrieval(
            model_path,
            tasks=task_names,
            query_prompt=query_prompt,
            batch_size=batch_size,
            output_dir=model_mteb_dir,
            miracl_subsets=miracl_subsets,
            use_local_retrieval=use_local_retrieval,
            local_task_paths=local_task_paths,
            max_length=max_length,
        )

    comparison_rows: list[dict[str, Any]] = []
    for task_name in task_names:
        scores = {
            label: summaries[label]["tasks"].get(task_name, {}).get("main_score")
            for label in summaries
        }
        comparison_rows.append(_build_comparison_row(task_name, scores))

    macro: dict[str, Any] = {
        "teacher": _macro_average(comparison_rows, "teacher"),
        "student": _macro_average(comparison_rows, "student"),
    }
    if baseline_path is not None:
        macro["baseline"] = _macro_average(comparison_rows, "baseline")
    if macro["teacher"] is not None and macro["student"] is not None:
        macro["delta"] = macro["student"] - macro["teacher"]
        macro["pct_of_teacher"] = (
            macro["student"] / macro["teacher"] * 100.0 if macro["teacher"] != 0 else None
        )

    return {
        "backend": "local" if use_local_retrieval else "mteb",
        "suite": suite,
        "tasks": task_names,
        "teacher_path": str(teacher_path),
        "student_path": str(student_path),
        "baseline_path": str(baseline_path) if baseline_path else None,
        "summaries": summaries,
        "comparison": comparison_rows,
        "macro": macro,
    }


def print_retrieval_summary(summary: dict[str, Any]) -> None:
    print(f"\nModel: {summary['model_path']}")
    for task_name, payload in summary.get("tasks", {}).items():
        score = payload.get("main_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        subset = payload.get("hf_subset")
        suffix = f" ({subset})" if subset else ""
        print(f"  {task_name}{suffix}: {score_str}")


def print_retrieval_compare_summary(comparison: dict[str, Any]) -> None:
    has_baseline = comparison.get("baseline_path") is not None
    print(f"\nRetrieval comparison (suite={comparison['suite']})")
    print(f"  Teacher: {comparison['teacher_path']}")
    print(f"  Student: {comparison['student_path']}")
    if has_baseline:
        print(f"  Baseline: {comparison['baseline_path']}")

    if has_baseline:
        header = f"{'Task':<28} {'Teacher':>9} {'Student':>9} {'Baseline':>9} {'Delta':>9} {'%Teacher':>9}"
    else:
        header = f"{'Task':<28} {'Teacher':>9} {'Student':>9} {'Delta':>9} {'%Teacher':>9}"
    print(header)
    print("-" * len(header))

    for row in comparison["comparison"]:
        teacher = row.get("teacher")
        student = row.get("student")
        teacher_str = f"{teacher:.4f}" if teacher is not None else "n/a"
        student_str = f"{student:.4f}" if student is not None else "n/a"
        delta = row.get("delta")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        pct = row.get("pct_of_teacher")
        pct_str = f"{pct:.1f}%" if pct is not None else "n/a"

        if has_baseline:
            baseline = row.get("baseline")
            baseline_str = f"{baseline:.4f}" if baseline is not None else "n/a"
            print(
                f"{row['task']:<28} {teacher_str:>9} {student_str:>9} "
                f"{baseline_str:>9} {delta_str:>9} {pct_str:>9}"
            )
        else:
            print(
                f"{row['task']:<28} {teacher_str:>9} {student_str:>9} "
                f"{delta_str:>9} {pct_str:>9}"
            )

    macro = comparison["macro"]
    teacher = macro.get("teacher")
    student = macro.get("student")
    teacher_str = f"{teacher:.4f}" if teacher is not None else "n/a"
    student_str = f"{student:.4f}" if student is not None else "n/a"
    delta = macro.get("delta")
    delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
    pct = macro.get("pct_of_teacher")
    pct_str = f"{pct:.1f}%" if pct is not None else "n/a"
    print(
        f"{'MACRO AVG':<28} {teacher_str:>9} {student_str:>9} "
        f"{delta_str:>9} {pct_str:>9}"
    )


def get_tasks_for_suite(suite: str, *, tasks: list[str] | None = None) -> list[str]:
    if tasks:
        return tasks
    if suite not in STS_SUITES:
        available = ", ".join(sorted(STS_SUITES))
        raise ValueError(f"Unknown suite '{suite}'. Available: {available}")
    return list(STS_SUITES[suite])


def _apply_sts_prompts(model, task_names: list[str], prompt_name: str) -> None:
    if not prompt_name or not getattr(model, "prompts", None):
        return
    prompts = dict(model.prompts)
    if prompt_name not in prompts:
        return
    sts_instruction = prompts[prompt_name]
    for task_name in task_names:
        prompts[task_name] = sts_instruction
    prompts["STS"] = sts_instruction
    model.prompts = prompts


def _extract_main_score(result) -> float | None:
    if not result.scores:
        return None
    split_scores = result.scores.get("test") or result.scores.get("validation") or []
    if not split_scores:
        return None
    return split_scores[0].get("main_score")


def get_local_task_paths(
    task_names: list[str],
    sts_paths: dict[str, Path],
) -> dict[str, Path]:
    missing = [name for name in task_names if name not in sts_paths]
    if missing:
        raise ValueError(
            f"No local STS parquet configured for tasks: {', '.join(missing)}. "
            "Run scripts/01_download_sts_local.py and set paths.sts_data_root."
        )
    selected = {name: sts_paths[name] for name in task_names}
    missing_files = [name for name, path in selected.items() if not path.exists()]
    if missing_files:
        paths_str = ", ".join(str(selected[name]) for name in missing_files)
        raise FileNotFoundError(
            f"Local STS parquet missing for {', '.join(missing_files)}: {paths_str}"
        )
    return selected


def evaluate_sts(
    model_path: str | Path,
    *,
    tasks: list[str] | None = None,
    prompt_name: str = "sts_query",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    use_local_sts: bool = False,
    local_task_paths: dict[str, Path] | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    """Run STS tasks via MTEB or local parquet and return per-task Spearman scores."""
    task_names = tasks or [*MTEB_ENG_V2_STS, "KorSTS"]
    if use_local_sts:
        if local_task_paths is None:
            raise ValueError("local_task_paths is required when use_local_sts=True")
        return evaluate_sts_local(
            model_path,
            task_paths=get_local_task_paths(task_names, local_task_paths),
            prompt_name=prompt_name,
            batch_size=batch_size,
            max_length=max_length,
        )

    mteb_tasks = resolve_mteb_sts_task_objects(task_names)

    model = load_sentence_transformer(model_path)
    _apply_sts_prompts(model, task_names, prompt_name)

    evaluation = mteb.MTEB(tasks=mteb_tasks)
    results = evaluation.run(
        model,
        output_folder=str(output_dir) if output_dir else None,
        encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
    )

    summary: dict[str, Any] = {"model_path": str(model_path), "backend": "mteb", "tasks": {}}
    for result in results:
        task_name = result.task_name
        summary["tasks"][task_name] = {
            "main_score": _extract_main_score(result),
            "scores": result.scores,
        }
    return summary


def _build_comparison_row(
    task_name: str,
    scores: dict[str, float | None],
) -> dict[str, Any]:
    teacher = scores.get("teacher")
    student = scores.get("student")
    row: dict[str, Any] = {
        "task": task_name,
        "teacher": teacher,
        "student": student,
    }
    if teacher is not None and student is not None:
        row["delta"] = student - teacher
        row["pct_of_teacher"] = (student / teacher * 100.0) if teacher != 0 else None
    if "baseline" in scores:
        baseline = scores["baseline"]
        row["baseline"] = baseline
        if baseline is not None and student is not None:
            row["student_vs_baseline"] = student - baseline
    return row


def _macro_average(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def compare_sts(
    *,
    teacher_path: str | Path,
    student_path: str | Path,
    baseline_path: str | Path | None = None,
    suite: str = "multilingual",
    tasks: list[str] | None = None,
    prompt_name: str = "sts_query",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    use_local_sts: bool = False,
    local_task_paths: dict[str, Path] | None = None,
    max_length: int = 512,
) -> dict[str, Any]:
    """Evaluate teacher and student (and optional baseline) on the same STS suite."""
    task_names = get_tasks_for_suite(suite, tasks=tasks)
    if use_local_sts:
        unsupported = [name for name in task_names if name not in (local_task_paths or {})]
        if unsupported:
            raise ValueError(
                f"Local STS mode does not support tasks: {', '.join(unsupported)}. "
                "Run scripts/01_download_sts_local.py and set eval.local_sts.tasks in distill.yaml."
            )

    output_root = Path(output_dir) if output_dir else None
    mteb_root = output_root / "mteb_runs" if output_root else None

    models: list[tuple[str, str | Path]] = [
        ("teacher", teacher_path),
        ("student", student_path),
    ]
    if baseline_path is not None:
        models.append(("baseline", baseline_path))

    summaries: dict[str, dict[str, Any]] = {}
    for label, model_path in models:
        model_mteb_dir = mteb_root / label if mteb_root and not use_local_sts else None
        summaries[label] = evaluate_sts(
            model_path,
            tasks=task_names,
            prompt_name=prompt_name,
            batch_size=batch_size,
            output_dir=model_mteb_dir,
            use_local_sts=use_local_sts,
            local_task_paths=local_task_paths,
            max_length=max_length,
        )

    comparison_rows: list[dict[str, Any]] = []
    for task_name in task_names:
        scores = {
            label: summaries[label]["tasks"].get(task_name, {}).get("main_score")
            for label in summaries
        }
        comparison_rows.append(_build_comparison_row(task_name, scores))

    macro: dict[str, Any] = {
        "teacher": _macro_average(comparison_rows, "teacher"),
        "student": _macro_average(comparison_rows, "student"),
    }
    if baseline_path is not None:
        macro["baseline"] = _macro_average(comparison_rows, "baseline")
    if macro["teacher"] is not None and macro["student"] is not None:
        macro["delta"] = macro["student"] - macro["teacher"]
        macro["pct_of_teacher"] = (
            macro["student"] / macro["teacher"] * 100.0 if macro["teacher"] != 0 else None
        )

    return {
        "backend": "local" if use_local_sts else "mteb",
        "suite": suite,
        "tasks": task_names,
        "teacher_path": str(teacher_path),
        "student_path": str(student_path),
        "baseline_path": str(baseline_path) if baseline_path else None,
        "summaries": summaries,
        "comparison": comparison_rows,
        "macro": macro,
    }


def print_eval_summary(summary: dict[str, Any]) -> None:
    print(f"\nModel: {summary['model_path']}")
    for task_name, payload in summary.get("tasks", {}).items():
        score = payload.get("main_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  {task_name}: {score_str}")


def print_compare_summary(comparison: dict[str, Any]) -> None:
    has_baseline = comparison.get("baseline_path") is not None
    print(f"\nSTS comparison (suite={comparison['suite']})")
    print(f"  Teacher: {comparison['teacher_path']}")
    print(f"  Student: {comparison['student_path']}")
    if has_baseline:
        print(f"  Baseline: {comparison['baseline_path']}")

    if has_baseline:
        header = f"{'Task':<28} {'Teacher':>9} {'Student':>9} {'Baseline':>9} {'Delta':>9} {'%Teacher':>9}"
    else:
        header = f"{'Task':<28} {'Teacher':>9} {'Student':>9} {'Delta':>9} {'%Teacher':>9}"
    print(header)
    print("-" * len(header))

    for row in comparison["comparison"]:
        teacher = row.get("teacher")
        student = row.get("student")
        teacher_str = f"{teacher:.4f}" if teacher is not None else "n/a"
        student_str = f"{student:.4f}" if student is not None else "n/a"
        delta = row.get("delta")
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        pct = row.get("pct_of_teacher")
        pct_str = f"{pct:.1f}%" if pct is not None else "n/a"

        if has_baseline:
            baseline = row.get("baseline")
            baseline_str = f"{baseline:.4f}" if baseline is not None else "n/a"
            print(
                f"{row['task']:<28} {teacher_str:>9} {student_str:>9} "
                f"{baseline_str:>9} {delta_str:>9} {pct_str:>9}"
            )
        else:
            print(
                f"{row['task']:<28} {teacher_str:>9} {student_str:>9} "
                f"{delta_str:>9} {pct_str:>9}"
            )

    macro = comparison["macro"]
    teacher = macro.get("teacher")
    student = macro.get("student")
    teacher_str = f"{teacher:.4f}" if teacher is not None else "n/a"
    student_str = f"{student:.4f}" if student is not None else "n/a"
    delta = macro.get("delta")
    delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
    pct = macro.get("pct_of_teacher")
    pct_str = f"{pct:.1f}%" if pct is not None else "n/a"

    if has_baseline:
        baseline = macro.get("baseline")
        baseline_str = f"{baseline:.4f}" if baseline is not None else "n/a"
        print(
            f"{'MACRO AVG':<28} {teacher_str:>9} {student_str:>9} "
            f"{baseline_str:>9} {delta_str:>9} {pct_str:>9}"
        )
    else:
        print(
            f"{'MACRO AVG':<28} {teacher_str:>9} {student_str:>9} "
            f"{delta_str:>9} {pct_str:>9}"
        )


def save_eval_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
