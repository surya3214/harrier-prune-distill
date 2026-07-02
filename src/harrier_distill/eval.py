from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mteb

from harrier_distill.model import load_sentence_transformer

STS_SUITES: dict[str, list[str]] = {
    "en": ["STSBenchmark"],
    "ko": ["KorSTS"],
    "multilingual": ["STSBenchmark", "KorSTS"],
    "extended": [
        "STSBenchmark",
        "KorSTS",
        "STS22.v2",
        "STSBenchmarkMultilingualSTS",
    ],
}


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


def evaluate_sts(
    model_path: str | Path,
    *,
    tasks: list[str] | None = None,
    prompt_name: str = "sts_query",
    batch_size: int = 64,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run MTEB STS tasks and return per-task Spearman scores."""
    task_names = tasks or ["STSBenchmark", "KorSTS"]
    try:
        mteb_tasks = mteb.get_tasks(tasks=task_names)
    except Exception:
        mteb_tasks = task_names

    model = load_sentence_transformer(model_path)
    _apply_sts_prompts(model, task_names, prompt_name)

    evaluation = mteb.MTEB(tasks=mteb_tasks)
    results = evaluation.run(
        model,
        output_folder=str(output_dir) if output_dir else None,
        encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
    )

    summary: dict[str, Any] = {"model_path": str(model_path), "tasks": {}}
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
) -> dict[str, Any]:
    """Evaluate teacher and student (and optional baseline) on the same STS suite."""
    task_names = get_tasks_for_suite(suite, tasks=tasks)
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
        model_mteb_dir = mteb_root / label if mteb_root else None
        summaries[label] = evaluate_sts(
            model_path,
            tasks=task_names,
            prompt_name=prompt_name,
            batch_size=batch_size,
            output_dir=model_mteb_dir,
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
