from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mteb
import torch
from sentence_transformers import SentenceTransformer


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

  model = SentenceTransformer(
    str(model_path),
    model_kwargs={"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
    trust_remote_code=True,
  )

  if prompt_name and getattr(model, "prompts", None):
    prompts = dict(model.prompts)
    if prompt_name in prompts:
      sts_instruction = prompts[prompt_name]
      for task_name in task_names:
        prompts[task_name] = sts_instruction
      prompts["STS"] = sts_instruction
      model.prompts = prompts

  evaluation = mteb.MTEB(tasks=mteb_tasks)
  results = evaluation.run(
    model,
    output_folder=str(output_dir) if output_dir else None,
    encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
  )

  summary: dict[str, Any] = {"model_path": str(model_path), "tasks": {}}
  for result in results:
    task_name = result.task_name
    main_score = None
    if result.scores:
      split_scores = result.scores.get("test") or result.scores.get("validation") or []
      if split_scores:
        main_score = split_scores[0].get("main_score")
    summary["tasks"][task_name] = {
      "main_score": main_score,
      "scores": result.scores,
    }
  return summary


def print_eval_summary(summary: dict[str, Any]) -> None:
  print(f"\nModel: {summary['model_path']}")
  for task_name, payload in summary.get("tasks", {}).items():
    score = payload.get("main_score")
    score_str = f"{score:.4f}" if score is not None else "n/a"
    print(f"  {task_name}: {score_str}")


def save_eval_summary(summary: dict[str, Any], output_path: Path) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
