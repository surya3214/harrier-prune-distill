from __future__ import annotations

from typing import Any

MTEB_ENG_V2_BENCHMARK = "MTEB(eng, v2)"

MTEB_ENG_V2_STS_HF_SUBSETS: dict[str, list[str]] = {
    "STS17": ["en-en"],
    "STS22.v2": ["en"],
}

MTEB_ENG_V2_STS_TASK_NAMES: tuple[str, ...] = (
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STSBenchmark",
    "STS17",
    "STS22.v2",
)


def mteb_eng_v2_sts_task_names() -> list[str]:
    """Return STS task names from MTEB(eng, v2) in benchmark order."""
    return list(MTEB_ENG_V2_STS_TASK_NAMES)


def resolve_mteb_sts_task_objects(task_names: list[str]) -> list[Any]:
    """Resolve MTEB task objects with eval splits and hf_subsets matching MTEB(eng, v2)."""
    import mteb

    tasks = []
    for name in task_names:
        kwargs: dict[str, Any] = {"eval_splits": ["test"]}
        if name in MTEB_ENG_V2_STS_HF_SUBSETS:
            kwargs["hf_subsets"] = MTEB_ENG_V2_STS_HF_SUBSETS[name]
        tasks.append(mteb.get_task(name, **kwargs))
    return tasks
