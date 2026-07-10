from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.eval import resolve_miracl_subsets_for_suite
from harrier_distill.eval_parallel import (
    assign_gpus_to_models,
    parse_gpu_ids,
    probe_visible_cuda_device_count,
    resolve_gpu_ids,
    resolve_physical_cuda_id,
)
from harrier_distill.eval_progress import format_elapsed, is_quiet, log_eval, task_progress


class EvalProgressTests(unittest.TestCase):
    def test_format_elapsed(self) -> None:
        self.assertEqual(format_elapsed(12.3), "12.3s")
        self.assertTrue(format_elapsed(125).startswith("2m"))

    def test_is_quiet_env(self) -> None:
        with patch.dict("os.environ", {"EVAL_QUIET": "1"}):
            self.assertTrue(is_quiet())
        with patch.dict("os.environ", {"EVAL_QUIET": "0"}, clear=False):
            self.assertFalse(is_quiet(quiet=False))

    def test_log_eval_format(self) -> None:
        buf = io.StringIO()
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("EVAL_LOG_FILE", None)
            log_eval("hello", label="teacher", gpu=0, file=buf)
        line = buf.getvalue()
        self.assertIn("[eval]", line)
        self.assertIn("[teacher]", line)
        self.assertIn("[gpu=0]", line)
        self.assertIn("hello", line)

    def test_log_eval_mirrors_to_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "teacher.log"
            buf = io.StringIO()
            with patch.dict("os.environ", {"EVAL_LOG_FILE": str(log_path)}):
                log_eval("mirrored", label="teacher", gpu=1, file=buf)
            self.assertIn("mirrored", buf.getvalue())
            self.assertIn("mirrored", log_path.read_text(encoding="utf-8"))

    def test_task_progress_disabled_when_quiet(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("EVAL_LOG_FILE", None)
            items = list(task_progress([1, 2, 3], desc="x", quiet=True))
        self.assertEqual(items, [1, 2, 3])

    def test_task_progress_quiet_writes_tqdm_to_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "student.log"
            log_path.write_text("", encoding="utf-8")
            with patch.dict("os.environ", {"EVAL_LOG_FILE": str(log_path), "EVAL_QUIET": "1"}):
                items = list(task_progress([1, 2, 3], desc="encode", quiet=True))
            self.assertEqual(items, [1, 2, 3])
            # tqdm should have written progress into the per-model log
            self.assertIn("encode", log_path.read_text(encoding="utf-8"))


class EvalParallelTests(unittest.TestCase):
    def test_parse_gpu_ids(self) -> None:
        self.assertEqual(parse_gpu_ids("0,1,2"), [0, 1, 2])
        self.assertIsNone(parse_gpu_ids(None))
        self.assertEqual(parse_gpu_ids([3, 1]), [3, 1])

    def test_probe_visible_cuda_device_count_from_cvd(self) -> None:
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,2,5"}):
            self.assertEqual(probe_visible_cuda_device_count(), 3)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(probe_visible_cuda_device_count(), 0)

    def test_resolve_physical_cuda_id_respects_cvd(self) -> None:
        self.assertEqual(resolve_physical_cuda_id(1, visible="4,5,6"), 5)
        self.assertEqual(resolve_physical_cuda_id(0, visible=None), 0)

    def test_resolve_gpu_ids_uses_probe_not_torch(self) -> None:
        with patch(
            "harrier_distill.eval_parallel.probe_visible_cuda_device_count",
            return_value=3,
        ) as probe:
            ids = resolve_gpu_ids(parallel=True, n_models=3)
        probe.assert_called_once()
        self.assertEqual(ids, [0, 1, 2])

    def test_resolve_gpu_ids_sequential_when_not_parallel(self) -> None:
        self.assertEqual(resolve_gpu_ids(parallel=False, n_models=3, available=4), [])

    def test_resolve_gpu_ids_auto(self) -> None:
        self.assertEqual(resolve_gpu_ids(parallel=True, n_models=3, available=4), [0, 1, 2])
        self.assertEqual(resolve_gpu_ids(parallel=True, n_models=3, available=2), [0, 1])
        self.assertEqual(resolve_gpu_ids(parallel=True, n_models=3, available=0), [])

    def test_resolve_gpu_ids_explicit(self) -> None:
        self.assertEqual(
            resolve_gpu_ids(parallel=True, n_models=3, gpus=[0, 2], available=4),
            [0, 2],
        )
        with self.assertRaises(ValueError):
            resolve_gpu_ids(parallel=True, n_models=2, gpus=[0, 9], available=2)

    def test_assign_gpus_round_robin(self) -> None:
        models = [("teacher", "a"), ("student", "b"), ("baseline", "c")]
        assigned = assign_gpus_to_models(models, [0, 1])
        self.assertEqual(
            assigned,
            [("teacher", "a", 0), ("student", "b", 1), ("baseline", "c", 0)],
        )

    def test_run_parallel_jobs_uses_one_shot_workers(self) -> None:
        from harrier_distill.eval_parallel import run_parallel_jobs

        captured: dict[str, Any] = {}

        class FakePool:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, job):
                class Fut:
                    def result(self_inner):
                        return fn(job)

                return Fut()

        def worker(job):
            return job["label"], {"ok": True}

        with patch("harrier_distill.eval_parallel.ProcessPoolExecutor", FakePool):
            with patch("harrier_distill.eval_parallel.as_completed", lambda futs: list(futs)):
                out = run_parallel_jobs(
                    [{"label": "teacher", "gpu_id": 0}, {"label": "student", "gpu_id": 1}],
                    worker=worker,
                )
        self.assertEqual(captured.get("max_tasks_per_child"), 1)
        self.assertEqual(set(out), {"teacher", "student"})


class MiraclSuiteFilterTests(unittest.TestCase):
    def test_en_ko_filters_to_en_and_ko(self) -> None:
        languages = {
            "MIRACLRetrieval": [
                "eng-Latn",
                "ara-Arab",
                "kor-Kore",
                "jpn-Jpan",
            ]
        }
        self.assertEqual(
            resolve_miracl_subsets_for_suite("en_ko", languages),
            ["en", "ko"],
        )

    def test_all16_keeps_full_config_list(self) -> None:
        languages = {
            "MIRACLRetrieval": ["eng-Latn", "ara-Arab", "kor-Kore", "jpn-Jpan"]
        }
        self.assertEqual(
            resolve_miracl_subsets_for_suite("all16", languages),
            ["en", "ar", "ko", "ja"],
        )

    def test_miracl12_keeps_full_config_list(self) -> None:
        languages = {"MIRACLRetrieval": ["eng-Latn", "deu-Latn"]}
        self.assertEqual(
            resolve_miracl_subsets_for_suite("miracl12", languages),
            ["en", "de"],
        )

    def test_ko_suite_filters_to_ko(self) -> None:
        languages = {"MIRACLRetrieval": ["eng-Latn", "kor-Kore"]}
        self.assertEqual(resolve_miracl_subsets_for_suite("ko", languages), ["ko"])


if __name__ == "__main__":
    unittest.main()
