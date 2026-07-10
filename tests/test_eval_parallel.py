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
    _probe_eval_worker,
    assign_gpus_to_models,
    parse_gpu_ids,
    probe_visible_cuda_device_count,
    resolve_gpu_ids,
    resolve_physical_cuda_id,
    run_parallel_jobs,
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

    def test_make_jsonable_strips_numpy(self) -> None:
        import numpy as np
        from harrier_distill.eval_parallel import make_jsonable

        payload = {"score": np.float32(0.5), "path": Path("/tmp/x")}
        out = make_jsonable(payload)
        self.assertEqual(out["score"], 0.5)
        self.assertEqual(out["path"], "/tmp/x")
        self.assertIsInstance(out["score"], float)

    def test_multi_gpu_subprocess_isolates_cvd(self) -> None:
        """Concurrent workers must each see only their own CUDA_VISIBLE_DEVICES."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            out = run_parallel_jobs(
                [
                    {
                        "label": "teacher",
                        "gpu_id": 0,
                        "cuda_visible_devices": "0",
                    },
                    {
                        "label": "student",
                        "gpu_id": 1,
                        "cuda_visible_devices": "1",
                    },
                ],
                worker=_probe_eval_worker,
                max_workers=2,
            )
        self.assertEqual(out["teacher"]["cvd"], "0")
        self.assertEqual(out["student"]["cvd"], "1")

    def test_launch_sets_cvd_in_child_env(self) -> None:
        from harrier_distill.eval_parallel import _launch_worker_subprocess

        captured: dict[str, Any] = {}

        class FakeProc:
            def poll(self):
                return 0

            def terminate(self):
                return None

            def kill(self):
                return None

            def wait(self, timeout=None):
                return 0

        def fake_popen(cmd, env=None, **kwargs):
            captured["env_cvd"] = env.get("CUDA_VISIBLE_DEVICES") if env else None
            captured["cmd"] = cmd
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("harrier_distill.eval_parallel.subprocess.Popen", side_effect=fake_popen):
                _launch_worker_subprocess(
                    {
                        "label": "teacher",
                        "gpu_id": 1,
                        "cuda_visible_devices": "5",
                        "worker_kind": "probe",
                    },
                    work_dir=Path(tmp),
                )
        self.assertEqual(captured["env_cvd"], "5")
        self.assertIn("harrier_distill.eval_cuda_worker", captured["cmd"])


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
