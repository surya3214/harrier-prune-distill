from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.eval import resolve_miracl_subsets_for_suite
from harrier_distill.eval_parallel import (
    assign_gpus_to_models,
    parse_gpu_ids,
    resolve_gpu_ids,
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
        log_eval("hello", label="teacher", gpu=0, file=buf)
        line = buf.getvalue()
        self.assertIn("[eval]", line)
        self.assertIn("[teacher]", line)
        self.assertIn("[gpu=0]", line)
        self.assertIn("hello", line)

    def test_task_progress_disabled_when_quiet(self) -> None:
        items = list(task_progress([1, 2, 3], desc="x", quiet=True))
        self.assertEqual(items, [1, 2, 3])


class EvalParallelTests(unittest.TestCase):
    def test_parse_gpu_ids(self) -> None:
        self.assertEqual(parse_gpu_ids("0,1,2"), [0, 1, 2])
        self.assertIsNone(parse_gpu_ids(None))
        self.assertEqual(parse_gpu_ids([3, 1]), [3, 1])

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
