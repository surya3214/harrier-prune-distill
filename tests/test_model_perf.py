from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.model import (
    ATTN_IMPLEMENTATION_ALIASES,
    build_adamw,
    enable_gradient_checkpointing,
    get_model_kwargs,
    get_transformer_auto_model,
    maybe_enable_tf32,
    resolve_attn_implementation,
)


class AttnImplementationTests(unittest.TestCase):
    def test_resolve_aliases(self) -> None:
        self.assertEqual(resolve_attn_implementation("sdpa"), "sdpa")
        self.assertEqual(resolve_attn_implementation("flash_attention_2"), "flash_attention_2")
        self.assertEqual(resolve_attn_implementation("none"), "eager")
        self.assertEqual(resolve_attn_implementation("eager"), "eager")
        self.assertEqual(resolve_attn_implementation("SDPA"), "sdpa")
        self.assertIsNone(resolve_attn_implementation(None))
        self.assertIsNone(resolve_attn_implementation(""))

    def test_resolve_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            resolve_attn_implementation("xformers")

    def test_get_model_kwargs_includes_attn(self) -> None:
        kwargs = get_model_kwargs(prefer_bf16=False, attn_implementation="none")
        self.assertEqual(kwargs["attn_implementation"], "eager")
        self.assertTrue("torch_dtype" in kwargs or "dtype" in kwargs)

    def test_get_model_kwargs_omits_attn_when_unset(self) -> None:
        kwargs = get_model_kwargs(prefer_bf16=False, attn_implementation=None)
        self.assertNotIn("attn_implementation", kwargs)

    def test_alias_table_covers_config_values(self) -> None:
        for key in ("sdpa", "flash_attention_2", "none"):
            self.assertIn(key, ATTN_IMPLEMENTATION_ALIASES)


class GradientCheckpointingHelperTests(unittest.TestCase):
    def test_get_transformer_auto_model(self) -> None:
        auto_model = MagicMock(name="auto_model")
        st_model = [SimpleNamespace(auto_model=auto_model)]
        self.assertIs(get_transformer_auto_model(st_model), auto_model)

    def test_get_transformer_auto_model_missing(self) -> None:
        st_model = [SimpleNamespace()]
        with self.assertRaises(AttributeError):
            get_transformer_auto_model(st_model)

    def test_enable_gradient_checkpointing(self) -> None:
        auto_model = MagicMock()
        auto_model.config = SimpleNamespace(use_cache=True)
        st_model = [SimpleNamespace(auto_model=auto_model)]
        enable_gradient_checkpointing(st_model)
        auto_model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.assertFalse(auto_model.config.use_cache)


class Tf32AndAdamWTests(unittest.TestCase):
    def test_maybe_enable_tf32_disabled_flag(self) -> None:
        self.assertFalse(maybe_enable_tf32(enabled=False))

    def test_build_adamw_cpu_ignores_fused(self) -> None:
        import torch

        params = [torch.nn.Parameter(torch.zeros(4))]
        opt = build_adamw(params, lr=1e-3, weight_decay=0.01, fused=True, device_type="cpu")
        self.assertIsInstance(opt, torch.optim.AdamW)
        # fused flag is not set (or False) on CPU path
        fused = getattr(opt, "fused", False) or opt.defaults.get("fused", False)
        self.assertFalse(bool(fused))


if __name__ == "__main__":
    unittest.main()
