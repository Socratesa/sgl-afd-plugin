"""Routed-expert layout of ERNIE 5.5 Flash checkpoints.

SGLang reads MXFP4 routed experts from ``routed_experts_quant_method`` (and the
MiMo-V2 and DeepSeek-V4 layouts). The ERNIE 5.5 FP4 checkpoints declare them as
``"expert_dtype": "mxfp4"`` in their fp8 ``quantization_config``: int8-packed e2m1
weights with an E8M0 scale per 32 values, the layout ``Fp8MoEMethod`` creates for
``is_fp4_experts``. The FP8 checkpoints have no ``expert_dtype``: FP8 block experts.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def init(result, model_config, *args, **kwargs) -> None:
    """AFTER ``ModelConfig.__init__``."""
    from sglang.srt.configs.ernie55 import ERNIE5_5_MOE_ARCH
    from sglang.srt.configs.model_config import _hf_arch

    if (
        not model_config.is_fp4_experts
        and _hf_arch(model_config.hf_config) == ERNIE5_5_MOE_ARCH
        and str(model_config.hf_quant_config.get("expert_dtype") or "").lower()
        == "mxfp4"
    ):
        model_config.is_fp4_experts = True
        logger.info("Detected ERNIE 5.5 mxfp4 routed-expert layout.")
