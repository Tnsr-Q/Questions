import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger("QUFT_Precision")


class PrecisionState(str, Enum):
    FP8_ACTIVE = "fp8_e4m3fn_active"
    BF16_FALLBACK = "bf16_fallback"
    FP32_FALLBACK = "fp32_fallback"
    UNAVAILABLE = "precision_unavailable"


@dataclass
class PrecisionCapabilities:
    state: PrecisionState
    dtype: torch.dtype
    fp8_available: bool = False
    fallback_reason: str = ""
    compile_compatible: bool = True


class PrecisionController:
    """Runtime precision negotiator with explicit capability detection."""

    def __init__(self, target_dtype_str: str = "float8_e4m3fn", device: Optional[torch.device] = None):
        self.target_dtype_str = target_dtype_str
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.error_buffer: list[float] = []
        self.capabilities = self._detect_capabilities()
        self.active = self.capabilities.state == PrecisionState.FP8_ACTIVE

        if not self.active:
            log.info(
                "🔍 Precision negotiated: %s (%s)",
                self.capabilities.state.value,
                self.capabilities.fallback_reason,
            )

    def _detect_capabilities(self) -> PrecisionCapabilities:
        if dist.is_available() and dist.is_initialized():
            payload: list[PrecisionCapabilities] = []
            if dist.get_rank() == 0:
                payload = [self._detect_local_capabilities()]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
        return self._detect_local_capabilities()

    def _detect_local_capabilities(self) -> PrecisionCapabilities:
        state = PrecisionState.FP32_FALLBACK
        dtype = torch.float32
        fp8_available = False
        reason = "default_fp32"
        compile_compat = True

        target_dtype = getattr(torch, self.target_dtype_str, None)
        if target_dtype is None:
            reason = f"PyTorch {torch.__version__} lacks {self.target_dtype_str}"
            compile_compat = False
            return PrecisionCapabilities(state, dtype, fp8_available, reason, compile_compat)

        if self.device.type != "cuda" or not torch.cuda.is_available():
            reason = "No CUDA device; FP8 requires GPU"
            return PrecisionCapabilities(state, dtype, fp8_available, reason, compile_compat)

        cap = torch.cuda.get_device_capability(self.device.index or 0)
        try:
            dummy = torch.empty((1, 1), dtype=target_dtype, device=self.device)
            _ = torch.matmul(dummy, dummy)
            fp8_available = True
        except RuntimeError as exc:
            reason = f"Device runtime FP8 failure: {str(exc)[:80]}"
            compile_compat = False

        if fp8_available:
            if cap >= (8, 0):
                state = PrecisionState.FP8_ACTIVE
                dtype = target_dtype
                if cap < (9, 0):
                    reason = f"Software FP8 on {cap[0]}.{cap[1]}; expect overhead vs native"
                else:
                    reason = ""
            elif torch.cuda.is_bf16_supported():
                state = PrecisionState.BF16_FALLBACK
                dtype = torch.bfloat16
                reason = f"Device {cap[0]}.{cap[1]} lacks FP8; BF16 fallback for stability"
            else:
                reason = f"Device {cap[0]}.{cap[1]} lacks FP8/BF16; FP32 fallback"

        return PrecisionCapabilities(state, dtype, fp8_available, reason, compile_compat)

    def quantize(self, tensor: torch.Tensor, name: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        del name
        if not self.active:
            return tensor.to(self.capabilities.dtype), torch.tensor(1.0, device=tensor.device)

        scale = torch.max(torch.abs(tensor)).detach() + 1e-8
        quantized = (tensor / scale).clamp(-448.0, 448.0).to(self.capabilities.dtype)
        return quantized, scale

    def dequantize(self, tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return tensor.to(torch.float32)
        return (tensor.to(torch.float32) * scale).to(torch.float32)

    def track_quantization_error(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> float:
        if not self.active:
            return float("nan")

        reconstructed = self.dequantize(quantized, scale)
        mse = torch.mean((original.float() - reconstructed) ** 2).item()
        self.error_buffer.append(mse)
        return mse

    @property
    def telemetry(self) -> dict[str, Any]:
        return {
            "precision/state": self.capabilities.state.value,
            "precision/active_dtype": str(self.capabilities.dtype),
            "precision/fp8_available": float(self.active),
            "precision/compile_compatible": float(self.capabilities.compile_compatible),
            "precision/quantization_error": (
                float(np.nanmean(self.error_buffer[-20:])) if self.error_buffer else float("nan")
            ),
            "precision/fallback_reason": self.capabilities.fallback_reason,
        }
