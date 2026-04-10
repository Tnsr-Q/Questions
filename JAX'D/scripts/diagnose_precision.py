#!/usr/bin/env python3
"""One-shot precision capability diagnostic for CI/CD or pre-flight checks."""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.callbacks.precision_controller import PrecisionController


def main() -> None:
    ctrl = PrecisionController()
    print("🔍 Precision Capability Diagnostic")
    print(f"   State:          {ctrl.capabilities.state.value}")
    print(f"   Active Dtype:   {ctrl.capabilities.dtype}")
    print(f"   FP8 Available:  {ctrl.active}")
    print(f"   Fallback:       {ctrl.capabilities.fallback_reason}")
    print(f"   Torch Compile:  {ctrl.capabilities.compile_compatible}")

    x = torch.randn(1024, dtype=torch.float32, device=ctrl.device)
    q, s = ctrl.quantize(x)
    err = ctrl.track_quantization_error(x, q, s)
    print(f"   Quant Error:    {err:.2e} (NaN = FP8 inactive)")
    print(f"   Telemetry:      {ctrl.telemetry}")


if __name__ == "__main__":
    main()
