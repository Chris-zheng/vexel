"""
Vexel 2.0 RunPod Serverless GPU Worker Handler
==============================================
Drop-in handler for RunPod Serverless endpoints (RTX 4090 / L40S) using the exact same
declarative allowlist trainer (`train_worker.py`) and HMAC-SHA256 callback reporter.

Container Start Command:
  python /opt/vexel/app/runpod_handler.py
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict

from train_worker import run_declarative_training, send_signed_callback


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    job_input = event.get("input") or {}
    manifest = job_input.get("manifest")
    img_b64 = job_input.get("imagePngBase64", "")
    mask_b64 = job_input.get("maskPngBase64", "")

    if not isinstance(manifest, dict) or not img_b64 or not mask_b64:
        return {"ok": False, "error": "Missing declarative manifest, imagePngBase64, or maskPngBase64."}

    out_dir = Path("/tmp/vexel_work") / str(manifest.get("jobId", "runpod_job"))
    result = run_declarative_training(
        manifest=manifest,
        image_bytes=base64.b64decode(img_b64),
        mask_bytes=base64.b64decode(mask_b64),
        out_dir=out_dir,
    )
    # Approximate RunPod RTX 4090 serverless rate (~$0.69/hr)
    result["actualCostUsd"] = round(float(result["actualGpuSeconds"]) * (0.69 / 3600.0), 5)

    cb_url = os.environ.get("VEXEL_CALLBACK_URL", "").strip()
    cb_secret = os.environ.get("VEXEL_GPU_CALLBACK_SECRET", "").strip()
    if cb_url and cb_secret:
        send_signed_callback(cb_url, cb_secret, result)

    onnx_b64 = base64.b64encode(Path(result["onnxPath"]).read_bytes()).decode("ascii")
    return {
        "ok": True,
        "result": result,
        "onnxBase64": onnx_b64,
    }


if __name__ == "__main__":
    import runpod  # type: ignore

    runpod.serverless.start({"handler": handler})
