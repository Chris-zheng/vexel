"""
Vexel 2.0 Modal Serverless GPU Deployment Template
==================================================
Deploys Vexel's declarative LoRA training worker on Modal Serverless GPU (NVIDIA L4 / A10G):
- Per-second billing ($0 idle, scale-to-zero automatically)
- Fast ~8s cold starts with pre-baked PyTorch + timm image
- Hard container timeout (<= 500s) & Bearer token authentication
- Optional HMAC-SHA256 callback to Vexel `/api/v1/train/callback`

Deploy:
  modal secret create vexel-worker-secrets \
    VEXEL_GPU_TRAIN_TOKEN="<strong-random-token>" \
    VEXEL_GPU_CALLBACK_SECRET="<same-secret-as-vexel-server>" \
    VEXEL_CALLBACK_URL="https://<your-vexel-host>/api/v1/train/callback"

  modal deploy deploy/gpu-worker/modal_app.py
"""

from __future__ import annotations

import base64
import hmac
import os
from pathlib import Path
from typing import Any, Dict

import modal

app = modal.App("vexel-gpu-lora-worker")

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "timm>=1.0.9,<2.0.0",
        "onnx>=1.16.0,<2.0.0",
        "onnxruntime>=1.19.0,<2.0.0",
        "pillow>=10.4.0,<12.0.0",
        "numpy>=1.26.0,<3.0.0",
        "fastapi[standard]>=0.115.0",
    )
    .add_local_file("deploy/gpu-worker/train_worker.py", remote_path="/opt/vexel/train_worker.py")
)


@app.function(
    image=worker_image,
    gpu="L4",
    timeout=500,
    scaledown_window=60,
    max_containers=4,
    secrets=[modal.Secret.from_name("vexel-worker-secrets")],
)
@modal.fastapi_endpoint(method="POST")
def train_lora_endpoint(payload: Dict[str, Any], authorization: str = "") -> Dict[str, Any]:
    import sys

    sys.path.insert(0, "/opt/vexel")
    from train_worker import run_declarative_training, send_signed_callback

    expected_token = os.environ.get("VEXEL_GPU_TRAIN_TOKEN", "").strip()
    supplied = authorization.replace("Bearer ", "").strip()
    if not expected_token or not hmac.compare_digest(supplied, expected_token):
        return {"ok": False, "error": "Unauthorized GPU worker request."}

    manifest = payload.get("manifest")
    img_b64 = payload.get("imagePngBase64", "")
    mask_b64 = payload.get("maskPngBase64", "")
    if not isinstance(manifest, dict) or not img_b64 or not mask_b64:
        return {"ok": False, "error": "Missing declarative manifest, sanitized image, or mask."}

    work_dir = Path("/tmp/vexel_modal") / str(manifest.get("jobId", "job"))
    result = run_declarative_training(
        manifest=manifest,
        image_bytes=base64.b64decode(img_b64),
        mask_bytes=base64.b64decode(mask_b64),
        out_dir=work_dir,
    )

    # Estimate actual Modal L4 cost (~$0.80/hr)
    result["actualCostUsd"] = round(float(result["actualGpuSeconds"]) * (0.80 / 3600.0), 5)

    cb_url = os.environ.get("VEXEL_CALLBACK_URL", "").strip()
    cb_secret = os.environ.get("VEXEL_GPU_CALLBACK_SECRET", "").strip()
    if cb_url and cb_secret:
        send_signed_callback(cb_url, cb_secret, result)

    onnx_b64 = base64.b64encode(Path(result["onnxPath"]).read_bytes()).decode("ascii")
    return {"ok": True, "result": result, "onnxBase64": onnx_b64}
