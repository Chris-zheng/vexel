# Vexel 2.0 Isolated GPU LoRA Training Worker Pack

This directory contains production-ready deployment templates for Vexel's declarative few-shot GPU LoRA fine-tuning pipeline:

- `Dockerfile`: Non-root (`uid 10001`), CUDA 12.4 + PyTorch 2.5 + `timm` + ONNX runtime container; runs air-gapped (`--network=none`).

> **Status (2026-11-27):** this pack has not been run on a GPU yet. The encoders are created with `pretrained=False`: no pretrained weights are downloaded or baked in, so a LoRA run starts from random encoder weights and will not give a useful model until pretrained weights are provided.
- `train_worker.py`: Declarative allowlist trainer (`dinov2_vit_s14`, `convnext_tiny`, `sam_vit_b_lora`, `unet_resnet34`) with frozen encoder + injected LoRA adapters, Focal+Dice / CE+Dice / Tversky loss, BF16/FP16 AMP, wall-clock watchdog, ONNX export (`<= 50 MB`), and HMAC-SHA256 signed callback reporting.
- `modal_app.py`: Ready-to-deploy **Modal Serverless GPU (NVIDIA L4)** endpoint with scale-to-zero (`$0` idle) and Bearer authentication.
- `runpod_handler.py`: Ready-to-deploy **RunPod Serverless (RTX 4090 / L40S)** handler.

## Quick Start

### Option A: Local Workstation (Air-Gapped Docker `--network=none`)
```bash
docker build -t vexel-gpu-worker ./deploy/gpu-worker

# Run using the pure-pixel sanitized dataset bundle exported by Vexel Model Hub:
docker run --rm --gpus all --network=none --read-only \
  --tmpfs /tmp/vexel_work:rw,size=2g \
  -v "$(pwd)/bundle:/input:ro" \
  -v "$(pwd)/output:/tmp/vexel_work/output:rw" \
  vexel-gpu-worker \
  --manifest /input/manifest.json \
  --image /input/image.png \
  --mask /input/mask.png \
  --out-dir /tmp/vexel_work/output
```

### Option B: Modal Serverless GPU (Recommended Elastic Cloud)
```bash
modal secret create vexel-worker-secrets \
  VEXEL_GPU_TRAIN_TOKEN="<strong-random-token>" \
  VEXEL_GPU_CALLBACK_SECRET="<shared-hmac-secret>" \
  VEXEL_CALLBACK_URL="https://<your-vexel-domain>/api/v1/train/callback"

modal deploy deploy/gpu-worker/modal_app.py
```
Then set `VEXEL_GPU_TRAIN_ENDPOINT`, `VEXEL_GPU_TRAIN_TOKEN`, and `VEXEL_GPU_CALLBACK_SECRET` in Vexel's server environment (Settings → Secrets). Never expose these tokens to the browser.
