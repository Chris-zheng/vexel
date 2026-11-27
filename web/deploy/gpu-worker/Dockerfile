# Vexel GPU LoRA training worker (see README.md). CUDA 12.4 runtime with PyTorch 2.5.
#   docker build -t vexel-gpu-worker ./deploy/gpu-worker
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "timm>=1.0.9,<2.0.0" "onnx>=1.16.0,<2.0.0" "onnxruntime>=1.19.0,<2.0.0" \
        "pillow>=10.4.0,<12.0.0" "numpy>=1.26.0,<3.0.0"

# non-root user; the worker only writes under /tmp/vexel_work (mount it, or run with --tmpfs)
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin vexel && mkdir -p /opt/vexel /tmp/vexel_work && chown vexel /tmp/vexel_work
COPY train_worker.py /opt/vexel/train_worker.py
USER 10001
WORKDIR /tmp/vexel_work
ENTRYPOINT ["python", "/opt/vexel/train_worker.py"]
