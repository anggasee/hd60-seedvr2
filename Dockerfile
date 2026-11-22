# HD60 Studio · SeedVR2 worker untuk RunPod Serverless
# GPU 24 GB. PyTorch 2.8 + CUDA 12.8: jalan di GPU lama (Ampere/Ada/Hopper) dan Blackwell.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.8 (cu128) dipasang lebih dulu, lalu dikunci supaya requirements tidak menggantinya
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    && printf "torch==2.8.0\ntorchvision==0.23.0\n" > /tmp/constraints.txt

# SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0) – dipakai lewat CLI standalone
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/seedvr2 \
    && pip install --no-cache-dir -r /app/seedvr2/requirements.txt -c /tmp/constraints.txt \
    && pip install --no-cache-dir runpod requests huggingface_hub -c /tmp/constraints.txt \
    && python -c "import torch, diffusers; print('torch', torch.__version__, 'arch', torch.cuda.get_arch_list(), 'diffusers', diffusers.__version__)"

# Model 3B FP16 + VAE dimasukkan ke image supaya tidak download ulang tiap cold start.
# Kalau langkah ini gagal, build tetap lanjut dan CLI akan download otomatis saat dipakai pertama.
RUN mkdir -p /models/SEEDVR2 && python -c "\
from huggingface_hub import hf_hub_download as d; \
[d(repo_id='numz/SeedVR2_comfyUI', filename=f, local_dir='/models/SEEDVR2') for f in ['seedvr2_ema_3b_fp16.safetensors','ema_vae_fp16.safetensors']]" \
    || echo 'model akan di-download saat runtime'

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
