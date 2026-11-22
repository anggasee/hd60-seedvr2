# HD60 Studio · SeedVR2 worker untuk RunPod Serverless
# GPU yang disarankan: 24 GB (RTX 4090 / L4 / A5000 / RTX 3090)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0) – dipakai lewat CLI standalone
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/seedvr2 \
    && pip install --no-cache-dir -r /app/seedvr2/requirements.txt \
    && pip install --no-cache-dir runpod requests huggingface_hub hf_transfer

# Model 3B FP16 + VAE dimasukkan ke image supaya tidak download ulang tiap cold start.
# Kalau langkah ini gagal, build tetap lanjut dan CLI akan download otomatis saat dipakai pertama.
RUN mkdir -p /models/SEEDVR2 && python -c "\
from huggingface_hub import hf_hub_download as d; \
[d(repo_id='numz/SeedVR2_comfyUI', filename=f, local_dir='/models/SEEDVR2') for f in ['seedvr2_ema_3b_fp16.safetensors','ema_vae_fp16.safetensors']]" \
    || echo 'model akan di-download saat runtime'

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
