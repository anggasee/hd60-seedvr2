# HD60 Studio · SeedVR2 worker untuk RunPod Serverless
# GPU 24 GB. PyTorch 2.8 + CUDA 12.8: jalan di GPU lama (Ampere/Ada/Hopper) dan Blackwell.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Versi dikunci ke rilis yang sezaman dengan PyTorch 2.8 supaya diffusers/peft/transformers saling cocok
RUN printf "%s\n" \
      "torch==2.8.0" \
      "torchvision==0.23.0" \
      "diffusers==0.35.2" \
      "peft==0.17.1" \
      "transformers==4.56.2" \
      "accelerate==1.10.1" \
      "huggingface_hub==0.35.3" \
      "numpy==2.2.6" \
      "opencv-python==4.12.0.88" \
      > /tmp/constraints.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, Apache-2.0) – dipakai lewat CLI standalone
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git /app/seedvr2 \
    && pip install --no-cache-dir -r /app/seedvr2/requirements.txt -c /tmp/constraints.txt \
    && pip install --no-cache-dir runpod requests -c /tmp/constraints.txt

# Tes impor saat build: kalau ada paket yang tidak cocok, build langsung gagal di sini (bukan saat GPU menyala)
RUN python -c "import torch, torchvision, diffusers, peft, transformers, cv2, numpy; \
import diffusers.loaders.peft; \
print('OK torch', torch.__version__, '| diffusers', diffusers.__version__, '| peft', peft.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '| cv2', cv2.__version__)"

# Model 3B FP16 + VAE dimasukkan ke image supaya tidak download ulang tiap cold start.
RUN mkdir -p /models/SEEDVR2 && python -c "\
from huggingface_hub import hf_hub_download as d; \
[d(repo_id='numz/SeedVR2_comfyUI', filename=f, local_dir='/models/SEEDVR2') for f in ['seedvr2_ema_3b_fp16.safetensors','ema_vae_fp16.safetensors']]" \
    || echo 'model akan di-download saat runtime'

COPY handler.py /app/handler.py
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
