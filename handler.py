"""
HD60 Studio · RunPod Serverless worker: SeedVR2 video upscale.

Input (job["input"]):
  video_url   : URL video sumber (wajib)
  result_url  : URL tujuan upload hasil (PUT, wajib)
  resolution  : sisi pendek target, default 1080 (video 9:16 -> 1080x1920)
  batch_size  : jumlah frame per batch, harus 4n+1, default 13
  model       : nama file DiT, default seedvr2_ema_3b_fp16.safetensors

Output:
  {"ok": true, "bytes": <ukuran hasil>, "seconds": <lama proses>}
"""
import glob
import os
import shutil
import subprocess
import tempfile
import threading
import time

import requests
import runpod

SEEDVR_DIR = "/app/seedvr2"
MODEL_DIR = "/models/SEEDVR2"
DEFAULT_MODEL = "seedvr2_ema_3b_fp16.safetensors"


def run(cmd, timeout=None):
    """Jalankan perintah, cetak outputnya LANGSUNG ke log RunPod, dan matikan kalau lewat batas waktu."""
    print("[hd60] $ " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines = []
    deadline = time.time() + timeout if timeout else None

    def watchdog():
        while p.poll() is None:
            if deadline and time.time() > deadline:
                print("[hd60] BATAS WAKTU TERLEWAT, proses dimatikan", flush=True)
                p.kill()
                return
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    for line in p.stdout:
        line = line.rstrip()
        lines.append(line)
        print(line, flush=True)
    p.wait()
    tail = "\n".join(lines[-40:])
    if deadline and time.time() > deadline and p.returncode != 0:
        raise RuntimeError(f"batas waktu {timeout}s terlewat | log: {tail[-600:]}")
    if p.returncode != 0:
        # baris error terakhir (mis. "RuntimeError: ...") ditaruh paling depan supaya terlihat di HD60 Studio
        ls = [l for l in lines if l.strip()]
        last = next((l for l in reversed(ls) if "Error" in l or "error:" in l), ls[-1] if ls else "")
        raise RuntimeError(f"{last.strip()[:300]} | log: {tail[-600:]}")
    return p


def has_audio(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(p.stdout.strip())


def handler(job):
    inp = job.get("input") or {}
    video_url = inp.get("video_url")
    result_url = inp.get("result_url")
    if not video_url or not result_url:
        return {"error": "video_url dan result_url wajib diisi"}

    resolution = int(inp.get("resolution") or 1080)
    batch = int(inp.get("batch_size") or 13)
    if (batch - 1) % 4 != 0:
        batch = 13
    model = inp.get("model") or DEFAULT_MODEL
    t0 = time.time()

    work = tempfile.mkdtemp(prefix="hd60_")
    try:
        src = os.path.join(work, "input.mp4")
        outdir = os.path.join(work, "out")
        os.makedirs(outdir, exist_ok=True)

        # 1. download video sumber
        runpod.serverless.progress_update(job, "download")
        with requests.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(src, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

        # 2. SeedVR2
        runpod.serverless.progress_update(job, "upscale")
        run([
            "python", os.path.join(SEEDVR_DIR, "inference_cli.py"), src,
            "--output", outdir,
            "--output_format", "mp4",
            "--video_backend", "ffmpeg",
            "--model_dir", MODEL_DIR,
            "--dit_model", model,
            "--resolution", str(resolution),
            "--batch_size", str(batch),
            "--color_correction", "lab",
            "--vae_encode_tiled",
            "--vae_decode_tiled",
        ], timeout=25 * 60)

        outs = sorted(glob.glob(os.path.join(outdir, "**", "*.mp4"), recursive=True), key=os.path.getmtime)
        if not outs:
            raise RuntimeError("SeedVR2 tidak menghasilkan file video")
        up = outs[-1]

        # 3. encode ulang ke H.264 yang aman untuk browser/TikTok + salin audio asli
        runpod.serverless.progress_update(job, "encode")
        final = os.path.join(work, "final.mp4")
        cmd = ["ffmpeg", "-y", "-i", up]
        if has_audio(src):
            cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += ["-map", "0:v:0"]
        cmd += [
            "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-maxrate", "20M", "-bufsize", "40M",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", final,
        ]
        run(cmd, timeout=1800)

        # 4. upload hasil
        runpod.serverless.progress_update(job, "upload")
        size = os.path.getsize(final)
        with open(final, "rb") as f:
            r = requests.put(result_url, data=f, headers={"Content-Type": "video/mp4", "Content-Length": str(size)}, timeout=600)
        if r.status_code >= 300:
            raise RuntimeError(f"upload hasil gagal: HTTP {r.status_code} {r.text[:200]}")

        return {"ok": True, "bytes": size, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:1500]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
