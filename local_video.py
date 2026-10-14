#!/usr/bin/env python3
"""Local Wan 2.1 video generation for Apple Silicon. No hosted inference API."""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NEGATIVE = "blurry, low quality, distorted, text, subtitles, watermark, static image"
PRESETS = {
    "preview": {"width": 512, "height": 288, "frames": 33, "steps": 20},
    "480p": {"width": 832, "height": 480, "frames": 81, "steps": 50},
}


def identity(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def run_lock(output):
    import fcntl
    with (output / ".run.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another generation is using this output folder.") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt")
    source.add_argument("--file", type=Path)
    parser.add_argument("--preset", choices=PRESETS, default="preview")
    parser.add_argument("--portrait", action="store_true")
    parser.add_argument("--frames", type=int, help="4n+1 frames, such as 17, 33, 49 or 81")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative", default=NEGATIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "videos")
    parser.add_argument("--offline", action="store_true", help="Use only downloaded weights")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check installation and Apple GPU; no downloads")
    mode.add_argument("--download-only", action="store_true", help="Download weights without generation")
    mode.add_argument("--dry-run", action="store_true", help="Show settings without imports or downloads")
    args = parser.parse_args(argv)
    settings = dict(PRESETS[args.preset])
    if args.portrait:
        settings["width"], settings["height"] = settings["height"], settings["width"]
    for name in ("frames", "steps"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if settings["frames"] < 5 or (settings["frames"] - 1) % 4:
        parser.error("--frames must be 4n+1 and at least 5")
    if not 1 <= settings["steps"] <= 100:
        parser.error("--steps must be between 1 and 100")
    if not 0 <= args.seed < 2 ** 63:
        parser.error("--seed must be between 0 and 2^63-1")
    prompts = []
    if args.file:
        prompts = [line.strip() for line in args.file.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    elif args.prompt:
        prompts = [args.prompt.strip()] if args.prompt.strip() else []
    if not prompts and not (args.check or args.download_only):
        parser.error("Provide --prompt or --file with at least one description")
    return args, settings, prompts


def check_environment():
    print(f"System: {platform.system()} {platform.machine()}; Python: {platform.python_version()}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Use an Apple Silicon Mac and an arm64 Python installation.")
    packages = ["torch", "diffusers", "transformers", "accelerate", "safetensors", "imageio_ffmpeg"]
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Run bash setup.sh first. Missing: " + ", ".join(missing))
    import torch
    print(f"PyTorch: {torch.__version__}; Apple GPU available: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable. Run from macOS Terminal outside a restricted sandbox.")
    # A small allocation catches driver/backend problems before the large model download.
    test = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
    (test @ test).cpu()
    del test
    torch.mps.empty_cache()
    print(f"Disk space available: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB")
    return torch


def get_snapshot(offline):
    from huggingface_hub import snapshot_download
    print("Preparing local model weights (first download is about 29 GB)...", flush=True)
    return snapshot_download(
        MODEL, local_files_only=offline,
        allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                        "text_encoder/*.json", "text_encoder/*.safetensors",
                        "transformer/*.json", "transformer/*.safetensors",
                        "vae/*.json", "vae/*.safetensors"],
    )


def embedding_path(cache, snapshot, prompt, negative):
    key = identity({"snapshot": snapshot, "prompt": prompt, "negative": negative, "length": 512})
    return cache / f"{key}.safetensors"


def encode_prompts(torch, snapshot, prompts, negative, cache):
    from diffusers import WanPipeline, UniPCMultistepScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel
    from safetensors.torch import load_file, save_file

    pending = []
    for prompt in dict.fromkeys(prompts):
        path = embedding_path(cache, snapshot, prompt, negative)
        if path.exists():
            try:
                existing = load_file(str(path))
                if all(existing[name].shape == (1, 512, 4096) for name in ("positive", "negative")):
                    del existing
                    continue
            except (OSError, ValueError, KeyError):
                pass
        pending.append(prompt)
    if not pending:
        return
    print("Encoding descriptions; the text model is released before video generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer", local_files_only=True)
    for prompt in pending + [negative]:
        if len(tokenizer(prompt, truncation=False).input_ids) > 512:
            raise ValueError("A description exceeds the model's 512-token context. Shorten that description.")
    encoder = UMT5EncoderModel.from_pretrained(
        snapshot, subfolder="text_encoder", local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, use_safetensors=True,
    ).eval().to("mps")
    text_pipe = WanPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None,
                            scheduler=UniPCMultistepScheduler(), transformer=None)
    with torch.inference_mode():
        for index, prompt in enumerate(pending, 1):
            print(f"Encoding {index}/{len(pending)}", flush=True)
            positive, neg = text_pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative, device=torch.device("mps"),
                max_sequence_length=512, dtype=torch.bfloat16,
            )
            if not torch.isfinite(positive).all().item() or not torch.isfinite(neg).all().item():
                raise RuntimeError("Non-finite text embeddings. No video submitted or saved.")
            path = embedding_path(cache, snapshot, prompt, negative)
            temporary = path.with_suffix(".tmp")
            save_file({"positive": positive.cpu().contiguous(), "negative": neg.cpu().contiguous()}, str(temporary))
            temporary.replace(path)
            del positive, neg
    del text_pipe, encoder, tokenizer
    gc.collect()
    torch.mps.empty_cache()


def video_spec(prompt, negative, settings, seed):
    return {"model": MODEL, "prompt": prompt, "negative": negative,
            **settings, "seed": seed, "fps": 16, "guidance": 5.0, "flow_shift": 3.0}


def already_done(video, metadata, spec):
    if not video.exists() or not metadata.exists():
        return False
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return record.get("settings") == spec and record.get("sha256") == file_hash(video)
    except (OSError, ValueError):
        return False


def generate(torch, snapshot, jobs, cache):
    from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from safetensors.torch import load_file
    import numpy as np

    vae = AutoencoderKLWan.from_pretrained(snapshot, subfolder="vae", local_files_only=True,
                                          torch_dtype=torch.float32, use_safetensors=True)
    vae.enable_tiling()
    pipe = WanPipeline.from_pretrained(
        snapshot, text_encoder=None, tokenizer=None, vae=vae, local_files_only=True,
        torch_dtype=torch.bfloat16, use_safetensors=True,
    ).to("mps")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    with torch.inference_mode():
        for index, (spec, video, metadata) in enumerate(jobs, 1):
            print(f"Generating {index}/{len(jobs)}: {video.name}", flush=True)
            embeddings = load_file(str(embedding_path(cache, snapshot, spec["prompt"], spec["negative"])))
            started = time.monotonic()
            frames = pipe(
                prompt_embeds=embeddings["positive"].to("mps"),
                negative_prompt_embeds=embeddings["negative"].to("mps"),
                height=spec["height"], width=spec["width"], num_frames=spec["frames"],
                num_inference_steps=spec["steps"], guidance_scale=spec["guidance"],
                generator=torch.Generator(device="cpu").manual_seed(spec["seed"]),
                max_sequence_length=512, output_type="np",
            ).frames[0]
            if not np.isfinite(frames).all():
                raise RuntimeError("Generation produced invalid pixels. No output video was saved.")
            temporary = video.with_name(video.stem + ".partial.mp4")
            try:
                export_to_video(frames, str(temporary), fps=16)
                if not temporary.exists() or temporary.stat().st_size == 0:
                    raise RuntimeError("Video export did not produce a file.")
                temporary.replace(video)
            finally:
                temporary.unlink(missing_ok=True)
            atomic_json(metadata, {"settings": spec, "sha256": file_hash(video),
                                   "snapshot": snapshot, "elapsed_seconds": round(time.monotonic() - started, 2)})
            print(f"Saved: {video}", flush=True)
            del frames, embeddings
            gc.collect()
            torch.mps.empty_cache()


def main(argv=None):
    args, settings, prompts = options(argv)
    if args.dry_run:
        print(json.dumps({"model": MODEL, "settings": settings, "fps": 16,
                          "duration_seconds": settings["frames"] / 16, "prompts": prompts},
                         ensure_ascii=False, indent=2))
        return 0
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    torch = check_environment()
    if args.check:
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        jobs = []
        for index, prompt in enumerate(prompts):
            spec = video_spec(prompt, args.negative, settings, (args.seed + index) % 2**63)
            stem = f"video_{index + 1:03d}_{identity(spec)[:12]}"
            video, metadata = output / f"{stem}.mp4", output / f"{stem}.json"
            if already_done(video, metadata, spec):
                print(f"Already saved: {video}")
            else:
                jobs.append((spec, video, metadata))
        if not jobs and not args.download_only:
            print("All requested videos are already saved.")
            return 0
        snapshot = get_snapshot(args.offline)
        if args.download_only:
            print(f"Model ready: {snapshot}")
            return 0
        cache = output / ".embeddings"
        cache.mkdir(exist_ok=True)
        encode_prompts(torch, snapshot, [spec["prompt"] for spec, _, _ in jobs], args.negative, cache)
        generate(torch, snapshot, jobs, cache)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped. Completed videos are kept; rerun the same command to continue.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        text = str(exc)
        if "out of memory" in text.lower():
            text += "\nClose other apps and retry with --preset preview --frames 17."
        print(f"Error: {text}", file=sys.stderr)
        sys.exit(1)
