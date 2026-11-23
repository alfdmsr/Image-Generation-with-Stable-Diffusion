"""Core image processing helpers, Diffusers GPU loaders, and Modal client bridge for StudioAI."""

from __future__ import annotations

import gc
import io
import os
from typing import Any

from PIL import Image, ImageFilter

APP_NAME = "studioai-stable-diffusion"
TXT2IMG_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
INPAINT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"


# ---------------------------------------------------------------------------
# Pure Pillow Helpers (safe to import inside lightweight CPU UI container)
# ---------------------------------------------------------------------------


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL Image into bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    """Deserialize image bytes back into a PIL Image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def fit_dimensions(
    width: int,
    height: int,
    max_side: int = 768,
    min_side: int = 512,
) -> tuple[int, int]:
    """Scale (width, height) to fit within [min_side, max_side] as multiples of 8."""
    longest = max(width, height)
    scale = 1.0
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min_side / longest

    new_w = max(64, int(round(width * scale)))
    new_h = max(64, int(round(height * scale)))

    # Stable Diffusion requires dimensions divisible by 8
    new_w -= new_w % 8
    new_h -= new_h % 8
    return max(64, new_w), max(64, new_h)


def prepare_outpainting(
    image: Image.Image,
    expand_pixels: int = 128,
) -> tuple[Image.Image, Image.Image]:
    """Prepare expanded canvas with Gaussian-blurred border and binary mask for Zoom Out.

    White (255) = new border area to generate, Black (0) = original image to keep.
    """
    image = image.convert("RGB")
    w, h = image.size

    new_w = w + 2 * int(expand_pixels)
    new_h = h + 2 * int(expand_pixels)

    # Ensure dimensions are multiples of 8
    new_w -= new_w % 8
    new_h -= new_h % 8

    # Blurred background filler
    bg = image.resize((new_w, new_h), resample=Image.BICUBIC)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=50))

    canvas = bg.copy()
    paste_x = (new_w - w) // 2
    paste_y = (new_h - h) // 2
    canvas.paste(image, (paste_x, paste_y))

    # Mask: white border, black center
    mask = Image.new("L", (new_w, new_h), 255)
    inner_box = Image.new("L", (w, h), 0)
    mask.paste(inner_box, (paste_x, paste_y))

    return canvas, mask


# ---------------------------------------------------------------------------
# GPU / Diffusers Helpers (imported lazily so CPU container needs no torch)
# ---------------------------------------------------------------------------


def flush_gpu_memory() -> None:
    """Release unused Python & CUDA VRAM memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def set_scheduler(pipe: Any, scheduler_name: str) -> Any:
    """Swap scheduler on a Diffusers pipeline without reloading weights."""
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
    )

    if scheduler_name == "Euler A":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def load_pipelines(device: str | None = None):
    """Load Stable Diffusion v1.5 Text2Img and Inpainting pipelines."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading Stable Diffusion pipelines to {device} ({dtype})...")

    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        TXT2IMG_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe_txt2img.enable_attention_slicing()

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe_inpaint.enable_attention_slicing()

    return pipe_txt2img, pipe_inpaint


def generate_images_local(
    pipe: Any,
    prompt: str,
    neg_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    num_images: int = 1,
    scheduler_name: str = "Euler A",
) -> list[Image.Image]:
    """Run text-to-image generation on a loaded pipeline."""
    import torch

    pipe = set_scheduler(pipe, scheduler_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            generator=generator,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            num_images_per_prompt=int(num_images),
        )

    return list(result.images)


def run_inpainting_local(
    pipe: Any,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float = 0.85,
    neg_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 7.5,
    scheduler_name: str = "Euler A",
) -> Image.Image:
    """Run inpainting/outpainting on a loaded StableDiffusionInpaintPipeline."""
    import torch

    pipe = set_scheduler(pipe, scheduler_name)

    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    target_w, target_h = fit_dimensions(image.width, image.height, max_side=768, min_side=512)
    image = image.resize((target_w, target_h), Image.LANCZOS)
    mask = mask.resize((target_w, target_h), Image.NEAREST)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt or None,
            image=image,
            mask_image=mask,
            width=target_w,
            height=target_h,
            generator=generator,
            strength=float(strength),
            guidance_scale=float(cfg),
            num_inference_steps=int(steps),
        )

    return result.images[0]


# ---------------------------------------------------------------------------
# Modal Remote Bridge (called by Streamlit UI running on CPU)
# ---------------------------------------------------------------------------


def _dispatch_remote(action: str, payload: dict[str, Any]) -> Any:
    """Invoke GPU backend on Modal.

    Works seamlessly for both `modal serve` (ephemeral via DISPATCH_FN_ID)
    and `modal deploy` (deployed app lookup fallback).
    """
    import modal

    dispatch_fn_id = os.environ.get("DISPATCH_FN_ID")
    if dispatch_fn_id:
        fn = modal.Function.from_id(dispatch_fn_id)
        return fn.remote(action, payload)

    # Fallback when running app.py locally against an already deployed Modal app
    engine = modal.Cls.from_name(APP_NAME, "DiffusionEngine")()
    if action == "generate":
        return engine.generate.remote(**payload)
    if action == "inpaint":
        return engine.inpaint.remote(**payload)
    if action == "flush":
        return engine.flush.remote()
    raise ValueError(f"Unknown action: {action}")


def remote_generate(
    prompt: str,
    neg_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    num_images: int = 1,
    scheduler_name: str = "Euler A",
) -> list[Image.Image]:
    """Generate images remotely on Modal GPU."""
    raw_images: list[bytes] = _dispatch_remote(
        "generate",
        {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "num_images": int(num_images),
            "scheduler_name": scheduler_name,
        },
    )
    return [bytes_to_pil(img_bytes) for img_bytes in raw_images]


def remote_inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    strength: float = 0.85,
    neg_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 7.5,
    scheduler_name: str = "Euler A",
) -> Image.Image:
    """Run inpainting or outpainting remotely on Modal GPU."""
    raw_result: bytes = _dispatch_remote(
        "inpaint",
        {
            "image_bytes": pil_to_bytes(image.convert("RGB"), fmt="PNG"),
            "mask_bytes": pil_to_bytes(mask.convert("L"), fmt="PNG"),
            "prompt": prompt,
            "strength": float(strength),
            "neg_prompt": neg_prompt,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "scheduler_name": scheduler_name,
        },
    )
    return bytes_to_pil(raw_result)


def remote_flush_memory() -> str:
    """Flush VRAM on the remote Modal GPU container."""
    return str(_dispatch_remote("flush", {}))
