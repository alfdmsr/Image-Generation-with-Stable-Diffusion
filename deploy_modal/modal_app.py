"""Modal deployment definition for StudioAI (CPU Streamlit UI + Serverless T4 GPU Engine).

Run preview:  modal serve deploy_modal/modal_app.py
Deploy live:  modal deploy deploy_modal/modal_app.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "studioai-stable-diffusion"
MODEL_CACHE_PATH = "/models"
MINUTES = 60

LOCAL_DIR = Path(__file__).parent
APP_LOCAL_PATH = LOCAL_DIR / "app.py"
LOGIC_LOCAL_PATH = LOCAL_DIR / "logic.py"

if modal.is_local():
    if not APP_LOCAL_PATH.exists() or not LOGIC_LOCAL_PATH.exists():
        raise RuntimeError("app.py and logic.py must exist inside deploy_modal/")

# Persistent volume so Stable Diffusion checkpoints are downloaded only once
hf_cache_vol = modal.Volume.from_name("studioai-hf-cache", create_if_missing=True)

# Lightweight CPU image for Streamlit UI & dispatcher (fast startup, low cost)
ui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "streamlit==1.40.0",
        "streamlit-drawable-canvas==0.9.3",
        "Pillow>=10.0.0",
        "numpy>=1.26.0,<2.0.0",
        "pyarrow>=17.0.0,!=25.0.0,<26",
    )
    .add_local_file(APP_LOCAL_PATH, "/root/app.py")
    .add_local_file(LOGIC_LOCAL_PATH, "/root/logic.py")
)

# Dedicated GPU image for Stable Diffusion inference
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch>=2.5.0",
        "diffusers>=0.32.0",
        "transformers>=4.48.0",
        "accelerate>=1.3.0",
        "safetensors>=0.4.5",
        "Pillow>=10.0.0",
        "numpy>=1.26.0,<2.0.0",
    )
    .env(
        {
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_file(LOGIC_LOCAL_PATH, "/root/logic.py")
)

app = modal.App(name=APP_NAME)


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={MODEL_CACHE_PATH: hf_cache_vol},
    scaledown_window=2 * MINUTES,
    timeout=10 * MINUTES,
    max_containers=1,
)
class DiffusionEngine:
    """Serverless GPU engine that loads SD v1.5 pipelines once per container lifecycle."""

    @modal.enter()
    def load_models(self) -> None:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        import logic

        self.pipe_txt2img, self.pipe_inpaint = logic.load_pipelines(device="cuda")
        # Persist downloaded HuggingFace weights into the Modal Volume
        hf_cache_vol.commit()

    @modal.method()
    def generate(
        self,
        prompt: str,
        neg_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        num_images: int = 1,
        scheduler_name: str = "Euler A",
    ) -> list[bytes]:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        import logic

        images = logic.generate_images_local(
            pipe=self.pipe_txt2img,
            prompt=prompt,
            neg_prompt=neg_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            num_images=num_images,
            scheduler_name=scheduler_name,
        )
        logic.flush_gpu_memory()
        return [logic.pil_to_bytes(img, fmt="PNG") for img in images]

    @modal.method()
    def inpaint(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        strength: float = 0.85,
        neg_prompt: str = "",
        seed: int = 42,
        steps: int = 30,
        cfg: float = 7.5,
        scheduler_name: str = "Euler A",
    ) -> bytes:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        import io
        from PIL import Image
        import logic

        image = logic.bytes_to_pil(image_bytes)
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        result = logic.run_inpainting_local(
            pipe=self.pipe_inpaint,
            image=image,
            mask=mask,
            prompt=prompt,
            strength=strength,
            neg_prompt=neg_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            scheduler_name=scheduler_name,
        )
        logic.flush_gpu_memory()
        return logic.pil_to_bytes(result, fmt="PNG")

    @modal.method()
    def flush(self) -> str:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        import logic

        logic.flush_gpu_memory()
        return "GPU VRAM cache cleared."


@app.function(
    image=ui_image,
    timeout=10 * MINUTES,
)
def dispatch_gpu(action: str, payload: dict[str, Any]) -> Any:
    """Thin internal bridge so Streamlit subprocess can call DiffusionEngine in both serve & deploy."""
    engine = DiffusionEngine()
    if action == "generate":
        return engine.generate.remote(**payload)
    if action == "inpaint":
        return engine.inpaint.remote(**payload)
    if action == "flush":
        return engine.flush.remote()
    raise ValueError(f"Unknown action: {action}")


@app.function(
    image=ui_image,
    scaledown_window=5 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8000, startup_timeout=3 * MINUTES)
def serve_ui() -> None:
    """Serve StudioAI Streamlit UI on lightweight CPU while delegating inference to T4 GPU."""
    env = os.environ.copy()
    if dispatch_gpu.object_id:
        env["DISPATCH_FN_ID"] = dispatch_gpu.object_id

    target = shlex.quote("/root/app.py")
    cmd = (
        f"streamlit run {target} "
        "--server.port 8000 "
        "--server.address 0.0.0.0 "
        "--server.headless=true "
        "--server.enableCORS=false "
        "--server.enableXsrfProtection=false "
        "--browser.gatherUsageStats=false"
    )
    subprocess.Popen(cmd, shell=True, env=env)
