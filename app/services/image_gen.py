"""Image generation service.

Priority order (first match wins):
  1. Pre-cached file on disk  → instant, zero GPU, perfect for demo
  2. Local SD + ControlNet   → LOCAL_SD=true in .env, RTX 3050 4GB friendly
  3. Replicate API           → REPLICATE_API_TOKEN in .env
  4. Mock (Unsplash URL)     → always works, zero deps

Pre-cache lives at  app/static/precached/{style}_{room_type}.jpg
Results are saved at app/static/results/{design_id}.jpg
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

from app.config import get_settings
from app.models.schemas import RedesignResponse

STATIC_DIR = Path(__file__).parent.parent / "static"
PRECACHED_DIR = STATIC_DIR / "precached"
RESULTS_DIR   = STATIC_DIR / "results"

MOCK_RESULTS: dict[str, str] = {
    "minimal":    "https://images.unsplash.com/photo-1616594039964-ae9021a400a0?w=800&q=80",
    "modern":     "https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=800&q=80",
    "scandi":     "https://images.unsplash.com/photo-1567016432779-094069958ea5?w=800&q=80",
    "wabi-sabi":  "https://images.unsplash.com/photo-1493663284031-b7e3aefcae8e?w=800&q=80",
    "industrial": "https://images.unsplash.com/photo-1505691938895-1758d7feb511?w=800&q=80",
}

STYLE_PROMPTS: dict[str, str] = {
    "minimal": (
        "minimalist interior design, clean lines, neutral palette, "
        "white walls, natural light, uncluttered space, Scandinavian influence"
    ),
    "modern": (
        "modern contemporary interior, bold accents, sleek furniture, "
        "open concept, neutral tones with contrasting elements, designer lighting"
    ),
    "scandi": (
        "Scandinavian interior design, hygge, warm wood textures, "
        "white and beige tones, cozy textiles, potted plants, functional furniture"
    ),
    "wabi-sabi": (
        "wabi-sabi interior, natural imperfect textures, earthy tones, "
        "ceramic vases, linen textiles, raw wood, soft ambient lighting"
    ),
    "industrial": (
        "industrial loft interior, exposed brick walls, metal and wood, "
        "Edison bulbs, dark tones, open ceiling, urban feel"
    ),
}

_NEGATIVE = (
    "lowres, watermark, banner, logo, text, deformed, blurry, out of focus, "
    "surreal, ugly, beginner, cartoonish, low quality"
)

# Lazy singleton — loaded once, stays in VRAM
_local_pipe = None


async def generate_redesign(
    image_bytes: bytes,
    style: str,
    room_type: str,
    sd_prompt: str | None = None,   # from DesignBrief when available
) -> RedesignResponse:
    settings = get_settings()
    design_id = str(uuid.uuid4())[:8]

    # 1 ── Pre-cached (instant, never fails)
    cached = _get_precached(style, room_type, settings)
    if cached:
        return RedesignResponse(
            design_id=design_id, original_url="",
            result_url=cached, style=style, is_mock=False,
        )

    prompt = sd_prompt or _build_prompt(style, room_type)

    # 2 ── Local SD + ControlNet (RTX 3050 4GB)
    if settings.local_sd:
        return await _local_response(design_id, image_bytes, style, prompt, settings)

    # 3 ── Replicate cloud
    if settings.use_replicate:
        return await _replicate_response(design_id, image_bytes, style, room_type, prompt, settings)

    # 4 ── Mock
    return _mock_response(design_id, style)


# ── Helpers ──────────────────────────────────────

def _get_precached(style: str, room_type: str, settings) -> str | None:
    """Return an absolute URL for a pre-generated image if it exists on disk."""
    path = PRECACHED_DIR / f"{style}_{room_type}.jpg"
    if path.exists():
        return f"{settings.public_url}/static/precached/{style}_{room_type}.jpg"
    return None


def _build_prompt(style: str, room_type: str) -> str:
    base = STYLE_PROMPTS.get(style, STYLE_PROMPTS["minimal"])
    return f"{room_type.replace('_', ' ')}, {base}, photorealistic, 8k, interior photography"


def _save_result(design_id: str, img, public_url: str = "") -> str:
    """Save PIL image to disk, return an absolute URL the browser can load."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{design_id}.jpg"
    img.save(path, "JPEG", quality=90)
    base = public_url or get_settings().public_url
    return f"{base}/static/results/{design_id}.jpg"


async def _local_response(
    design_id: str,
    image_bytes: bytes,
    style: str,
    prompt: str,
    settings,
) -> RedesignResponse:
    global _local_pipe
    try:
        import cv2
        import numpy as np
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        from PIL import Image as PILImage

        # ── Load pipeline once ──────────────────────────────────────────
        if _local_pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                settings.local_sd_controlnet,
                torch_dtype=torch.float16,
            )
            _local_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                settings.local_sd_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,      # speed
            )
            # 4GB VRAM tricks: offload unused layers to CPU RAM between steps
            _local_pipe.enable_model_cpu_offload()
            _local_pipe.enable_attention_slicing(1)

        # ── Prepare images ──────────────────────────────────────────────
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
        arr = np.array(img)

        # Canny edges as ControlNet conditioning — preserves room geometry
        gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        control_img = PILImage.fromarray(np.stack([edges] * 3, axis=-1))

        # ── Run inference ───────────────────────────────────────────────
        result = _local_pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE,
            image=img,
            control_image=control_img,
            strength=settings.local_sd_strength,
            guidance_scale=settings.local_sd_guidance,
            num_inference_steps=settings.local_sd_steps,
        ).images[0]

        result_url = _save_result(design_id, result, settings.public_url)
        return RedesignResponse(
            design_id=design_id, original_url="",
            result_url=result_url, style=style, is_mock=False,
        )

    except Exception as e:
        # GPU OOM or missing deps → fall through to mock gracefully
        print(f"[local_sd] error: {e}")
        return _mock_response(design_id, style)


def _mock_response(design_id: str, style: str) -> RedesignResponse:
    result_url = MOCK_RESULTS.get(style, MOCK_RESULTS["minimal"])
    return RedesignResponse(
        design_id=design_id, original_url="",
        result_url=result_url, style=style, is_mock=True,
    )


async def _replicate_response(
    design_id: str,
    image_bytes: bytes,
    style: str,
    room_type: str,
    prompt: str,
    settings,
) -> RedesignResponse:
    import os
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token
    b64 = base64.b64encode(image_bytes).decode()
    image_data_url = f"data:image/jpeg;base64,{b64}"

    output = await replicate.async_run(
        settings.replicate_interior_model,
        input={
            "image": image_data_url,
            "prompt": prompt,
            "guidance_scale": 15,
            "negative_prompt": _NEGATIVE,
            "num_inference_steps": 50,
            "strength": 0.8,
        },
    )

    result_url = str(output[0]) if isinstance(output, list) else str(output)
    return RedesignResponse(
        design_id=design_id, original_url=image_data_url[:100] + "...",
        result_url=result_url, style=style, is_mock=False,
    )
