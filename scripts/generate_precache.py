"""Pre-generate demo images for the hackathon presentation.

Run ONCE before the demo with LOCAL_SD=true in your .env:

    python scripts/generate_precache.py

Saves {style}_{room_type}.jpg into app/static/precached/.
During the demo these are served instantly — no GPU wait, no API call.

We pre-generate the 3 sample rooms × all 5 styles = 15 images.
Takes ~5-10 min on RTX 3050. Run the night before.
"""
import sys
import asyncio
from pathlib import Path

# Make sure we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from PIL import Image
import io

from app.services.image_gen import _local_pipe, _build_prompt, _save_result, PRECACHED_DIR, _NEGATIVE

SAMPLE_ROOMS = [
    ("bedroom",     "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=800&q=80"),
    ("living_room", "https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=800&q=80"),
    ("kitchen",     "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&q=80"),
]

STYLES = ["minimal", "modern", "scandi", "wabi-sabi", "industrial"]


async def precache_all():
    import cv2
    import numpy as np
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    from app.config import get_settings
    settings = get_settings()

    print("Loading SD pipeline…")
    controlnet = ControlNetModel.from_pretrained(
        settings.local_sd_controlnet, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        settings.local_sd_model, controlnet=controlnet,
        torch_dtype=torch.float16, safety_checker=None)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)
    print("Pipeline ready.\n")

    async with httpx.AsyncClient(timeout=30) as client:
        for room_type, url in SAMPLE_ROOMS:
            print(f"Downloading sample: {room_type}")
            resp = await client.get(url)
            image_bytes = resp.content

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
            arr = np.array(img)
            gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 200)
            control_img = Image.fromarray(np.stack([edges] * 3, axis=-1))

            for style in STYLES:
                out_path = PRECACHED_DIR / f"{style}_{room_type}.jpg"
                if out_path.exists():
                    print(f"  ✓ {style}_{room_type}.jpg  (already exists, skip)")
                    continue

                print(f"  Generating {style}_{room_type}…", end=" ", flush=True)
                prompt = _build_prompt(style, room_type)

                result = pipe(
                    prompt=prompt,
                    negative_prompt=_NEGATIVE,
                    image=img,
                    control_image=control_img,
                    strength=0.80,
                    guidance_scale=12.0,
                    num_inference_steps=25,
                ).images[0]

                result.save(out_path, "JPEG", quality=92)
                print(f"saved → {out_path.name}")

    print("\nAll done! Pre-cache is ready for the demo.")


if __name__ == "__main__":
    asyncio.run(precache_all())
