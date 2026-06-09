import uuid
import base64
from io import BytesIO

from app.config import get_settings
from app.models.schemas import RedesignResponse

# Curated mock result images per style (Unsplash)
MOCK_RESULTS: dict[str, str] = {
    "minimal": "https://images.unsplash.com/photo-1616594039964-ae9021a400a0?w=800&q=80",
    "modern":  "https://images.unsplash.com/photo-1618221195710-dd6b41faaea6?w=800&q=80",
    "scandi":  "https://images.unsplash.com/photo-1567016432779-094069958ea5?w=800&q=80",
    "wabi-sabi": "https://images.unsplash.com/photo-1493663284031-b7e3aefcae8e?w=800&q=80",
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


async def generate_redesign(
    image_bytes: bytes,
    style: str,
    room_type: str,
) -> RedesignResponse:
    settings = get_settings()
    design_id = str(uuid.uuid4())[:8]

    # Store original as data URL for response (simplified)
    original_b64 = base64.b64encode(image_bytes).decode()
    original_url = f"data:image/jpeg;base64,{original_b64[:100]}..."  # truncated

    if not settings.use_replicate:
        return _mock_response(design_id, style)

    return await _replicate_response(design_id, image_bytes, style, room_type, settings)


def _mock_response(design_id: str, style: str) -> RedesignResponse:
    result_url = MOCK_RESULTS.get(style, MOCK_RESULTS["minimal"])
    return RedesignResponse(
        design_id=design_id,
        original_url="",
        result_url=result_url,
        style=style,
        is_mock=True,
    )


async def _replicate_response(
    design_id: str,
    image_bytes: bytes,
    style: str,
    room_type: str,
    settings,
) -> RedesignResponse:
    import replicate
    import os

    os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

    # Convert image to base64 data URL for Replicate
    b64 = base64.b64encode(image_bytes).decode()
    image_data_url = f"data:image/jpeg;base64,{b64}"

    prompt = (
        f"{room_type}, {STYLE_PROMPTS.get(style, STYLE_PROMPTS['minimal'])}, "
        "high quality, photorealistic, 8k, interior photography"
    )

    output = await replicate.async_run(
        settings.replicate_interior_model,
        input={
            "image": image_data_url,
            "prompt": prompt,
            "guidance_scale": 15,
            "negative_prompt": (
                "lowres, watermark, banner, logo, watermark, contactinfo, "
                "text, deformed, blurry, out of focus, surreal, ugly, beginner"
            ),
            "num_inference_steps": 50,
            "strength": 0.8,
        },
    )

    result_url = str(output[0]) if isinstance(output, list) else str(output)

    return RedesignResponse(
        design_id=design_id,
        original_url=image_data_url[:100] + "...",
        result_url=result_url,
        style=style,
        is_mock=False,
    )
