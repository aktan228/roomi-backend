"""Pre-generate demo images for the hackathon presentation.

Runs the SAME generation path the app uses (image_gen.generate_redesign), so it
works with whatever backend you've configured:
  - LOCAL_SD=true        → local Stable Diffusion (GPU)
  - REPLICATE_API_TOKEN  → Replicate cloud (no GPU needed)
  - neither              → mock images (still gives a fallback set)

Usage (from roomi-backend/, with your .env configured):

    python scripts/generate_precache.py

Generates 3 sample rooms × 5 styles → app/static/precached/{style}_{room}.jpg.
During the demo these are served instantly — no wait, no per-request API cost.
"""
import sys
import asyncio
import shutil
from pathlib import Path

# Make app importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx  # noqa: E402

from app.services import image_gen  # noqa: E402
from app.services.image_gen import PRECACHED_DIR, RESULTS_DIR  # noqa: E402

SAMPLE_ROOMS = [
    ("bedroom",     "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=800&q=80"),
    ("living_room", "https://images.unsplash.com/photo-1493809842364-78817add7ffb?w=800&q=80"),
    ("kitchen",     "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&q=80"),
]

STYLES = ["minimal", "modern", "scandi", "wabi-sabi", "industrial"]


async def main():
    PRECACHED_DIR.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for room_type, url in SAMPLE_ROOMS:
            print(f"\nDownloading sample: {room_type}")
            image_bytes = (await client.get(url)).content

            for style in STYLES:
                out = PRECACHED_DIR / f"{style}_{room_type}.jpg"
                if out.exists():
                    print(f"  ✓ {out.name} (exists, skip)")
                    continue

                print(f"  Generating {style}_{room_type} …", end=" ", flush=True)
                result = await image_gen.generate_redesign(image_bytes, style, room_type)
                ru = result.result_url

                if "/static/results/" in ru:
                    # Local SD wrote a file we can copy directly (no server needed)
                    shutil.copy(RESULTS_DIR / ru.rsplit("/", 1)[-1], out)
                else:
                    # Remote (Replicate) or mock URL → download
                    out.write_bytes((await client.get(ru)).content)

                print(f"saved → {out.name}{'  (mock)' if result.is_mock else ''}")

    print("\nAll done! Pre-cache is ready for the demo.")


if __name__ == "__main__":
    asyncio.run(main())
