from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import get_settings
from app.api import redesign, products, plan, chat

settings = get_settings()

app = FastAPI(
    title="roomi.ai API",
    description="AI-powered interior design platform for Central Asia",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(redesign.router, prefix="/api", tags=["redesign"])
app.include_router(products.router, prefix="/api", tags=["products"])
app.include_router(plan.router, prefix="/api", tags=["plan"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

# Serve locally generated and pre-cached images
_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
(_static / "results").mkdir(exist_ok=True)
(_static / "precached").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": {
            "image_gen": (
                "local_sd" if settings.local_sd else
                "replicate" if settings.use_replicate else "mock"
            ),
            "chat": "groq" if settings.use_groq else "openai" if settings.use_openai else "mock",
        },
    }
