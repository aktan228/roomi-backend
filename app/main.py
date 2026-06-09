from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": {
            "image_gen": "replicate" if settings.use_replicate else "mock",
            "chat": "openai" if settings.use_openai else "mock",
        },
    }
