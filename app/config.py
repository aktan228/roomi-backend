from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    # Public URL used to build absolute URLs for locally generated images.
    # Must match what the browser can reach (e.g. http://localhost:8000 locally,
    # or https://api.yourdomain.com in production).
    public_url: str = "http://localhost:8000"

    cors_origins: str = "http://localhost:3000"

    # AI keys — empty string means mock mode
    replicate_api_token: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Groq (Llama) — design brief + chat
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    replicate_interior_model: str = (
        "adirik/interior-design:"
        "76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38"
    )

    # Local Stable Diffusion + ControlNet (RTX 3050 4GB friendly)
    local_sd: bool = False
    local_sd_model: str = "runwayml/stable-diffusion-v1-5"
    local_sd_controlnet: str = "lllyasviel/control_v11p_sd15_canny"
    local_sd_steps: int = 25
    local_sd_strength: float = 0.80
    local_sd_guidance: float = 12.0

    # Perception — local CV models
    enable_perception: bool = True
    yolo_model: str = "yolov8n.pt"
    fastsam_model: str = "FastSAM-s.pt"
    midas_hub_model: str = "MiDaS_small"
    # Soviet/typical apartments have standardized ceilings → use as metric anchor
    ceiling_height_m: float = 2.5

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_replicate(self) -> bool:
        return bool(self.replicate_api_token)

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def use_groq(self) -> bool:
        return bool(self.groq_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
