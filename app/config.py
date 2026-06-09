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

    cors_origins: str = "http://localhost:3000"

    # AI keys — empty string means mock mode
    replicate_api_token: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    replicate_interior_model: str = (
        "adirik/interior-design:"
        "76604baddc85b1b4616e1c6475eca080da339c8875bd4996705440484a6eac38"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def use_replicate(self) -> bool:
        return bool(self.replicate_api_token)

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
