from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = "AIzaSyCztLKhnxuPyigTVcni1RBW2Zpklu0bU3Q" 
    adk_model: str = "gemini-2.5-flash-lite"

    llm_timeout_seconds: int = 30
    tool_timeout_seconds: int = 10


settings = Settings()

# ADK/Gemini clients often read API keys from process env vars.
# Mirror loaded settings into env so local .env-based runs work reliably.
if settings.google_api_key:
    os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    os.environ["GEMINI_API_KEY"] = settings.google_api_key
