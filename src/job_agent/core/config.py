from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "groq"
    groq_api_key: str = ""
    openai_api_key: str = ""

    fast_model_name: str = "openai/gpt-oss-20b"
    writer_model_name: str = "openai/gpt-oss-120b"
    temperature: float = 0.1

    max_revisions: int = 2
    min_grounding_score: float = 0.90
    min_specificity_score: float = 0.80

    profile_path: Path = (
        Path(__file__).resolve().parents[1] / "memory/candidate_profile.yaml"
    )


settings = Settings()
