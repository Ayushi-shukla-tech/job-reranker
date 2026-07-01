from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str = ""
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_pro_model: str = "gemini-2.5-pro-preview"
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemma_model: str = "google/gemma-3-300M"
    prefilter_top_n: int = 2500
    final_candidates_n: int = 150
    semantic_weight: float = 0.65
    redrob_weight: float = 0.35
    candidates_file: str = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.json"

    @property
    def cache_dir(self) -> Path:
        return Path("cache")

    @property
    def embeddings_cache_dir(self) -> Path:
        p = Path("cache/embeddings_cache")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def gemma_cache_dir(self) -> Path:
        p = Path("cache/gemma_cache")
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()
