"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """Runtime configuration for LLM, search, database, and API."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()
    search_provider: str = os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower()
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()
    database_path: str = os.getenv("DATABASE_PATH", "data/travel_planner.db").strip()
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip()

    def effective_llm_model(self) -> str:
        """Return configured model name or sensible provider default."""
        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "openai":
            return "gpt-4o-mini"
        return "llama-3.3-70b-versatile"

    def ensure_db_dir(self) -> Path:
        """Ensure parent directory for the SQLite file exists."""
        db_file = Path(self.database_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return db_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (reload env on demand in tests)."""
    load_dotenv(override=False)
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        search_provider=os.getenv("SEARCH_PROVIDER", "duckduckgo").strip().lower(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        database_path=os.getenv("DATABASE_PATH", "data/travel_planner.db").strip(),
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").strip(),
    )

