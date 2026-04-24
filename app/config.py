from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    briefing_model: str
    eval_model: str
    general_qa_model: str
    briefing_temperature: float
    eval_temperature: float
    general_qa_temperature: float
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    concentration_threshold_pct: float
    env: str

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        briefing_model=os.getenv("BRIEFING_MODEL", "gpt-4o-mini"),
        eval_model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
        general_qa_model=os.getenv("GENERAL_QA_MODEL", "gpt-4o-mini"),
        briefing_temperature=float(os.getenv("BRIEFING_TEMPERATURE", "0.2")),
        eval_temperature=float(os.getenv("EVAL_TEMPERATURE", "0.0")),
        general_qa_temperature=float(os.getenv("GENERAL_QA_TEMPERATURE", "0.4")),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        concentration_threshold_pct=float(os.getenv("CONCENTRATION_THRESHOLD_PCT", "40.0")),
        env=os.getenv("ENV", "dev"),
    )


def require_openai_key() -> str:
    key = get_settings().openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key
