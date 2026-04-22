from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"

for _candidate in (PROJECT_ROOT / ".env", ROOT / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_host: str
    concentration_threshold_pct: float = 40.0


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
