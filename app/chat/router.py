from __future__ import annotations

from typing import Literal

from app.analytics.portfolio import load_portfolios

Intent = Literal["portfolio", "general"]

_PORTFOLIO_KEYWORDS: tuple[str, ...] = (
    "my ",
    "portfolio",
    "holding",
    "briefing",
    "brief me",
    "p&l",
    "pnl",
    "today",
    "my stocks",
    "mine",
    "returns",
    "performance",
    "exposure",
    "allocation",
    "concentration",
    "risk flag",
    "how did",
    "why did",
)


def route_intent(message: str, portfolio_id: str | None) -> Intent:
    text = (message or "").lower().strip()
    known_ids = [p.id.lower() for p in load_portfolios()]
    if any(pid in text for pid in known_ids):
        return "portfolio"
    if not portfolio_id:
        return "general"
    if any(kw in text for kw in _PORTFOLIO_KEYWORDS):
        return "portfolio"
    return "general"
