from __future__ import annotations

import json
import re

from app.config import DATA_DIR
from app.models import NewsArticle, Scope, Sentiment

_POSITIVE_TOKENS = {
    "approval", "beats", "beat", "record", "raise", "raises", "upgraded",
    "upgrade", "rally", "rallied", "cushion", "momentum", "recovery",
    "supports", "boost", "strong", "expansion", "outperformed",
}
_NEGATIVE_TOKENS = {
    "hawkish", "hike", "hiked", "pressure", "pull", "sold off", "selloff",
    "disappointed", "concerns", "delay", "delayed", "subdued", "sticky",
    "cautious", "margin hit", "outflows", "fell", "falls", "downgrade",
    "downgraded", "risk", "risks",
}


def _heuristic_sentiment(text: str, provided: Sentiment | None) -> Sentiment:
    if provided:
        return provided
    lowered = text.lower()
    pos = sum(1 for t in _POSITIVE_TOKENS if t in lowered)
    neg = sum(1 for t in _NEGATIVE_TOKENS if t in lowered)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _infer_scope(entities: dict[str, list[str]], provided: Scope | None) -> Scope:
    if provided:
        return provided
    stocks = entities.get("stocks") or []
    sectors = entities.get("sectors") or []
    if len(stocks) == 1 and len(sectors) <= 1:
        return "stock"
    if sectors:
        return "sector"
    return "market"


def load_and_classify_news() -> list[NewsArticle]:
    raw = json.loads((DATA_DIR / "news.json").read_text(encoding="utf-8"))
    articles: list[NewsArticle] = []
    for item in raw["articles"]:
        text = f"{item['headline']} {item.get('summary', '')}"
        sentiment = _heuristic_sentiment(text, item.get("sentiment"))
        scope = _infer_scope(item.get("entities", {}), item.get("scope"))
        articles.append(
            NewsArticle(
                id=item["id"],
                headline=item["headline"],
                summary=item.get("summary", ""),
                entities=item.get("entities", {}),
                scope=scope,
                sentiment=sentiment,
            )
        )
    return articles


def news_for_portfolio(
    news: list[NewsArticle],
    symbols: set[str],
    sectors: set[str],
) -> list[NewsArticle]:
    """Filter to articles that could plausibly affect the portfolio."""
    relevant: list[NewsArticle] = []
    for article in news:
        if article.scope == "market":
            relevant.append(article)
            continue
        ents = article.entities
        touched_stocks = set(ents.get("stocks", [])) & symbols
        touched_sectors = set(ents.get("sectors", [])) & sectors
        if touched_stocks or touched_sectors:
            relevant.append(article)
    return relevant


_WS_RE = re.compile(r"\s+")


def compact_news(articles: list[NewsArticle]) -> list[dict]:
    """Compact representation suitable for LLM prompts (token-efficient)."""
    return [
        {
            "id": a.id,
            "headline": _WS_RE.sub(" ", a.headline).strip(),
            "summary": _WS_RE.sub(" ", a.summary).strip(),
            "scope": a.scope,
            "sentiment": a.sentiment,
            "sectors": a.entities.get("sectors", []),
            "stocks": a.entities.get("stocks", []),
        }
        for a in articles
    ]
