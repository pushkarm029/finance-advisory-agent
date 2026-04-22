from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.ingestion.news import compact_news
from app.models import Briefing, MarketContext, NewsArticle, PortfolioAnalytics
from app.observability.tracing import get_tracer
from app.reasoning.prompts import (
    REASONING_SYSTEM,
    build_reasoning_user_prompt,
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(text: str) -> dict[str, Any]:
    text = _JSON_FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _compact_market(market: MarketContext) -> dict[str, Any]:
    return {
        "date": market.date,
        "market_sentiment": market.market_sentiment,
        "indices": [
            {"symbol": i.symbol, "change_pct": i.change_pct} for i in market.indices
        ],
        "sector_trends": [
            {
                "sector": s.sector,
                "avg_change_pct": s.avg_change_pct,
                "sentiment": s.sentiment,
            }
            for s in market.sector_trends
        ],
    }


def _compact_portfolio(analytics: PortfolioAnalytics) -> dict[str, Any]:
    return {
        "portfolio_name": analytics.portfolio_name,
        "day_pnl_pct": analytics.day_pnl_pct,
        "day_pnl_abs": analytics.day_pnl_abs,
        "overall_pnl_pct": analytics.overall_pnl_pct,
        "sector_exposure": [
            {"sector": s.sector, "weight_pct": s.weight_pct}
            for s in analytics.sector_exposure
        ],
        "asset_type_exposure": analytics.asset_type_exposure,
        "risk_flags": [
            {"kind": r.kind, "severity": r.severity, "message": r.message}
            for r in analytics.risk_flags
        ],
        "top_contributors": [
            {
                "symbol": h.symbol,
                "sector": h.sector,
                "day_change_pct": h.day_change_pct,
                "day_change_abs": h.day_change_abs,
            }
            for h in analytics.top_contributors
        ],
        "top_detractors": [
            {
                "symbol": h.symbol,
                "sector": h.sector,
                "day_change_pct": h.day_change_pct,
                "day_change_abs": h.day_change_abs,
            }
            for h in analytics.top_detractors
        ],
    }


def generate_briefing(
    *,
    market: MarketContext,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
    span,
) -> Briefing:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    client = OpenAI(api_key=settings.openai_api_key)
    tracer = get_tracer()

    market_payload = _compact_market(market)
    portfolio_payload = _compact_portfolio(analytics)
    news_payload = compact_news(relevant_news)

    user_prompt = build_reasoning_user_prompt(
        market_context=market_payload,
        portfolio_analytics=portfolio_payload,
        relevant_news=news_payload,
    )

    resp = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=1600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = resp.choices[0].message.content or ""
    parsed = _extract_json(raw_text)

    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
        "output_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
    }
    tracer.log_generation(
        span,
        name="reasoning.briefing",
        model=settings.openai_model,
        input={"system": REASONING_SYSTEM, "user": user_prompt},
        output=raw_text,
        usage=usage,
    )

    briefing = Briefing(
        headline=parsed.get("headline", "").strip() or "Portfolio update",
        summary=parsed.get("summary", "").strip(),
        causal_links=parsed.get("causal_links", []),
        conflicts=parsed.get("conflicts", []),
        confidence=float(parsed.get("confidence", 0.5)),
        model=settings.openai_model,
    )
    return briefing
