from __future__ import annotations

from typing import Any

from app.analytics.portfolio import compact_movers
from app.config import get_settings, require_openai_key
from app.ingestion.news import compact_news
from app.models import (
    Briefing,
    BriefingDraft,
    CausalLink,
    Conflict,
    MarketContext,
    NewsArticle,
    PortfolioAnalytics,
)
from app.observability.tracing import logger, openai_client
from app.reasoning.prompts import REASONING_SYSTEM, build_reasoning_user_prompt


def _compact_market(market: MarketContext) -> dict[str, Any]:
    return {
        "date": market.date,
        "market_sentiment": market.market_sentiment,
        "indices": [i.model_dump(include={"symbol", "change_pct"}) for i in market.indices],
        "sector_trends": [
            s.model_dump(include={"sector", "avg_change_pct", "sentiment"})
            for s in market.sector_trends
        ],
    }


def _compact_portfolio(analytics: PortfolioAnalytics) -> dict[str, Any]:
    exp_fields = {"sector", "weight_pct"}
    return {
        "portfolio_name": analytics.portfolio_name,
        "day_pnl_pct": analytics.day_pnl_pct,
        "day_pnl_abs": analytics.day_pnl_abs,
        "overall_pnl_pct": analytics.overall_pnl_pct,
        "sector_exposure": [s.model_dump(include=exp_fields) for s in analytics.sector_exposure],
        "fund_category_exposure": [
            s.model_dump(include=exp_fields) for s in analytics.fund_category_exposure
        ],
        "asset_type_exposure": analytics.asset_type_exposure,
        "risk_flags": [
            r.model_dump(include={"kind", "severity", "message"}) for r in analytics.risk_flags
        ],
        "top_contributors": compact_movers(analytics.top_contributors),
        "top_detractors": compact_movers(analytics.top_detractors),
    }


def _validate_links(
    draft: BriefingDraft,
    *,
    valid_news_ids: set[str],
    portfolio_symbols: set[str],
) -> tuple[list[CausalLink], list[str]]:
    kept: list[CausalLink] = []
    drops: list[str] = []
    for link in draft.causal_links:
        if link.news_id not in valid_news_ids:
            drops.append(f"news_id={link.news_id!r}: not in provided news")
            continue
        allowed = [s for s in link.stocks_affected if s in portfolio_symbols]
        if not allowed:
            drops.append(f"news_id={link.news_id!r}: no valid portfolio stocks cited")
            continue
        if allowed != link.stocks_affected:
            link = link.model_copy(update={"stocks_affected": allowed})
        kept.append(link)
    return kept, drops


def _derive_confidence(
    kept_links: list[CausalLink],
    analytics: PortfolioAnalytics,
) -> float:
    movers = list(analytics.top_contributors) + list(analytics.top_detractors)
    if not movers:
        return 1.0
    cited = {s for link in kept_links for s in link.stocks_affected}
    total = sum(abs(m.day_change_abs) for m in movers) or 1e-9
    covered = sum(abs(m.day_change_abs) for m in movers if m.symbol in cited)
    return max(0.0, min(1.0, covered / total))


def generate_briefing(
    *,
    market: MarketContext,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
    detected_conflicts: list[Conflict],
) -> Briefing:
    require_openai_key()
    settings = get_settings()
    client = openai_client()

    valid_news_ids = [a.id for a in relevant_news]
    portfolio_symbols = [h.symbol for h in analytics.holdings if h.type == "stock"]

    user_prompt = build_reasoning_user_prompt(
        market_context=_compact_market(market),
        portfolio_analytics=_compact_portfolio(analytics),
        relevant_news=compact_news(relevant_news),
        pre_detected_conflicts=detected_conflicts,
        valid_news_ids=valid_news_ids,
        portfolio_symbols=portfolio_symbols,
    )

    resp = client.chat.completions.parse(
        model=settings.briefing_model,
        temperature=settings.briefing_temperature,
        max_completion_tokens=1600,
        response_format=BriefingDraft,
        messages=[
            {"role": "system", "content": REASONING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    msg = resp.choices[0].message
    if getattr(msg, "refusal", None):
        raise RuntimeError(f"Briefing model refused: {msg.refusal}")
    draft: BriefingDraft | None = msg.parsed
    if draft is None:
        raise RuntimeError("Briefing model returned no parsed output")

    kept_links, drops = _validate_links(
        draft,
        valid_news_ids=set(valid_news_ids),
        portfolio_symbols={h.symbol for h in analytics.holdings},
    )
    if drops:
        logger.warning("briefing_links_dropped | %s", drops)

    return Briefing(
        headline=draft.headline.strip() or "Portfolio update",
        summary=draft.summary.strip(),
        causal_links=kept_links,
        conflicts=list(draft.conflicts),
        confidence=round(_derive_confidence(kept_links, analytics), 3),
        model=settings.briefing_model,
    )
