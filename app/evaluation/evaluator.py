from __future__ import annotations

from openai import OpenAI

from app.config import get_settings
from app.ingestion.news import compact_news
from app.models import Briefing, EvaluationScore, NewsArticle, PortfolioAnalytics
from app.observability.tracing import get_tracer
from app.reasoning.agent import _extract_json
from app.reasoning.prompts import EVALUATION_SYSTEM, build_evaluation_user_prompt


def _rule_based_fallback(
    *,
    briefing: Briefing,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
) -> EvaluationScore:
    """Cheap deterministic score used when no API key is configured or LLM errors."""
    valid_ids = {a.id for a in relevant_news}
    link_citations = [
        l for l in briefing.causal_links
        if (l.get("news_id") in valid_ids if isinstance(l, dict) else False)
    ]
    has_stock_mentions = any(
        (l.get("stocks_affected") if isinstance(l, dict) else [])
        for l in briefing.causal_links
    )

    grounding = 5 if link_citations and len(link_citations) == len(briefing.causal_links) else 3
    depth = 5 if has_stock_mentions and len(briefing.causal_links) >= 2 else 3
    quality = 4 if briefing.summary and len(briefing.causal_links) >= 2 else 2
    overall = round((grounding + depth + quality) / 3)
    notes = (
        f"Rule-based fallback: {len(link_citations)}/{len(briefing.causal_links)} "
        f"causal links cited valid news IDs."
    )
    return EvaluationScore(
        reasoning_quality=quality,
        causal_depth=depth,
        factual_grounding=grounding,
        overall=overall,
        notes=notes,
    )


def evaluate_briefing(
    *,
    briefing: Briefing,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
    span,
) -> EvaluationScore:
    settings = get_settings()
    if not settings.openai_api_key:
        return _rule_based_fallback(
            briefing=briefing, analytics=analytics, relevant_news=relevant_news
        )

    client = OpenAI(api_key=settings.openai_api_key)
    tracer = get_tracer()

    portfolio_summary = {
        "day_pnl_pct": analytics.day_pnl_pct,
        "top_contributors": [
            {"symbol": h.symbol, "sector": h.sector, "day_change_pct": h.day_change_pct}
            for h in analytics.top_contributors
        ],
        "top_detractors": [
            {"symbol": h.symbol, "sector": h.sector, "day_change_pct": h.day_change_pct}
            for h in analytics.top_detractors
        ],
    }
    news_payload = compact_news(relevant_news)
    briefing_payload = briefing.model_dump()

    user_prompt = build_evaluation_user_prompt(
        briefing=briefing_payload,
        portfolio_summary=portfolio_summary,
        relevant_news=news_payload,
    )

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        parsed = _extract_json(raw)
        tracer.log_generation(
            span,
            name="evaluation.score",
            model=settings.openai_model,
            input={"system": EVALUATION_SYSTEM, "user": user_prompt},
            output=raw,
            usage={
                "input_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                "output_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            },
        )
        return EvaluationScore(
            reasoning_quality=int(parsed.get("reasoning_quality", 3)),
            causal_depth=int(parsed.get("causal_depth", 3)),
            factual_grounding=int(parsed.get("factual_grounding", 3)),
            overall=int(parsed.get("overall", 3)),
            notes=str(parsed.get("notes", "")).strip(),
        )
    except Exception as exc:
        fallback = _rule_based_fallback(
            briefing=briefing, analytics=analytics, relevant_news=relevant_news
        )
        fallback = EvaluationScore(
            **{**fallback.model_dump(), "notes": f"LLM eval failed ({exc}); used rules."}
        )
        return fallback
