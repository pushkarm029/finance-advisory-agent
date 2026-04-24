from __future__ import annotations

from app.analytics.portfolio import compact_movers
from app.config import get_settings
from app.ingestion.news import compact_news
from app.models import (
    Briefing,
    EvaluationScore,
    EvaluationScoreDraft,
    NewsArticle,
    PortfolioAnalytics,
    StructuralScore,
)
from app.observability.tracing import logger, openai_client
from app.reasoning.prompts import EVALUATION_SYSTEM, build_evaluation_user_prompt


def _structural_score(
    briefing: Briefing,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
) -> StructuralScore:
    valid_ids = {a.id for a in relevant_news}
    links = briefing.causal_links
    total = len(links)

    if total == 0:
        return StructuralScore(
            valid_citation_ratio=0.0,
            impact_sum_gap_pct=abs(analytics.day_pnl_pct),
            sectors_populated_ratio=0.0,
            movers_coverage_ratio=0.0,
            score=0,
            notes="Briefing produced zero causal links.",
        )

    valid_citation_ratio = sum(1 for l in links if l.news_id in valid_ids) / total
    sectors_populated_ratio = sum(1 for l in links if l.sector.strip()) / total

    impact_sum = sum(l.portfolio_impact_pct for l in links)
    impact_sum_gap_pct = abs(impact_sum - analytics.day_pnl_pct)

    movers = list(analytics.top_contributors) + list(analytics.top_detractors)
    cited = {s for l in links for s in l.stocks_affected}
    movers_coverage_ratio = (
        sum(1 for m in movers if m.symbol in cited) / len(movers) if movers else 1.0
    )

    score = 0
    if valid_citation_ratio >= 0.99:
        score += 2
    elif valid_citation_ratio >= 0.5:
        score += 1
    if impact_sum_gap_pct <= 1.0:
        score += 1
    if movers_coverage_ratio >= 0.75:
        score += 2
    elif movers_coverage_ratio >= 0.5:
        score += 1
    score = min(5, score)

    issues: list[str] = []
    if valid_citation_ratio < 1.0:
        bad = round((1 - valid_citation_ratio) * total)
        issues.append(f"{bad} link(s) cite invalid news_ids")
    if impact_sum_gap_pct > 2:
        issues.append(
            f"sum(portfolio_impact_pct)={impact_sum:+.2f}pp vs actual "
            f"{analytics.day_pnl_pct:+.2f}pp (gap {impact_sum_gap_pct:.2f}pp)"
        )
    if movers_coverage_ratio < 0.5:
        issues.append(f"only {movers_coverage_ratio*100:.0f}% of top movers cited")
    notes = "; ".join(issues) or "Structural checks pass."

    return StructuralScore(
        valid_citation_ratio=round(valid_citation_ratio, 3),
        impact_sum_gap_pct=round(impact_sum_gap_pct, 3),
        sectors_populated_ratio=round(sectors_populated_ratio, 3),
        movers_coverage_ratio=round(movers_coverage_ratio, 3),
        score=score,
        notes=notes,
    )


def _llm_score(
    briefing: Briefing,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
) -> EvaluationScoreDraft | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    try:
        client = openai_client()
        portfolio_summary = {
            "day_pnl_pct": analytics.day_pnl_pct,
            "top_contributors": compact_movers(analytics.top_contributors),
            "top_detractors": compact_movers(analytics.top_detractors),
        }
        user_prompt = build_evaluation_user_prompt(
            briefing=briefing.model_dump(),
            portfolio_summary=portfolio_summary,
            relevant_news=compact_news(relevant_news),
        )
        resp = client.chat.completions.parse(
            model=settings.eval_model,
            temperature=settings.eval_temperature,
            max_completion_tokens=500,
            response_format=EvaluationScoreDraft,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            logger.warning("eval_llm_refused | %s", msg.refusal)
            return None
        return msg.parsed
    except Exception as exc:
        logger.warning("eval_llm_failed | %s", exc)
        return None


def evaluate_briefing(
    *,
    briefing: Briefing,
    analytics: PortfolioAnalytics,
    relevant_news: list[NewsArticle],
) -> EvaluationScore:
    structural = _structural_score(briefing, analytics, relevant_news)
    llm = _llm_score(briefing, analytics, relevant_news)

    if llm is None:
        return EvaluationScore(
            reasoning_quality=structural.score,
            causal_depth=structural.score,
            factual_grounding=structural.score,
            overall=structural.score,
            notes=f"structural only (no LLM grader available): {structural.notes}",
            structural=structural,
            llm=None,
        )

    return EvaluationScore(
        reasoning_quality=llm.reasoning_quality,
        causal_depth=llm.causal_depth,
        factual_grounding=min(structural.score, llm.factual_grounding),
        overall=min(structural.score, llm.overall),
        notes=f"{llm.notes.strip()} | structural: {structural.notes}",
        structural=structural,
        llm=llm,
    )
