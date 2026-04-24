from __future__ import annotations

from app.analytics.portfolio import (
    compute_analytics,
    detect_conflicts,
    load_portfolio,
    portfolio_universe,
)
from app.evaluation.evaluator import evaluate_briefing
from app.ingestion.market import (
    classify_market_sentiment,
    derive_sector_trends,
    load_market_snapshot,
)
from app.ingestion.news import load_and_classify_news, news_for_portfolio
from app.models import AgentResponse, MarketContext
from app.observability.tracing import flush, trace_span
from app.reasoning.agent import generate_briefing


def run_agent(portfolio_id: str) -> AgentResponse:
    with trace_span("agent.run", portfolio_id=portfolio_id) as tr:
        market = load_market_snapshot()
        all_news = load_and_classify_news()
        portfolio = load_portfolio(portfolio_id)
        analytics = compute_analytics(portfolio, market)

        symbols, sectors = portfolio_universe(analytics)
        relevant_news = news_for_portfolio(all_news, symbols=symbols, sectors=sectors)
        conflicts = detect_conflicts(analytics.holdings, relevant_news)
        analytics = analytics.model_copy(update={"detected_conflicts": conflicts})

        market_context = MarketContext(
            date=market.date,
            market_sentiment=classify_market_sentiment(market.indices),
            indices=list(market.indices.values()),
            sector_trends=derive_sector_trends(market.stocks),
            classified_news=relevant_news,
        )

        tr.update(
            day_pnl_pct=analytics.day_pnl_pct,
            relevant_news_count=len(relevant_news),
            risk_flag_count=len(analytics.risk_flags),
            detected_conflict_count=len(conflicts),
            missing_symbols=analytics.missing_symbols,
        )

        briefing = generate_briefing(
            market=market_context,
            analytics=analytics,
            relevant_news=relevant_news,
            detected_conflicts=conflicts,
        )
        evaluation = evaluate_briefing(
            briefing=briefing,
            analytics=analytics,
            relevant_news=relevant_news,
        )

        response = AgentResponse(
            portfolio=analytics,
            market=market_context,
            briefing=briefing,
            evaluation=evaluation,
            trace_id=tr.trace_id,
            trace_url=tr.trace_url,
        )

    flush()
    return response
