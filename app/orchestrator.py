from __future__ import annotations

from app.analytics.portfolio import (
    compute_analytics,
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
from app.observability.tracing import get_tracer
from app.reasoning.agent import generate_briefing


def run_agent(portfolio_id: str) -> AgentResponse:
    """Top-level entry point: produces the full briefing for a single portfolio."""
    tracer = get_tracer()

    with tracer.trace("agent.run", portfolio_id=portfolio_id) as span:
        # 1. Deterministic ingestion + analytics (no LLM calls)
        market = load_market_snapshot()
        all_news = load_and_classify_news()
        sector_trends = derive_sector_trends(market.stocks)
        market_sentiment = classify_market_sentiment(market.indices)

        portfolio = load_portfolio(portfolio_id)
        analytics = compute_analytics(portfolio, market)

        symbols, sectors = portfolio_universe(analytics)
        relevant_news = news_for_portfolio(all_news, symbols=symbols, sectors=sectors)

        market_context = MarketContext(
            date=market.date,
            market_sentiment=market_sentiment,
            indices=list(market.indices.values()),
            sector_trends=sector_trends,
            classified_news=relevant_news,
        )

        span.update(
            day_pnl_pct=analytics.day_pnl_pct,
            relevant_news_count=len(relevant_news),
            risk_flag_count=len(analytics.risk_flags),
        )

        # 2. Single LLM call for reasoning
        briefing = generate_briefing(
            market=market_context,
            analytics=analytics,
            relevant_news=relevant_news,
            span=span,
        )

        # 3. Self-evaluation
        evaluation = evaluate_briefing(
            briefing=briefing,
            analytics=analytics,
            relevant_news=relevant_news,
            span=span,
        )

        response = AgentResponse(
            portfolio=analytics,
            market=market_context,
            briefing=briefing,
            evaluation=evaluation,
            trace_id=span.trace_id,
        )

    tracer.flush()
    return response
