from __future__ import annotations

from app.evaluation.evaluator import _structural_score
from app.ingestion.market import load_market_snapshot
from app.ingestion.news import load_and_classify_news, news_for_portfolio
from app.analytics.portfolio import compute_analytics, load_portfolio, portfolio_universe
from app.models import Briefing, CausalLink


def _ctx(pid: str = "P001"):
    market = load_market_snapshot()
    analytics = compute_analytics(load_portfolio(pid), market)
    symbols, sectors = portfolio_universe(analytics)
    relevant = news_for_portfolio(load_and_classify_news(), symbols=symbols, sectors=sectors)
    return analytics, relevant


def test_empty_briefing_scores_zero():
    analytics, relevant = _ctx()
    briefing = Briefing(
        headline="", summary="", causal_links=[], conflicts=[],
        confidence=0.0, model="test",
    )
    score = _structural_score(briefing, analytics, relevant)
    assert score.score == 0
    assert "zero causal links" in score.notes.lower()


def test_good_briefing_scores_high():
    analytics, relevant = _ctx("P002")
    # Build a briefing citing N001 and covering the bank detractors
    links = [
        CausalLink(
            news_id="N001", driver="RBI hawkish hold", sector="Banking",
            stocks_affected=[m.symbol for m in analytics.top_detractors if m.sector == "Banking"],
            portfolio_impact_pct=analytics.day_pnl_pct,
            explanation="Banking sell-off compresses NIM on rate-cut delay.",
        ),
    ]
    briefing = Briefing(
        headline="Banks drag portfolio", summary="...", causal_links=links,
        conflicts=[], confidence=0.9, model="test",
    )
    score = _structural_score(briefing, analytics, relevant)
    assert score.valid_citation_ratio == 1.0
    assert score.score >= 4


def test_invalid_news_id_penalized():
    analytics, relevant = _ctx()
    links = [
        CausalLink(
            news_id="N999", driver="fake", sector="IT",
            stocks_affected=["TCS"], portfolio_impact_pct=0.5,
            explanation="made-up news",
        ),
    ]
    briefing = Briefing(
        headline="x", summary="x", causal_links=links, conflicts=[],
        confidence=0.5, model="test",
    )
    score = _structural_score(briefing, analytics, relevant)
    assert score.valid_citation_ratio == 0.0
    assert "invalid news_ids" in score.notes
