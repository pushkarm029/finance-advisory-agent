from __future__ import annotations

from app.analytics.portfolio import (
    compute_analytics,
    detect_conflicts,
    load_portfolio,
)
from app.ingestion.market import load_market_snapshot
from app.ingestion.news import load_and_classify_news, news_for_portfolio
from app.models import (
    Holding,
    MarketSnapshot,
    Portfolio,
    StockQuote,
)


def test_p002_flags_banking_concentration_above_40():
    market = load_market_snapshot()
    analytics = compute_analytics(load_portfolio("P002"), market)
    kinds = {f.kind for f in analytics.risk_flags}
    assert "sector_concentration" in kinds
    banking = next(e for e in analytics.sector_exposure if e.sector == "Banking")
    assert banking.weight_pct > 40.0


def test_p001_diversified_has_no_sector_concentration_flag():
    market = load_market_snapshot()
    analytics = compute_analytics(load_portfolio("P001"), market)
    kinds = {f.kind for f in analytics.risk_flags}
    assert "sector_concentration" not in kinds


def test_p003_retiree_surfaces_fund_category_exposure():
    market = load_market_snapshot()
    analytics = compute_analytics(load_portfolio("P003"), market)
    categories = {e.sector for e in analytics.fund_category_exposure}
    assert categories & {"Debt-Liquid", "Debt-Gilt", "Equity-Largecap", "Equity-Flexicap"}
    # MF-heavy portfolio should flag fund category concentration
    kinds = {f.kind for f in analytics.risk_flags}
    assert "fund_category_concentration" in kinds


def test_concentration_threshold_strict_greater_than():
    """>40 means exactly 40.0 does NOT trigger."""
    market = MarketSnapshot(
        date="2026-04-22",
        indices={},
        stocks={
            "A": StockQuote(symbol="A", sector="X", previous_close=100, close=100, change_pct=0),
            "B": StockQuote(symbol="B", sector="Y", previous_close=100, close=100, change_pct=0),
        },
        mutual_funds={},
    )
    # 40% X, 60% Y → X NOT flagged, Y flagged
    pf = Portfolio(
        id="T", name="test", type="test",
        holdings=[
            Holding(symbol="A", type="stock", quantity=40, avg_price=100),
            Holding(symbol="B", type="stock", quantity=60, avg_price=100),
        ],
    )
    analytics = compute_analytics(pf, market)
    sectors_flagged = {
        f.message.split()[0] for f in analytics.risk_flags
        if f.kind == "sector_concentration"
    }
    assert "X" not in sectors_flagged
    assert "Y" in sectors_flagged


def test_missing_symbol_skipped_not_raised():
    market = load_market_snapshot()
    pf = Portfolio(
        id="T", name="test", type="test",
        holdings=[
            Holding(symbol="HDFCBANK", type="stock", quantity=10, avg_price=1500),
            Holding(symbol="UNKNOWN_SYMBOL", type="stock", quantity=10, avg_price=100),
        ],
    )
    analytics = compute_analytics(pf, market)
    assert analytics.missing_symbols == ["UNKNOWN_SYMBOL"]
    assert len(analytics.holdings) == 1
    assert analytics.holdings[0].symbol == "HDFCBANK"


def test_empty_portfolio_does_not_crash():
    market = load_market_snapshot()
    pf = Portfolio(id="T", name="empty", type="test", holdings=[])
    analytics = compute_analytics(pf, market)
    assert analytics.current_value == 0.0
    assert analytics.day_pnl_pct == 0.0
    assert analytics.top_contributors == []
    assert analytics.top_detractors == []


def test_day_pnl_pct_matches_price_derived_computation():
    """Regression on B3: day_pnl_pct comes from prev prices, not re-derived from day_pnl_abs."""
    market = load_market_snapshot()
    analytics = compute_analytics(load_portfolio("P002"), market)
    # Manually compute: sum(qty * prev_px) should match prev_total
    pf = load_portfolio("P002")
    prev_total = sum(
        h.quantity * market.stocks[h.symbol].previous_close
        for h in pf.holdings
        if h.type == "stock"
    )
    curr_total = sum(
        h.quantity * market.stocks[h.symbol].close
        for h in pf.holdings
        if h.type == "stock"
    )
    expected_pct = (curr_total / prev_total - 1.0) * 100
    assert abs(analytics.day_pnl_pct - expected_pct) < 0.01


def test_detect_conflicts_finds_opposite_direction():
    market = load_market_snapshot()
    all_news = load_and_classify_news()
    pf = Portfolio(
        id="T", name="test", type="test",
        holdings=[
            Holding(symbol="ITC", type="stock", quantity=50, avg_price=400),
        ],
    )
    analytics = compute_analytics(pf, market)
    symbols = {h.symbol for h in analytics.holdings}
    sectors = {h.sector for h in analytics.holdings if h.type == "stock"}
    relevant = news_for_portfolio(all_news, symbols=symbols, sectors=sectors)
    # ITC fell -0.37% but N008 (FMCG) is positive sentiment → potential conflict.
    # Actual move is < 0.5% threshold, so no conflict.
    conflicts = detect_conflicts(analytics.holdings, relevant, min_move_pct=0.5)
    assert conflicts == []
    # Lower threshold surfaces the mismatch
    conflicts = detect_conflicts(analytics.holdings, relevant, min_move_pct=0.1)
    assert any(c.stock == "ITC" and c.news_id == "N008" for c in conflicts)


def test_detect_conflicts_surfaces_hdfcbank_n007_as_ambiguous():
    """Brief's canonical example: HDFCBANK -3.37% vs N007 'in-line but cautious' (neutral)
    must surface as an ambiguous-signal conflict because the stock moved >2% on neutral
    news that explicitly tags it."""
    market = load_market_snapshot()
    all_news = load_and_classify_news()
    analytics = compute_analytics(load_portfolio("P002"), market)
    relevant = news_for_portfolio(
        all_news,
        symbols={h.symbol for h in analytics.holdings},
        sectors={h.sector for h in analytics.holdings if h.type == "stock"},
    )
    conflicts = detect_conflicts(analytics.holdings, relevant)
    hdfc_n007 = [
        c for c in conflicts if c.stock == "HDFCBANK" and c.news_id == "N007"
    ]
    assert len(hdfc_n007) == 1
    assert hdfc_n007[0].news_sentiment == "neutral"


def test_detect_conflicts_skips_neutral_small_move():
    """Neutral news + small move is NOT a conflict (it's just noise)."""
    market = load_market_snapshot()
    all_news = load_and_classify_news()
    pf = Portfolio(
        id="T", name="t", type="t",
        holdings=[Holding(symbol="ITC", type="stock", quantity=10, avg_price=400)],
    )
    analytics = compute_analytics(pf, market)
    relevant = news_for_portfolio(
        all_news, symbols={"ITC"}, sectors={"FMCG"},
    )
    # ITC only moved -0.37% — below both thresholds. Even at ambiguous_move_pct=2.0 it's skipped.
    conflicts = detect_conflicts(analytics.holdings, relevant)
    assert not any(c.stock == "ITC" and c.news_sentiment == "neutral" for c in conflicts)
