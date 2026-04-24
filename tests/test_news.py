from __future__ import annotations

from app.ingestion.news import load_and_classify_news, news_for_portfolio


def test_news_loads_with_expected_scope_distribution():
    articles = load_and_classify_news()
    scopes = {a.id: a.scope for a in articles}
    assert scopes["N006"] == "market"  # FIIs pull — market-wide
    assert scopes["N001"] == "sector"  # RBI banking
    assert scopes["N007"] == "stock"   # HDFC Bank only


def test_filter_to_portfolio_universe_includes_market_scope():
    articles = load_and_classify_news()
    relevant = news_for_portfolio(articles, symbols={"HDFCBANK"}, sectors={"Banking"})
    ids = {a.id for a in relevant}
    assert "N001" in ids  # Banking sector
    assert "N006" in ids  # market-scope, always included
    assert "N002" not in ids  # IT sector, not in universe


def test_pharma_only_portfolio_filters_out_banking():
    articles = load_and_classify_news()
    relevant = news_for_portfolio(articles, symbols={"SUNPHARMA"}, sectors={"Pharma"})
    ids = {a.id for a in relevant}
    assert "N003" in ids
    assert "N001" not in ids
    assert "N006" in ids  # market-scope
