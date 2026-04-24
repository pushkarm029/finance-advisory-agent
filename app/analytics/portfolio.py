from __future__ import annotations

import json
from collections import defaultdict

from app.config import DATA_DIR, get_settings
from app.models import (
    Conflict,
    Holding,
    HoldingPnL,
    MarketSnapshot,
    NewsArticle,
    Portfolio,
    PortfolioAnalytics,
    RiskFlag,
    SectorExposure,
)


def load_portfolios() -> list[Portfolio]:
    raw = json.loads((DATA_DIR / "portfolios.json").read_text(encoding="utf-8"))
    return [Portfolio(**p) for p in raw["portfolios"]]


def load_portfolio(portfolio_id: str) -> Portfolio:
    for p in load_portfolios():
        if p.id == portfolio_id:
            return p
    raise KeyError(f"Portfolio {portfolio_id!r} not found")


def _lookup_price(
    holding: Holding, market: MarketSnapshot
) -> tuple[float, float, str] | None:
    if holding.type == "stock":
        quote = market.stocks.get(holding.symbol)
        if quote is None:
            return None
        return quote.previous_close, quote.close, quote.sector
    quote = market.mutual_funds.get(holding.symbol)
    if quote is None:
        return None
    return quote.previous_nav, quote.nav, quote.asset_type


def compute_analytics(portfolio: Portfolio, market: MarketSnapshot) -> PortfolioAnalytics:
    settings = get_settings()
    holdings_pnl: list[HoldingPnL] = []
    missing: list[str] = []
    invested_total = 0.0
    current_total = 0.0
    prev_total = 0.0

    for h in portfolio.holdings:
        prices = _lookup_price(h, market)
        if prices is None:
            missing.append(h.symbol)
            continue
        prev_px, curr_px, bucket = prices

        invested = h.quantity * h.avg_price
        prev_value = h.quantity * prev_px
        current_value = h.quantity * curr_px
        day_abs = current_value - prev_value
        day_pct = (curr_px / prev_px - 1.0) * 100 if prev_px else 0.0
        overall_abs = current_value - invested
        overall_pct = (current_value / invested - 1.0) * 100 if invested else 0.0

        invested_total += invested
        current_total += current_value
        prev_total += prev_value

        holdings_pnl.append(
            HoldingPnL(
                symbol=h.symbol,
                type=h.type,
                sector=bucket,
                quantity=h.quantity,
                avg_price=h.avg_price,
                current_price=curr_px,
                invested=round(invested, 2),
                current_value=round(current_value, 2),
                day_change_abs=round(day_abs, 2),
                day_change_pct=round(day_pct, 3),
                overall_pnl_abs=round(overall_abs, 2),
                overall_pnl_pct=round(overall_pct, 3),
            )
        )

    day_pnl_abs = current_total - prev_total
    day_pnl_pct = (current_total / prev_total - 1.0) * 100 if prev_total else 0.0
    overall_abs = current_total - invested_total
    overall_pct = (current_total / invested_total - 1.0) * 100 if invested_total else 0.0

    stock_sector_totals: dict[str, float] = defaultdict(float)
    fund_category_totals: dict[str, float] = defaultdict(float)
    asset_type_totals: dict[str, float] = defaultdict(float)
    for h in holdings_pnl:
        if h.type == "stock":
            stock_sector_totals[h.sector] += h.current_value
        else:
            fund_category_totals[h.sector] += h.current_value
        asset_type_totals[h.type] += h.current_value

    exposures = _weighted_exposures(stock_sector_totals, current_total)
    fund_exposures = _weighted_exposures(fund_category_totals, current_total)

    asset_type_pct = {
        k: round(v / current_total * 100, 2) if current_total else 0.0
        for k, v in asset_type_totals.items()
    }

    risk_flags = _detect_risks(
        exposures=exposures,
        fund_exposures=fund_exposures,
        asset_type_pct=asset_type_pct,
        holdings_pnl=holdings_pnl,
        threshold=settings.concentration_threshold_pct,
    )

    ranked = sorted(holdings_pnl, key=lambda h: h.day_change_abs, reverse=True)
    top_contributors = [h for h in ranked if h.day_change_abs > 0][:3]
    top_detractors = [h for h in reversed(ranked) if h.day_change_abs < 0][:3]

    return PortfolioAnalytics(
        portfolio_id=portfolio.id,
        portfolio_name=portfolio.name,
        invested_total=round(invested_total, 2),
        current_value=round(current_total, 2),
        day_pnl_abs=round(day_pnl_abs, 2),
        day_pnl_pct=round(day_pnl_pct, 3),
        overall_pnl_abs=round(overall_abs, 2),
        overall_pnl_pct=round(overall_pct, 3),
        holdings=holdings_pnl,
        sector_exposure=exposures,
        fund_category_exposure=fund_exposures,
        asset_type_exposure=asset_type_pct,
        risk_flags=risk_flags,
        top_contributors=top_contributors,
        top_detractors=top_detractors,
        missing_symbols=missing,
    )


def _weighted_exposures(
    totals: dict[str, float], current_total: float
) -> list[SectorExposure]:
    return [
        SectorExposure(
            sector=sector,
            value=round(value, 2),
            weight_pct=round(value / current_total * 100, 2) if current_total else 0.0,
        )
        for sector, value in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]


def _detect_risks(
    *,
    exposures: list[SectorExposure],
    fund_exposures: list[SectorExposure],
    asset_type_pct: dict[str, float],
    holdings_pnl: list[HoldingPnL],
    threshold: float,
) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for exp in exposures:
        if exp.weight_pct > threshold:
            flags.append(
                RiskFlag(
                    kind="sector_concentration",
                    message=(
                        f"{exp.sector} is {exp.weight_pct:.1f}% of the portfolio, "
                        f"above the {threshold:.0f}% concentration threshold."
                    ),
                    severity="high" if exp.weight_pct > threshold + 15 else "medium",
                )
            )
    for exp in fund_exposures:
        if exp.weight_pct > threshold:
            flags.append(
                RiskFlag(
                    kind="fund_category_concentration",
                    message=(
                        f"{exp.weight_pct:.1f}% allocated to {exp.sector} funds — "
                        "heavy single-category MF concentration."
                    ),
                    severity="medium",
                )
            )

    stocks_by_value = [h for h in holdings_pnl if h.type == "stock"]
    total = sum(h.current_value for h in holdings_pnl)
    if stocks_by_value and total:
        top = max(stocks_by_value, key=lambda h: h.current_value)
        weight = top.current_value / total * 100
        if weight > 25:
            flags.append(
                RiskFlag(
                    kind="single_stock_concentration",
                    message=(
                        f"{top.symbol} alone is {weight:.1f}% of the portfolio — "
                        "consider diversifying idiosyncratic exposure."
                    ),
                    severity="medium",
                )
            )

    equity_pct = asset_type_pct.get("stock", 0.0)
    if equity_pct > 90:
        flags.append(
            RiskFlag(
                kind="equity_skew",
                message=f"Portfolio is {equity_pct:.0f}% equities with minimal fixed-income buffer.",
                severity="low",
            )
        )

    return flags


def portfolio_universe(analytics: PortfolioAnalytics) -> tuple[set[str], set[str]]:
    symbols = {h.symbol for h in analytics.holdings}
    sectors = {h.sector for h in analytics.holdings if h.type == "stock"}
    return symbols, sectors


def compact_movers(holdings: list[HoldingPnL]) -> list[dict]:
    """Token-efficient mover summary shared by the reasoning prompt and the eval prompt."""
    return [
        {
            "symbol": h.symbol,
            "sector": h.sector,
            "day_change_pct": h.day_change_pct,
            "day_change_abs": h.day_change_abs,
        }
        for h in holdings
    ]


def detect_conflicts(
    holdings_pnl: list[HoldingPnL],
    news: list[NewsArticle],
    *,
    min_move_pct: float = 0.5,
    ambiguous_move_pct: float = 2.0,
) -> list[Conflict]:
    """Deterministically flag holdings where news signal doesn't cleanly explain the move.

    Two cases emit a conflict (one per (symbol, news_id) pair):

    1. **Opposite direction** — news sentiment is positive/negative and the stock moved
       the other way. Requires |day_change_pct| >= min_move_pct.
    2. **Ambiguous signal** — neutral-sentiment news explicitly tags the stock, and the
       stock still moved |day_change_pct| >= ambiguous_move_pct. This catches cases like
       HDFC Bank's 'in-line earnings but cautious guidance' which are mixed signals
       rather than genuine non-signals.
    """
    conflicts: list[Conflict] = []
    seen: set[tuple[str, str]] = set()

    for h in holdings_pnl:
        if abs(h.day_change_pct) < min_move_pct:
            continue
        price_dir = 1 if h.day_change_pct > 0 else -1
        for n in news:
            touches_stock = h.symbol in (n.entities.get("stocks") or [])
            touches_sector = (
                h.type == "stock" and h.sector in (n.entities.get("sectors") or [])
            )
            if not (touches_stock or touches_sector):
                continue

            key = (h.symbol, n.id)
            if key in seen:
                continue

            if n.sentiment == "neutral":
                if touches_stock and abs(h.day_change_pct) >= ambiguous_move_pct:
                    seen.add(key)
                    conflicts.append(
                        Conflict(
                            stock=h.symbol,
                            day_change_pct=h.day_change_pct,
                            news_id=n.id,
                            news_sentiment="neutral",
                            note=(
                                f"Neutral-sentiment news but {h.symbol} moved "
                                f"{h.day_change_pct:+.2f}% — mixed signal worth narrating."
                            ),
                        )
                    )
                continue

            sentiment_dir = 1 if n.sentiment == "positive" else -1
            if price_dir == sentiment_dir:
                continue
            seen.add(key)
            conflicts.append(
                Conflict(
                    stock=h.symbol,
                    day_change_pct=h.day_change_pct,
                    news_id=n.id,
                    news_sentiment=n.sentiment,
                    note=(
                        f"{n.sentiment.title()} news (scope={n.scope}) but {h.symbol} "
                        f"moved {h.day_change_pct:+.2f}%."
                    ),
                )
            )
    return conflicts
