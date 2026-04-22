from __future__ import annotations

import json
from collections import defaultdict

from app.config import DATA_DIR, get_settings
from app.models import (
    Holding,
    HoldingPnL,
    MarketSnapshot,
    Portfolio,
    PortfolioAnalytics,
    RiskFlag,
    SectorExposure,
)

_CASH_LIKE = {"Debt-Liquid", "Debt-Gilt"}


def load_portfolios() -> list[Portfolio]:
    raw = json.loads((DATA_DIR / "portfolios.json").read_text(encoding="utf-8"))
    return [Portfolio(**p) for p in raw["portfolios"]]


def load_portfolio(portfolio_id: str) -> Portfolio:
    for p in load_portfolios():
        if p.id == portfolio_id:
            return p
    raise KeyError(f"Portfolio {portfolio_id!r} not found")


def _price_for(holding: Holding, market: MarketSnapshot) -> tuple[float, float, str]:
    """Returns (previous_close_or_nav, current_price_or_nav, sector_or_asset_type)."""
    if holding.type == "stock":
        quote = market.stocks.get(holding.symbol)
        if quote is None:
            raise KeyError(f"No market data for stock {holding.symbol}")
        return quote.previous_close, quote.close, quote.sector
    quote = market.mutual_funds.get(holding.symbol)
    if quote is None:
        raise KeyError(f"No market data for fund {holding.symbol}")
    return quote.previous_nav, quote.nav, quote.asset_type


def compute_analytics(portfolio: Portfolio, market: MarketSnapshot) -> PortfolioAnalytics:
    settings = get_settings()
    holdings_pnl: list[HoldingPnL] = []

    for h in portfolio.holdings:
        prev_px, curr_px, bucket = _price_for(h, market)
        invested = h.quantity * h.avg_price
        current_value = h.quantity * curr_px
        day_abs = h.quantity * (curr_px - prev_px)
        day_pct = (curr_px / prev_px - 1.0) * 100 if prev_px else 0.0
        overall_abs = current_value - invested
        overall_pct = (current_value / invested - 1.0) * 100 if invested else 0.0

        sector: str | None
        asset_type = h.type
        if h.type == "stock":
            sector = bucket
        else:
            sector = bucket  # e.g. "Equity-Largecap"

        holdings_pnl.append(
            HoldingPnL(
                symbol=h.symbol,
                type=asset_type,
                sector=sector,
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

    invested_total = sum(h.invested for h in holdings_pnl)
    current_total = sum(h.current_value for h in holdings_pnl)
    day_pnl_abs = sum(h.day_change_abs for h in holdings_pnl)
    prev_total = current_total - day_pnl_abs
    day_pnl_pct = (current_total / prev_total - 1.0) * 100 if prev_total else 0.0
    overall_abs = current_total - invested_total
    overall_pct = (current_total / invested_total - 1.0) * 100 if invested_total else 0.0

    stock_sector_totals: dict[str, float] = defaultdict(float)
    fund_category_totals: dict[str, float] = defaultdict(float)
    asset_type_totals: dict[str, float] = defaultdict(float)
    for h in holdings_pnl:
        if h.type == "stock" and h.sector:
            stock_sector_totals[h.sector] += h.current_value
        elif h.type == "mutual_fund" and h.sector:
            fund_category_totals[h.sector] += h.current_value
        asset_type_totals[h.type] += h.current_value

    exposures = [
        SectorExposure(
            sector=sector,
            value=round(value, 2),
            weight_pct=round(value / current_total * 100, 2) if current_total else 0.0,
        )
        for sector, value in sorted(stock_sector_totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    fund_exposures = [
        SectorExposure(
            sector=cat,
            value=round(value, 2),
            weight_pct=round(value / current_total * 100, 2) if current_total else 0.0,
        )
        for cat, value in sorted(fund_category_totals.items(), key=lambda kv: kv[1], reverse=True)
    ]

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
    top_detractors = [h for h in ranked[::-1] if h.day_change_abs < 0][:3]

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
        asset_type_exposure=asset_type_pct,
        risk_flags=risk_flags,
        top_contributors=top_contributors,
        top_detractors=top_detractors,
    )


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
        if exp.weight_pct >= threshold:
            flags.append(
                RiskFlag(
                    kind="sector_concentration",
                    message=(
                        f"{exp.sector} represents {exp.weight_pct:.1f}% of the portfolio, "
                        f"above the {threshold:.0f}% concentration threshold."
                    ),
                    severity="high" if exp.weight_pct >= threshold + 15 else "medium",
                )
            )
    for exp in fund_exposures:
        if exp.weight_pct >= threshold:
            flags.append(
                RiskFlag(
                    kind="fund_category_concentration",
                    message=(
                        f"{exp.weight_pct:.1f}% allocated to {exp.sector} funds - "
                        "heavy single-category MF concentration."
                    ),
                    severity="medium",
                )
            )

    single_stock = max(
        (h for h in holdings_pnl if h.type == "stock"),
        key=lambda h: h.current_value,
        default=None,
    )
    total = sum(h.current_value for h in holdings_pnl)
    if single_stock and total and (single_stock.current_value / total) * 100 >= 25:
        weight = single_stock.current_value / total * 100
        flags.append(
            RiskFlag(
                kind="single_stock_concentration",
                message=(
                    f"{single_stock.symbol} alone is {weight:.1f}% of the portfolio - "
                    "consider diversifying idiosyncratic exposure."
                ),
                severity="medium",
            )
        )

    equity_pct = sum(v for k, v in asset_type_pct.items() if k == "stock")
    if equity_pct >= 90:
        flags.append(
            RiskFlag(
                kind="equity_skew",
                message=f"Portfolio is {equity_pct:.0f}% equities with minimal fixed-income buffer.",
                severity="low",
            )
        )

    return flags


def portfolio_universe(analytics: PortfolioAnalytics) -> tuple[set[str], set[str]]:
    """Return (symbols_held, sectors_held) — used to filter relevant news."""
    symbols = {h.symbol for h in analytics.holdings}
    sectors = {h.sector for h in analytics.holdings if h.type == "stock" and h.sector}
    return symbols, sectors
