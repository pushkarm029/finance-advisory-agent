from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from app.config import DATA_DIR
from app.models import (
    FundQuote,
    IndexQuote,
    MarketSentiment,
    MarketSnapshot,
    SectorTrend,
    StockQuote,
)


def load_market_snapshot() -> MarketSnapshot:
    raw = json.loads((DATA_DIR / "market.json").read_text(encoding="utf-8"))
    return MarketSnapshot(
        date=raw["date"],
        indices={k: IndexQuote(**v) for k, v in raw["indices"].items()},
        stocks={k: StockQuote(**v) for k, v in raw["stocks"].items()},
        mutual_funds={k: FundQuote(**v) for k, v in raw["mutual_funds"].items()},
    )


def classify_market_sentiment(indices: dict[str, IndexQuote]) -> MarketSentiment:
    broad = [indices[k].change_pct for k in ("NIFTY50", "SENSEX") if k in indices]
    if not broad:
        return "neutral"
    avg = mean(broad)
    if avg >= 0.5:
        return "bullish"
    if avg <= -0.5:
        return "bearish"
    return "neutral"


def derive_sector_trends(stocks: dict[str, StockQuote]) -> list[SectorTrend]:
    buckets: dict[str, list[StockQuote]] = defaultdict(list)
    for quote in stocks.values():
        buckets[quote.sector].append(quote)

    trends: list[SectorTrend] = []
    for sector, quotes in buckets.items():
        avg_change = mean(q.change_pct for q in quotes)
        if avg_change >= 0.75:
            sentiment: MarketSentiment = "bullish"
        elif avg_change <= -0.75:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        trends.append(
            SectorTrend(
                sector=sector,
                avg_change_pct=round(avg_change, 3),
                sentiment=sentiment,
                constituents=[q.symbol for q in quotes],
            )
        )
    trends.sort(key=lambda t: abs(t.avg_change_pct), reverse=True)
    return trends
