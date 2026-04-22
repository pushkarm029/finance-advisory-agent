from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AssetType = Literal["stock", "mutual_fund"]
Sentiment = Literal["positive", "negative", "neutral"]
Scope = Literal["market", "sector", "stock"]
MarketSentiment = Literal["bullish", "bearish", "neutral"]


class IndexQuote(BaseModel):
    symbol: str
    previous_close: float
    close: float
    change_pct: float


class StockQuote(BaseModel):
    symbol: str
    sector: str
    previous_close: float
    close: float
    change_pct: float


class FundQuote(BaseModel):
    symbol: str
    asset_type: str
    previous_nav: float
    nav: float
    change_pct: float


class MarketSnapshot(BaseModel):
    date: str
    indices: dict[str, IndexQuote]
    stocks: dict[str, StockQuote]
    mutual_funds: dict[str, FundQuote]


class NewsArticle(BaseModel):
    id: str
    headline: str
    summary: str
    entities: dict[str, list[str]] = Field(default_factory=dict)
    scope: Scope
    sentiment: Sentiment


class Holding(BaseModel):
    symbol: str
    type: AssetType
    quantity: float
    avg_price: float


class Portfolio(BaseModel):
    id: str
    name: str
    type: str
    holdings: list[Holding]


class HoldingPnL(BaseModel):
    symbol: str
    type: AssetType
    sector: str | None
    quantity: float
    avg_price: float
    current_price: float
    invested: float
    current_value: float
    day_change_abs: float
    day_change_pct: float
    overall_pnl_abs: float
    overall_pnl_pct: float


class SectorExposure(BaseModel):
    sector: str
    value: float
    weight_pct: float


class RiskFlag(BaseModel):
    kind: str
    message: str
    severity: Literal["low", "medium", "high"]


class PortfolioAnalytics(BaseModel):
    portfolio_id: str
    portfolio_name: str
    invested_total: float
    current_value: float
    day_pnl_abs: float
    day_pnl_pct: float
    overall_pnl_abs: float
    overall_pnl_pct: float
    holdings: list[HoldingPnL]
    sector_exposure: list[SectorExposure]
    asset_type_exposure: dict[str, float]
    risk_flags: list[RiskFlag]
    top_contributors: list[HoldingPnL]
    top_detractors: list[HoldingPnL]


class SectorTrend(BaseModel):
    sector: str
    avg_change_pct: float
    sentiment: MarketSentiment
    constituents: list[str]


class MarketContext(BaseModel):
    date: str
    market_sentiment: MarketSentiment
    indices: list[IndexQuote]
    sector_trends: list[SectorTrend]
    classified_news: list[NewsArticle]


class CausalLink(BaseModel):
    news_id: str | None = None
    driver: str
    sector: str | None = None
    stocks_affected: list[str] = Field(default_factory=list)
    portfolio_impact_pct: float | None = None
    explanation: str


class Briefing(BaseModel):
    headline: str
    summary: str
    causal_links: list[CausalLink]
    conflicts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    model: str


class EvaluationScore(BaseModel):
    reasoning_quality: int = Field(ge=0, le=5)
    causal_depth: int = Field(ge=0, le=5)
    factual_grounding: int = Field(ge=0, le=5)
    overall: int = Field(ge=0, le=5)
    notes: str


class AgentResponse(BaseModel):
    portfolio: PortfolioAnalytics
    market: MarketContext
    briefing: Briefing
    evaluation: EvaluationScore
    trace_id: str | None = None
