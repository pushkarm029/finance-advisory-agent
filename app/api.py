from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.analytics.portfolio import load_portfolios
from app.ingestion.market import load_market_snapshot
from app.ingestion.news import load_and_classify_news
from app.models import AgentResponse
from app.orchestrator import run_agent

app = FastAPI(
    title="Autonomous Financial Advisor Agent",
    description=(
        "Reasons through market data and news to explain how external events "
        "impacted a specific user's portfolio."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/portfolios")
def list_portfolios() -> list[dict]:
    return [
        {"id": p.id, "name": p.name, "type": p.type, "holdings_count": len(p.holdings)}
        for p in load_portfolios()
    ]


@app.get("/market")
def market_snapshot() -> dict:
    snap = load_market_snapshot()
    return snap.model_dump()


@app.get("/news")
def classified_news() -> list[dict]:
    return [a.model_dump() for a in load_and_classify_news()]


@app.get("/analyze/{portfolio_id}", response_model=AgentResponse)
def analyze(portfolio_id: str) -> AgentResponse:
    try:
        return run_agent(portfolio_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
