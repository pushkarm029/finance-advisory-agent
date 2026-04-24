from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.analytics.portfolio import load_portfolios
from app.chat.general_qa import answer_general, portfolio_reply_from_briefing
from app.chat.router import route_intent
from app.config import get_settings
from app.ingestion.market import load_market_snapshot
from app.ingestion.news import load_and_classify_news
from app.models import AgentResponse, ChatRequest, ChatResponse
from app.observability.tracing import trace_span
from app.orchestrator import run_agent

app = FastAPI(
    title="Autonomous Financial Advisor Agent",
    description=(
        "Reasons through market data and news to explain how external events "
        "impacted a user's portfolio, and answers general finance questions."
    ),
    version="0.2.0",
)

_allowed_origins = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return {
        "status": "ready",
        "briefing_model": settings.briefing_model,
        "langfuse_enabled": str(settings.langfuse_enabled).lower(),
    }


@app.get("/portfolios")
def list_portfolios() -> list[dict]:
    return [
        {"id": p.id, "name": p.name, "type": p.type, "holdings_count": len(p.holdings)}
        for p in load_portfolios()
    ]


@app.get("/market")
def market_snapshot() -> dict:
    return load_market_snapshot().model_dump()


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    intent = route_intent(req.message, req.portfolio_id)
    with trace_span(
        "chat.turn",
        intent=intent,
        portfolio_id=req.portfolio_id,
        session_id=req.session_id,
    ) as tr:
        if intent == "portfolio":
            if not req.portfolio_id:
                raise HTTPException(400, "portfolio_id is required for portfolio intent")
            try:
                agent_resp = run_agent(req.portfolio_id)
            except KeyError as exc:
                raise HTTPException(404, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            reply = portfolio_reply_from_briefing(agent_resp.briefing)
            return ChatResponse(
                reply=reply,
                intent="portfolio",
                portfolio_id=req.portfolio_id,
                agent_response=agent_resp,
                trace_id=tr.trace_id,
                trace_url=tr.trace_url,
            )
        try:
            reply = answer_general(req.message)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return ChatResponse(
            reply=reply,
            intent="general",
            trace_id=tr.trace_id,
            trace_url=tr.trace_url,
        )
