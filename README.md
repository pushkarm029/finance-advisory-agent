# Autonomous Financial Advisor Agent

A reasoning-first portfolio agent. It ingests market data and news, computes deterministic portfolio analytics, and uses an LLM (OpenAI `gpt-4o-mini` by default) to produce a **causal** briefing that links `News -> Sector -> Stock -> Portfolio Impact` for a specific investor.

## Architecture

```
┌───────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│ Market Ingestion  │     │ Portfolio Analytics│     │ Reasoning (4o-mini)  │
│  - indices         │     │  - P&L (day/total) │     │  - causal linking    │
│  - sector trends   │─▶──▶│  - sector weights  │─▶──▶│  - conflict notes    │
│  - news classify   │     │  - risk flags      │     │  - confidence score  │
└───────────────────┘     └────────────────────┘     └──────────┬───────────┘
         ▲                           ▲                          │
         │                           │                          ▼
         │                           │                 ┌──────────────────┐
         │                           │                 │ Self-Evaluation  │
         │                           │                 │  (LLM / rules)   │
         │                           │                 └──────────────────┘
         │                           │                          │
         └────────── Langfuse tracing wraps every step ─────────┘
```

Each phase is a separate module in `app/`:

| Phase | Module | LLM calls |
|---|---|---|
| 1. Market Intelligence | `app/ingestion/` | 0 |
| 2. Portfolio Analytics | `app/analytics/` | 0 |
| 3. Reasoning | `app/reasoning/` | 1 |
| 4. Self-Evaluation | `app/evaluation/` | 1 |
| Observability | `app/observability/` | - |
| Orchestration | `app/orchestrator.py` | - |
| API | `app/api.py` | - |

News is classified at ingestion (scope + sentiment, rule-based) so the reasoning LLM receives pre-filtered, pre-tagged items — no redundant LLM calls per article. Portfolio analytics are fully deterministic. **Two LLM calls total per request**: one for the briefing, one for self-evaluation.

## Setup

Requires Python 3.10+.

```bash
# from the project root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in OPENAI_API_KEY (Langfuse keys optional)
```

## Run

### CLI (simplest)

```bash
# interactive (prompts for portfolio id)
python -m scripts.run_cli

# single portfolio
python -m scripts.run_cli P001

# all three portfolios in sequence
python -m scripts.run_cli --all

# machine-readable JSON
python -m scripts.run_cli --json P002
```

### FastAPI

```bash
python -m app.main
# or: uvicorn app.api:app --reload
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/portfolios` | list the three mock portfolios |
| GET | `/market` | raw market snapshot |
| GET | `/news` | classified news (scope + sentiment) |
| GET | `/analyze/{portfolio_id}` | **full agent briefing + evaluation** |

Swagger docs: `http://localhost:8000/docs`.

## Mock Portfolios

| ID | Name | Profile |
|---|---|---|
| P001 | Diversified Growth | 8 stocks across 6 sectors + 2 MFs |
| P002 | Banking Concentrated | ~89% Banking exposure (triggers concentration risk) |
| P003 | Conservative Retiree | MF-heavy (liquid + gilt + bluechip) with small equity tail |

## Observability

- **Structured logs** are emitted for every trace start/end and every LLM generation to stdout.
- **Langfuse** traces fire automatically when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set. Without them, the agent runs identically and logs to stdout.
- Each `/analyze` response includes a `trace_id` you can search for in Langfuse.

## Reasoning Quality

The agent is constrained to:
- cite a `news_id` on every causal link (prevents hallucinated drivers)
- surface **conflicts** when a stock moved opposite to the news sentiment (e.g. HDFC Bank in-line earnings but cautious guidance → price drop)
- prioritise holdings by absolute P&L contribution, not just percentage
- emit a `confidence` score reflecting how well news explains observed moves

## Self-Evaluation

A second LLM call scores the briefing on four dimensions (0–5):
`reasoning_quality`, `causal_depth`, `factual_grounding`, `overall`. Falls back to a rule-based scorer if the API call fails or the key is absent.

## Project Layout

```
app/
  config.py              # env loading, settings
  models.py              # pydantic types shared across layers
  data/                  # fabricated market, news, portfolio JSON
  ingestion/
    market.py            # indices, sector trends, market sentiment
    news.py              # scope + sentiment classification, relevance filter
  analytics/
    portfolio.py         # P&L, exposures, risk flags, contributors/detractors
  reasoning/
    prompts.py           # system prompts + schema contracts
    agent.py             # single LLM call that produces the briefing
  evaluation/
    evaluator.py         # LLM-based grader with deterministic fallback
  observability/
    tracing.py           # Langfuse wrapper that no-ops without keys
  orchestrator.py        # wires all four phases together
  api.py                 # FastAPI app
  main.py                # uvicorn entry
scripts/
  run_cli.py             # interactive / batch CLI
```
