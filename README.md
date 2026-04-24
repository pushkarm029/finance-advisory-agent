# Autonomous Financial Advisor — Chat Agent

A reasoning-first portfolio agent with a chat interface. It ingests market data and
news, computes deterministic portfolio analytics, detects sentiment-vs-price conflicts
programmatically, and uses an LLM to produce a **causal** briefing that links
**News → Sector → Stock → Portfolio Impact** for a specific investor. It also answers
general finance questions.

> **Live demo:** _add deployed URL here after Render deploy_
>
> **Walkthrough video:** _add 2–3 min Loom/YouTube link here_

---

## Architecture

```
┌──────────────────────┐   ┌────────────────────────┐   ┌────────────────────┐
│ Market Ingestion     │   │ Portfolio Analytics    │   │ Reasoning (1 LLM)  │
│  • indices           │   │  • P&L absolute + %    │   │  • causal linking  │
│  • sector trends     │──▶│  • sector + MF exposure│──▶│  • narrates        │
│  • news classify     │   │  • concentration risks │   │    pre-detected    │
│    (scope+sentiment) │   │  • CONFLICT DETECTOR   │   │    conflicts       │
└──────────────────────┘   └────────────────────────┘   └─────────┬──────────┘
            │                          │                          │
            │                          │                          ▼
            │                          │            ┌────────────────────────┐
            │                          │            │ Evaluation             │
            │                          │            │  • STRUCTURAL (always) │
            │                          │            │  • LLM grader (opt.)   │
            │                          │            │  • overall = min(both) │
            │                          │            └────────────────────────┘
            │                          │                          │
            └──── Langfuse tracing wraps every step ──────────────┘
```

### Modules

| Phase | Module | LLM calls |
|---|---|---|
| 1. Market Intelligence | `app/ingestion/` | 0 |
| 2. Portfolio Analytics + Conflict Detector | `app/analytics/` | 0 |
| 3. Reasoning Briefing | `app/reasoning/` | 1 |
| 4. Self-Evaluation (structural + LLM) | `app/evaluation/` | 1 |
| Chat (intent router + general Q&A) | `app/chat/` | 0–1 |
| Observability | `app/observability/` | — |
| Orchestration | `app/orchestrator.py` | — |
| API | `app/api.py` | — |
| Streamlit UI | `streamlit_app.py` | — |

**Two LLM calls per portfolio briefing**: reasoning + self-evaluation. General Q&A is one
extra LLM call; zero LLM calls for non-portfolio UI interactions. News is classified
rule-based at ingestion (no per-article LLM calls).

### Why this is not just a "data dump"

1. **Deterministic pre-ranking** — only the top-3 contributors/detractors (by |day P&L|)
   are sent to the LLM, so small noise can't bury the big movers.
2. **Deterministic conflict detector** — `app/analytics/portfolio.py::detect_conflicts`
   scans every holding's price direction against matched news sentiment. The LLM receives
   `PRE_DETECTED_CONFLICTS` and is told to narrate them, not discover them.
3. **Post-parse validator** — every `causal_link` returned by the LLM must cite a
   `news_id` present in the filtered news list, and at least one stock actually in the
   portfolio. Invalid links are dropped; the drop list is logged.
4. **Derived confidence** — not self-reported. Computed as
   `Σ|day_pnl_abs| of cited top movers / Σ|day_pnl_abs| of all top movers`, clamped to [0,1].
5. **Structural self-eval runs every time** — measures `valid_citation_ratio`,
   `impact_sum_gap_pct`, `movers_coverage_ratio` against ground truth. Final `overall`
   is `min(structural_score, llm_score)` so the LLM grader can't rubber-stamp broken output.

---

## Setup

Requires Python 3.10+ (tested on 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: fill OPENAI_API_KEY. Langfuse keys optional.
```

---

## Run

### Streamlit chat UI (primary surface)

```bash
streamlit run streamlit_app.py
# opens http://localhost:8501
```

Chat interface with portfolio picker sidebar, example prompts, briefing + analytics +
evaluation + Langfuse trace link per response.

### CLI

```bash
python -m scripts.run_cli                # interactive prompt
python -m scripts.run_cli P001           # single portfolio
python -m scripts.run_cli --all          # all three portfolios
python -m scripts.run_cli --json P002    # machine-readable JSON on stdout (logs go to stderr)
```

### FastAPI (programmatic REST access)

```bash
python -m app.main
# or: uvicorn app.api:app --reload
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | readiness (checks OPENAI_API_KEY) |
| GET | `/portfolios` | list mock portfolios |
| GET | `/market` | raw market snapshot |
| GET | `/news` | classified news |
| GET | `/analyze/{portfolio_id}` | full briefing + evaluation |
| POST | `/chat` | `{message, portfolio_id?, session_id}` → routed reply |

Swagger at `/docs`.

---

## Mock Portfolios

| ID | Name | Profile |
|---|---|---|
| P001 | Diversified Growth | 8 stocks across 6 sectors + 2 MFs |
| P002 | Banking Concentrated | ~78% Banking exposure → concentration flag |
| P003 | Conservative Retiree | MF-heavy (liquid + gilt + largecap + flexicap) + small equity tail |

---

## Observability

- **Langfuse** traces every request when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are
  set. OpenAI calls are auto-instrumented via `langfuse.openai` so generations nest
  cleanly under the parent trace span. Each response returns a `trace_url` you can open.
- **Structured logs** go to stderr (JSON format when `ENV=prod`, human-readable otherwise).
  CLI `--json` output goes to stdout — logs on stderr never contaminate it.

---

## Environment Variables

| Var | Default | Required |
|---|---|---|
| `OPENAI_API_KEY` | — | yes |
| `BRIEFING_MODEL` | `gpt-4o-mini` | |
| `EVAL_MODEL` | `gpt-4o-mini` | |
| `GENERAL_QA_MODEL` | `gpt-4o-mini` | |
| `INTENT_MODEL` | `gpt-4o-mini` | |
| `BRIEFING_TEMPERATURE` | `0.2` | |
| `EVAL_TEMPERATURE` | `0.0` | |
| `GENERAL_QA_TEMPERATURE` | `0.4` | |
| `CONCENTRATION_THRESHOLD_PCT` | `40.0` | |
| `LANGFUSE_PUBLIC_KEY` | — | no (disables tracing if absent) |
| `LANGFUSE_SECRET_KEY` | — | no |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | |
| `ENV` | `dev` | set `prod` for JSON logs |
| `PORT` | `8000` | respected by Streamlit/FastAPI |
| `ALLOWED_ORIGINS` | `*` | comma-separated, for CORS |

---

## Tests

```bash
pytest tests/ -v
```

19 tests covering: concentration threshold strictness, missing-symbol tolerance, MF
category surfacing, conflict detector (with/without neutral news), P&L derivation,
news scope filtering, structural evaluator scoring, and intent routing. None require
an API key.

---

## Deploy

### Docker

```bash
docker build -t fin-advisor .
docker run -p 8000:8000 --env-file .env fin-advisor
```

### Render

`render.yaml` is configured for a free Docker web service. Push to a GitHub repo,
connect it to Render, and set `OPENAI_API_KEY` + `LANGFUSE_*` as service env vars
(marked `sync: false`). Health check hits `/_stcore/health` (Streamlit built-in).

---

## Project Layout

```
app/
  config.py              # env-driven Settings dataclass, fail-fast key check
  models.py              # pydantic types (incl. strict BriefingDraft for Structured Outputs)
  data/                  # mock market / news / portfolios JSON
  ingestion/
    market.py            # indices, dynamic sector trends, market sentiment
    news.py              # rule-based scope + sentiment classification
  analytics/
    portfolio.py         # P&L, exposures, risk flags, detect_conflicts()
  reasoning/
    prompts.py           # system prompts + user prompt builders
    agent.py             # LLM call via Structured Outputs, validator, derived confidence
  evaluation/
    evaluator.py         # structural (always) + LLM grader, overall=min()
  chat/
    router.py            # rule-based intent classifier
    general_qa.py        # finance-literacy Q&A + briefing→markdown formatter
  observability/
    tracing.py           # Langfuse 4.x wrapper + structured logger
  orchestrator.py        # wires ingestion → analytics → reasoning → eval
  api.py                 # FastAPI: /health /ready /portfolios /market /news /analyze /chat
  main.py                # uvicorn entrypoint (respects $PORT)
scripts/
  run_cli.py             # CLI (--all, --json, portfolio_id)
streamlit_app.py         # Chat UI — deployed surface
tests/                   # 19 unit tests, no API key needed
Dockerfile               # multi-stage, non-root, runs Streamlit on $PORT
render.yaml              # Render web service config
```
