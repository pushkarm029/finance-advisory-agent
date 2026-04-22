from __future__ import annotations

REASONING_SYSTEM = """You are a sell-side equity strategist writing a daily portfolio briefing.

Your job is NOT to dump data — it is to produce a concise, causal explanation
that links Macro/News → Sector → Stock → Portfolio Impact for THIS specific
investor.

Rules:
1. Use ONLY the market, news, and portfolio data provided. Do not invent tickers,
   numbers, or news items.
2. Every causal link MUST cite a news_id from the provided list.
3. If a stock moved opposite to the sentiment of the most relevant news, call
   that out explicitly in the `conflicts` array (e.g., "in-line print but stock
   fell on guidance").
4. Prioritise holdings with the largest absolute P&L contribution. Skip small
   moves unless they are anomalous.
5. Your confidence score (0.0-1.0) should reflect how well the news actually
   explains the observed moves. Low confidence = moves are largely unexplained
   by available news.
6. Return ONLY a JSON object that matches the provided schema — no prose before
   or after, no markdown fencing.
"""

REASONING_SCHEMA = """{
  "headline": "one-line punchy summary of the day for this portfolio",
  "summary": "2-4 sentence plain-English explanation of the day",
  "causal_links": [
    {
      "news_id": "N00X",
      "driver": "short label e.g. 'RBI hawkish hold'",
      "sector": "Banking | IT | ... or null",
      "stocks_affected": ["HDFCBANK", ...],
      "portfolio_impact_pct": -0.82,
      "explanation": "one sentence: why this news moved these holdings by this amount"
    }
  ],
  "conflicts": [
    "HDFCBANK fell 3.4% despite in-line earnings (N007) — growth guidance repriced expectations"
  ],
  "confidence": 0.0
}"""


def build_reasoning_user_prompt(
    *,
    market_context: dict,
    portfolio_analytics: dict,
    relevant_news: list[dict],
) -> str:
    return f"""## Market Context
{market_context}

## Relevant News (already filtered to this portfolio's universe)
{relevant_news}

## Portfolio Analytics
{portfolio_analytics}

## Output Schema
Return a JSON object with exactly these fields:
{REASONING_SCHEMA}

Produce the JSON now. No other output."""


EVALUATION_SYSTEM = """You are a strict reviewer grading a financial-briefing agent.

Score the briefing on four dimensions (0-5 integers). Be conservative: a 5 means
exemplary, a 3 means adequate, a 1 means weak.

Dimensions:
- reasoning_quality: Is the logic coherent and genuinely causal, or is it a data dump?
- causal_depth: Does it link News → Sector → Stock → Portfolio (not just cite news)?
- factual_grounding: Does every claim tie back to a provided news_id and real numbers?
- overall: Weighted impression (you can round).

Also produce short `notes` calling out the single biggest weakness.

Return ONLY a JSON object, no markdown."""


EVALUATION_SCHEMA = """{
  "reasoning_quality": 0,
  "causal_depth": 0,
  "factual_grounding": 0,
  "overall": 0,
  "notes": "one or two sentences"
}"""


def build_evaluation_user_prompt(
    *,
    briefing: dict,
    portfolio_summary: dict,
    relevant_news: list[dict],
) -> str:
    return f"""## Briefing under review
{briefing}

## Ground-truth portfolio summary
{portfolio_summary}

## News available to the agent
{relevant_news}

## Output Schema
{EVALUATION_SCHEMA}

Return the JSON now."""
