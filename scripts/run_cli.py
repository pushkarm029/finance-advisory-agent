"""Simple CLI loop for running the agent against each mock portfolio.

Usage:
    python -m scripts.run_cli            # prompts for a portfolio ID
    python -m scripts.run_cli P001       # runs a single portfolio
    python -m scripts.run_cli --all      # runs all three portfolios
"""
from __future__ import annotations

import json
import sys
from typing import Iterable

from app.analytics.portfolio import load_portfolios
from app.models import AgentResponse
from app.orchestrator import run_agent


def _print_briefing(resp: AgentResponse) -> None:
    p = resp.portfolio
    b = resp.briefing
    e = resp.evaluation

    print("\n" + "=" * 78)
    print(f"Portfolio: {p.portfolio_name}  ({p.portfolio_id})")
    print(
        f"Day P&L: {p.day_pnl_abs:+,.2f} ({p.day_pnl_pct:+.2f}%)  |  "
        f"Overall P&L: {p.overall_pnl_abs:+,.2f} ({p.overall_pnl_pct:+.2f}%)"
    )
    print(f"Current value: {p.current_value:,.2f}  |  Invested: {p.invested_total:,.2f}")
    print("-" * 78)

    print(f"\n[BRIEFING]  (confidence: {b.confidence:.2f}, model: {b.model})")
    print(f"  Headline: {b.headline}")
    print(f"  Summary:  {b.summary}")

    if b.causal_links:
        print("\n  Causal Links:")
        for link in b.causal_links:
            news_id = link.get("news_id") if isinstance(link, dict) else link.news_id
            driver = link.get("driver") if isinstance(link, dict) else link.driver
            sector = link.get("sector") if isinstance(link, dict) else link.sector
            stocks = link.get("stocks_affected") if isinstance(link, dict) else link.stocks_affected
            impact = link.get("portfolio_impact_pct") if isinstance(link, dict) else link.portfolio_impact_pct
            expl = link.get("explanation") if isinstance(link, dict) else link.explanation
            impact_str = f"{impact:+.2f}%" if impact is not None else "n/a"
            print(
                f"    - [{news_id}] {driver}  |  sector={sector}  |  "
                f"stocks={stocks}  |  impact={impact_str}"
            )
            print(f"        {expl}")

    if b.conflicts:
        print("\n  Conflicts / Ambiguities:")
        for c in b.conflicts:
            print(f"    ! {c}")

    if p.risk_flags:
        print("\n  Risk Flags:")
        for flag in p.risk_flags:
            print(f"    [{flag.severity.upper()}] {flag.kind}: {flag.message}")

    print("\n[EVALUATION]")
    print(
        f"  reasoning_quality={e.reasoning_quality}/5  "
        f"causal_depth={e.causal_depth}/5  "
        f"factual_grounding={e.factual_grounding}/5  "
        f"overall={e.overall}/5"
    )
    print(f"  Notes: {e.notes}")
    if resp.trace_id:
        print(f"\n  trace_id: {resp.trace_id}")
    print("=" * 78)


def _run_ids(ids: Iterable[str]) -> None:
    for pid in ids:
        try:
            resp = run_agent(pid)
            _print_briefing(resp)
        except Exception as exc:
            print(f"\n[ERROR] portfolio {pid}: {exc}", file=sys.stderr)


def main() -> None:
    args = sys.argv[1:]
    portfolios = load_portfolios()
    ids = [p.id for p in portfolios]

    if args == ["--all"]:
        _run_ids(ids)
        return

    if args and args[0] in ids:
        _run_ids([args[0]])
        return

    if args and args[0] == "--json" and len(args) >= 2 and args[1] in ids:
        resp = run_agent(args[1])
        print(json.dumps(resp.model_dump(), indent=2, default=str))
        return

    print("Available portfolios:")
    for p in portfolios:
        print(f"  {p.id}  — {p.name}  ({p.type}, {len(p.holdings)} holdings)")
    try:
        pid = input("\nEnter portfolio id: ").strip()
    except EOFError:
        return
    if pid:
        _run_ids([pid])


if __name__ == "__main__":
    main()
