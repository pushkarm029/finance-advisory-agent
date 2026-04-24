"""CLI for running the agent against mock portfolios.

Usage:
    python -m scripts.run_cli                # prompt for a portfolio id
    python -m scripts.run_cli P001           # run one portfolio
    python -m scripts.run_cli --all          # run all three
    python -m scripts.run_cli --json P002    # machine-readable JSON to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from app.analytics.portfolio import load_portfolios
from app.models import AgentResponse
from app.orchestrator import run_agent


def _print_briefing(resp: AgentResponse) -> None:
    p, b, e = resp.portfolio, resp.briefing, resp.evaluation

    print("\n" + "=" * 78)
    print(f"Portfolio: {p.portfolio_name}  ({p.portfolio_id})")
    print(
        f"Day P&L: {p.day_pnl_abs:+,.2f} ({p.day_pnl_pct:+.2f}%)  |  "
        f"Overall P&L: {p.overall_pnl_abs:+,.2f} ({p.overall_pnl_pct:+.2f}%)"
    )
    print(f"Current value: {p.current_value:,.2f}  |  Invested: {p.invested_total:,.2f}")
    if p.missing_symbols:
        print(f"Missing symbols (skipped): {', '.join(p.missing_symbols)}")
    print("-" * 78)

    print(f"\n[BRIEFING]  (confidence: {b.confidence:.2f}, model: {b.model})")
    print(f"  Headline: {b.headline}")
    print(f"  Summary:  {b.summary}")

    if b.causal_links:
        print("\n  Causal Links:")
        for link in b.causal_links:
            stocks = ", ".join(link.stocks_affected) if link.stocks_affected else "—"
            print(
                f"    - [{link.news_id}] {link.driver}  |  sector={link.sector}  |  "
                f"stocks={stocks}  |  impact={link.portfolio_impact_pct:+.2f}%"
            )
            print(f"        {link.explanation}")

    if p.detected_conflicts:
        print("\n  Pre-detected Conflicts:")
        for c in p.detected_conflicts:
            print(f"    ! {c.stock} ({c.day_change_pct:+.2f}%) vs {c.news_id} [{c.news_sentiment}]: {c.note}")

    if b.conflicts:
        print("\n  LLM Conflict Narration:")
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
    print(
        f"  Structural: citation={e.structural.valid_citation_ratio:.2f} "
        f"impact_gap={e.structural.impact_sum_gap_pct:.2f}pp "
        f"movers_covered={e.structural.movers_coverage_ratio:.2f}"
    )
    print(f"\n  trace_id: {resp.trace_id}")
    if resp.trace_url:
        print(f"  trace_url: {resp.trace_url}")
    print("=" * 78)


def _run_ids(ids: Iterable[str]) -> None:
    for pid in ids:
        try:
            resp = run_agent(pid)
            _print_briefing(resp)
        except Exception as exc:
            print(f"\n[ERROR] portfolio {pid}: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the financial advisor agent.")
    parser.add_argument("portfolio_id", nargs="?", help="P001 | P002 | P003")
    parser.add_argument("--all", action="store_true", help="Run all portfolios")
    parser.add_argument("--json", action="store_true", help="Emit JSON on stdout")
    args = parser.parse_args()

    portfolios = load_portfolios()
    known_ids = [p.id for p in portfolios]

    if args.all:
        _run_ids(known_ids)
        return

    if args.portfolio_id:
        if args.portfolio_id not in known_ids:
            print(f"Unknown portfolio id {args.portfolio_id!r}", file=sys.stderr)
            sys.exit(2)
        if args.json:
            resp = run_agent(args.portfolio_id)
            print(json.dumps(resp.model_dump(), indent=2, default=str))
            return
        _run_ids([args.portfolio_id])
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
