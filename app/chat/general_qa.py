from __future__ import annotations

from app.config import get_settings, require_openai_key
from app.observability.tracing import openai_client
from app.reasoning.prompts import GENERAL_QA_SYSTEM


def answer_general(message: str) -> str:
    require_openai_key()
    settings = get_settings()
    client = openai_client()
    resp = client.chat.completions.create(
        model=settings.general_qa_model,
        temperature=settings.general_qa_temperature,
        max_completion_tokens=500,
        messages=[
            {"role": "system", "content": GENERAL_QA_SYSTEM},
            {"role": "user", "content": message},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def portfolio_reply_from_briefing(briefing) -> str:
    """Convert a structured Briefing into a conversational single-message reply."""
    lines = [f"**{briefing.headline}**", "", briefing.summary]
    if briefing.causal_links:
        lines.append("\n**What drove it:**")
        for link in briefing.causal_links:
            impact = f"{link.portfolio_impact_pct:+.2f}%"
            stocks = ", ".join(link.stocks_affected) if link.stocks_affected else "—"
            lines.append(
                f"- [{link.news_id}] **{link.driver}** ({link.sector}, impact ≈ {impact}) — "
                f"affected: {stocks}. {link.explanation}"
            )
    if briefing.conflicts:
        lines.append("\n**Conflicts / ambiguities:**")
        for c in briefing.conflicts:
            lines.append(f"- {c}")
    lines.append(f"\n_Confidence: {briefing.confidence:.2f} · model: {briefing.model}_")
    return "\n".join(lines)
