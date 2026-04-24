from __future__ import annotations

from uuid import uuid4

import streamlit as st

from app.analytics.portfolio import load_portfolios
from app.chat.general_qa import answer_general, portfolio_reply_from_briefing
from app.chat.router import route_intent
from app.config import get_settings
from app.models import AgentResponse
from app.orchestrator import run_agent


st.set_page_config(
    page_title="Autonomous Financial Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    st.session_state.setdefault("session_id", str(uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)


def _submit_example(text: str) -> None:
    st.session_state.pending_prompt = text


def _render_agent_details(resp: AgentResponse) -> None:
    with st.expander("Briefing details · analytics · evaluation · trace"):
        tabs = st.tabs(["Briefing", "Analytics", "Evaluation", "Market", "Trace"])
        with tabs[0]:
            st.json(resp.briefing.model_dump())
        with tabs[1]:
            p = resp.portfolio
            c1, c2, c3 = st.columns(3)
            c1.metric("Day P&L", f"{p.day_pnl_abs:+,.2f}", f"{p.day_pnl_pct:+.2f}%")
            c2.metric("Current Value", f"{p.current_value:,.2f}")
            c3.metric("Overall P&L", f"{p.overall_pnl_abs:+,.2f}", f"{p.overall_pnl_pct:+.2f}%")
            if p.risk_flags:
                st.markdown("**Risk flags**")
                for f in p.risk_flags:
                    st.markdown(f"- `[{f.severity.upper()}]` **{f.kind}** — {f.message}")
            if p.detected_conflicts:
                st.markdown("**Pre-detected conflicts**")
                for c in p.detected_conflicts:
                    st.markdown(
                        f"- **{c.stock}** ({c.day_change_pct:+.2f}%) vs `{c.news_id}` "
                        f"[{c.news_sentiment}] — {c.note}"
                    )
            st.markdown("**Full analytics**")
            st.json(p.model_dump())
        with tabs[2]:
            e = resp.evaluation
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reasoning", f"{e.reasoning_quality}/5")
            c2.metric("Causal Depth", f"{e.causal_depth}/5")
            c3.metric("Grounding", f"{e.factual_grounding}/5")
            c4.metric("Overall", f"{e.overall}/5")
            st.caption(e.notes)
            st.markdown("**Structural check**")
            st.json(e.structural.model_dump())
            if e.llm is not None:
                st.markdown("**LLM grader**")
                st.json(e.llm.model_dump())
        with tabs[3]:
            st.json(resp.market.model_dump())
        with tabs[4]:
            st.code(f"trace_id: {resp.trace_id}", language="text")
            if resp.trace_url:
                st.markdown(f"[Open in Langfuse ↗]({resp.trace_url})")
            else:
                st.caption("Langfuse tracing disabled (set LANGFUSE_PUBLIC_KEY / SECRET_KEY).")


@st.cache_data
def _cached_portfolios():
    return load_portfolios()


def _sidebar() -> str | None:
    with st.sidebar:
        st.title("📊 Advisor")
        st.caption("Reasoning-first portfolio agent.")

        portfolios = _cached_portfolios()
        options = ["(no portfolio — general Q&A only)"] + [
            f"{p.id} — {p.name}" for p in portfolios
        ]
        selected = st.selectbox("Active portfolio", options, index=0)
        portfolio_id = None if selected.startswith("(no portfolio") else selected.split(" ")[0]

        st.markdown("---")
        st.markdown("**Try it**")
        st.button(
            "Brief me on today",
            on_click=_submit_example,
            args=("Brief me on today. What drove the move?",),
            use_container_width=True,
        )
        st.button(
            "Biggest risk in this portfolio?",
            on_click=_submit_example,
            args=("What is the biggest risk flag in this portfolio right now?",),
            use_container_width=True,
        )
        st.button(
            "Explain: concentration risk",
            on_click=_submit_example,
            args=("What is concentration risk and why does it matter?",),
            use_container_width=True,
        )
        st.button(
            "What is a liquid fund?",
            on_click=_submit_example,
            args=("What is a liquid fund and when would a retiree use one?",),
            use_container_width=True,
        )

        st.markdown("---")
        settings = get_settings()
        st.caption("**Runtime**")
        st.caption(f"Briefing · `{settings.briefing_model}`")
        st.caption(f"Eval · `{settings.eval_model}`")
        st.caption(f"General · `{settings.general_qa_model}`")
        st.caption(
            f"Langfuse · {'🟢 on' if settings.langfuse_enabled else '⚪ off'}"
        )
        if not settings.openai_api_key:
            st.error("OPENAI_API_KEY not set — agent disabled.", icon="🚫")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid4())
            st.rerun()

    return portfolio_id


def _render_history() -> None:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("agent_response") is not None:
                _render_agent_details(m["agent_response"])


def _handle_prompt(prompt: str, portfolio_id: str | None) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = route_intent(prompt, portfolio_id)
    with st.chat_message("assistant"):
        badge = "🧭 portfolio" if intent == "portfolio" else "📚 general"
        st.caption(f"routed · {badge}")
        try:
            with st.spinner("Reasoning…"):
                if intent == "portfolio":
                    if not portfolio_id:
                        reply = (
                            "Please select a portfolio from the sidebar to get a "
                            "portfolio-scoped briefing."
                        )
                        st.markdown(reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reply}
                        )
                        return
                    agent_resp = run_agent(portfolio_id)
                    reply = portfolio_reply_from_briefing(agent_resp.briefing)
                    st.markdown(reply)
                    _render_agent_details(agent_resp)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "agent_response": agent_resp,
                        }
                    )
                else:
                    reply = answer_general(prompt)
                    st.markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
        except Exception as exc:
            err = f"❌ **Error:** {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


def main() -> None:
    _init_state()
    portfolio_id = _sidebar()

    st.title("Autonomous Financial Advisor")
    st.caption(
        "Ingests market data + news, computes deterministic analytics, and produces a "
        "causal briefing linking News → Sector → Stock → Portfolio Impact. "
        "Select a portfolio on the left, or ask any finance question."
    )

    _render_history()

    pending = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("Ask about your portfolio or a finance concept…")
    prompt = typed or pending
    if prompt:
        _handle_prompt(prompt, portfolio_id)


if __name__ == "__main__":
    main()
