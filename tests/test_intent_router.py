from __future__ import annotations

from app.chat.router import route_intent


def test_no_portfolio_means_general():
    assert route_intent("Brief me on today", portfolio_id=None) == "general"


def test_portfolio_id_mention_forces_portfolio():
    assert route_intent("tell me about P001", portfolio_id=None) == "portfolio"


def test_keywords_with_portfolio_route_to_portfolio():
    assert route_intent("why did my portfolio fall today?", portfolio_id="P001") == "portfolio"
    assert route_intent("what's my p&l", portfolio_id="P001") == "portfolio"


def test_general_question_with_portfolio_still_general():
    assert route_intent("what is a liquid fund?", portfolio_id="P001") == "general"
    assert route_intent("explain NIFTY index", portfolio_id="P001") == "general"
