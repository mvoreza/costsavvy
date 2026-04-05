import importlib
import os

def reload_main_with_env(enabled: str, keywords: str):
    os.environ["INTENT_OVERRIDE_FORCE_PRICE_ENABLED"] = enabled
    os.environ["INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS"] = keywords

    # Ensure required env vars exist for import (main.py reads them at import time)
    os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    import app.main as main
    return importlib.reload(main)

def test_force_price_mode_when_cost_words_and_service_present(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    msg = "How much does a colonoscopy cost?"
    merged = {"service_query": "colonoscopy"}

    assert main.should_force_price_mode(msg, merged) is True

def test_does_not_force_when_no_service_query(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    msg = "How much does it cost?"
    merged = {"service_query": ""}

    assert main.should_force_price_mode(msg, merged) is False

def test_does_not_force_when_disabled(monkeypatch):
    main = reload_main_with_env("false", "cost,how much,price")

    msg = "How much does a colonoscopy cost?"
    merged = {"service_query": "colonoscopy"}

    assert main.should_force_price_mode(msg, merged) is False

def test_apply_override_changes_mode_to_price(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    intent = {"mode": "general"}
    merged = {"service_query": "colonoscopy"}
    updated = main.apply_intent_override_if_needed(intent, "How much does a colonoscopy cost?", merged, "sess-123")

    assert updated["mode"] == "price"
    assert updated["intent_overridden"] is True
    assert updated["override_reason"] == "cost_keyword_plus_service_query"

def test_apply_override_does_not_change_if_already_price(monkeypatch):
    main = reload_main_with_env("true", "cost,how much,price")

    intent = {"mode": "price"}
    merged = {"service_query": "colonoscopy"}
    updated = main.apply_intent_override_if_needed(intent, "How much does a colonoscopy cost?", merged, "sess-123")

    assert updated["mode"] == "price"
    assert updated["intent_overridden"] is True
