"""
Tests for OpenRouter credential-expiry detection (main.py's _llm) and its
propagation all the way through ranking, the report banner, and the
Telegram bot's completion message -- not just that the exception fires,
but that it degrades gracefully into a clearly-labeled report instead of
a wall of generic UNRANKED/RANKING_FAILED noise, and never crashes the run.

Run directly: python tests/test_openrouter_credential_expired.py
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import openai

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import tools.telegram_bot as bot
from tools.job_apis import CredentialExpiredError


def _real_shaped_auth_error() -> openai.AuthenticationError:
    """Builds the exact exception shape confirmed live (2026-08-18) against
    a deliberately invalid OpenRouter key, through the real openai SDK
    call path this project actually uses."""
    resp = httpx.Response(
        401,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        json={"error": {"message": "User not found.", "code": 401}},
    )
    return openai.AuthenticationError(
        "Error code: 401", response=resp, body={"message": "User not found.", "code": 401}
    )


# ── 1. _llm() itself raises the right, distinctly-typed exception ────────


def test_llm_raises_credential_expired_on_real_shaped_401():
    with patch.object(
        main._sync_or_client.chat.completions, "create",
        side_effect=_real_shaped_auth_error(),
    ):
        try:
            main._llm("test prompt")
            assert False, "expected CredentialExpiredError"
        except CredentialExpiredError as e:
            assert e.confirmed is True
            assert e.source == "OpenRouter"
            print(f"PASS: _llm raises CredentialExpiredError on real-shaped 401: {e}")


def test_llm_does_not_burn_through_fallback_chain_on_auth_error():
    # A credential failure is guaranteed to repeat identically on every
    # fallback model (same OPENROUTER_API_KEY) -- confirm we fail fast on
    # the FIRST attempt rather than retrying N-1 more times uselessly.
    call_count = {"n": 0}

    def _side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise _real_shaped_auth_error()

    with patch.object(main._sync_or_client.chat.completions, "create", side_effect=_side_effect):
        try:
            main._llm("test prompt")
        except CredentialExpiredError:
            pass
    assert call_count["n"] == 1, f"expected exactly 1 call (fail fast), got {call_count['n']}"
    print("PASS: fails fast on the first model, doesn't burn through the fallback chain")


# ── 2. Propagates cleanly through ranking -- no crash, clear labeling ────


def test_write_opportunities_report_degrades_gracefully_on_openrouter_failure():
    cv_terms = {
        "candidate_name": "Test Candidate",
        "seniority_level": "intern",
        "job_titles": ["AI Engineer Intern"],
        "skills_technical": ["Python"],
    }
    opportunities = {
        "total_results": 2,
        "queries_executed": 1,
        "opportunities": [
            {"title": "Real Job 1", "company": "Acme", "source_url": "https://example.com/1", "platform": "unknown"},
            {"title": "Real Job 2", "company": "Beta", "source_url": "https://example.com/2", "platform": "unknown"},
        ],
    }

    original_llm = main._llm

    def _always_dead(*args, **kwargs):
        raise CredentialExpiredError("OpenRouter", "API key rejected (confirmed): fake", confirmed=True)

    main._llm = _always_dead
    try:
        result_str = main.write_opportunities_report(
            json.dumps(cv_terms), json.dumps(opportunities)
        )
    finally:
        main._llm = original_llm

    print("=== RETURNED STRING ===")
    print(result_str)

    assert "SOURCE_UNAVAILABLE" in result_str
    assert "OpenRouter" in result_str

    md_m = re.search(r"Markdown\s*:\s*(.+\.md)", result_str)
    json_m = re.search(r"JSON\s*:\s*(.+\.json)", result_str)
    assert md_m and json_m
    md_content = Path(md_m.group(1).strip()).read_text(encoding="utf-8")
    report = json.loads(Path(json_m.group(1).strip()).read_text(encoding="utf-8"))

    # Both jobs must survive as UNRANKED with the SPECIFIC OpenRouter
    # reason, not generic "parse failure" noise.
    assert len(report["ranked_opportunities"]) == 2
    for r in report["ranked_opportunities"]:
        assert r["tier"] == "UNRANKED"
        assert "OpenRouter" in r["concerns"][0]
    print("PASS: both jobs UNRANKED with distinct OpenRouter-specific reason, not generic noise")

    sources = report["meta"]["sources_unavailable"]
    assert any(s["source"] == "OpenRouter" and s["kind"] == "CREDENTIAL EXPIRED" for s in sources)
    print(f"PASS: sources_unavailable in JSON meta correctly includes OpenRouter: {sources}")

    assert "SOURCE(S) UNAVAILABLE" in md_content and "OpenRouter" in md_content
    print("PASS: rendered Markdown banner names OpenRouter")


# ── 3. run_pipeline's Step 1 (CV extraction) fails loud, not crash ───────


def test_run_pipeline_step1_credential_failure_returns_clean_dict_not_crash():
    import asyncio

    def _always_dead(cv_path):
        raise CredentialExpiredError("OpenRouter", "API key rejected (confirmed): fake", confirmed=True)

    with patch.object(main, "analyze_cv_and_extract_search_terms", side_effect=_always_dead), \
         patch("pathlib.Path.exists", return_value=True):
        result = asyncio.run(main.run_pipeline("/fake/cv.pdf"))

    assert result["total_jobs"] == 0
    assert len(result["sources_unavailable"]) == 1
    assert result["sources_unavailable"][0]["source"] == "OpenRouter"
    assert "OpenRouter" in result["output"]
    print(f"PASS: run_pipeline returns a clean dict on Step 1 credential failure, does not crash: {result['output']}")


# ── 4. Telegram bot surfaces sources_unavailable in the exact asked format ─


def test_telegram_completion_message_surfaces_sources_unavailable():
    sources = [
        {"source": "Adzuna", "kind": "CREDENTIAL_EXPIRED", "reason": "...", "confirmed": True},
        {"source": "OpenRouter", "kind": "CREDENTIAL_EXPIRED", "reason": "...", "confirmed": True},
    ]
    line = bot._format_sources_unavailable(sources)
    print(f"Formatted line: {line!r}")
    assert line.startswith("🚫 2 source(s) unavailable:")
    assert "Adzuna" in line and "credential expired" in line.lower()
    assert "OpenRouter" in line
    print("PASS: Telegram completion formatting matches the requested shape")

    # And confirm it's actually included when total_jobs == 0 (the Step-1
    # failure case never produces a report file, so this must come purely
    # from the result dict, not the JSON file).
    with patch.object(bot, "send_message") as mock_send:
        bot._send_completion(
            "fake-token", 123,
            {"total_jobs": 0, "new_jobs": 0, "output": "...", "sources_unavailable": sources},
        )
        sent_text = mock_send.call_args[0][2]
        assert "source(s) unavailable" in sent_text
        assert "OpenRouter" in sent_text
    print("PASS: _send_completion includes sources_unavailable even with no report file (total_jobs=0 path)")


if __name__ == "__main__":
    test_llm_raises_credential_expired_on_real_shaped_401()
    test_llm_does_not_burn_through_fallback_chain_on_auth_error()
    test_write_opportunities_report_degrades_gracefully_on_openrouter_failure()
    test_run_pipeline_step1_credential_failure_returns_clean_dict_not_crash()
    test_telegram_completion_message_surfaces_sources_unavailable()
    print("\nALL PASS")
