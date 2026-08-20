"""
Tests for credential/quota error signaling across Adzuna and the three
Apify sources (tools/job_apis.py).

Credential-error tests mock the EXACT response shapes captured live
(2026-08-17):
    Adzuna: 401, {"exception": "AUTH_FAIL", ...}
    Apify:  401, {"error": {"type": "user-or-token-not-found", ...}}

Quota-error tests mock a PLAUSIBLE-BUT-UNVERIFIED shape (documented
behavior only, no confirmed wire format exists for either API's monthly
usage cap) -- these confirm the heuristic fires and is labeled
confirmed=False, not that the real API actually returns this exact body.

Run directly: python tests/test_source_unavailable_errors.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.job_apis as job_apis
from tools.job_apis import (
    CredentialExpiredError,
    QuotaExhaustedError,
    fetch_adzuna,
    fetch_apify_linkedin,
    fetch_apify_wttj,
    fetch_apify_indeed,
)


def _fake_response(status_code, json_body):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body
    return r


# ── Adzuna ────────────────────────────────────────────────────────────────


def test_adzuna_credential_error_confirmed_shape():
    with patch.object(job_apis.os, "getenv", side_effect=lambda k, d=None: {
        "ADZUNA_APP_ID": "bad", "ADZUNA_APP_KEY": "bad"
    }.get(k, d)):
        with patch.object(
            job_apis.requests, "get",
            return_value=_fake_response(401, {"exception": "AUTH_FAIL"}),
        ):
            try:
                fetch_adzuna(["AI Engineer"], ["Germany"])
                assert False, "expected CredentialExpiredError to be raised"
            except CredentialExpiredError as e:
                assert e.confirmed is True
                assert e.source == "Adzuna"
                print(f"PASS: Adzuna credential error raised correctly: {e}")


def test_adzuna_generic_4xx_does_not_raise_special_error():
    # A 4xx that ISN'T the confirmed AUTH_FAIL shape must not be
    # misclassified -- falls through to the existing skip-this-country
    # behavior, not a fabricated CredentialExpiredError.
    with patch.object(job_apis.os, "getenv", side_effect=lambda k, d=None: {
        "ADZUNA_APP_ID": "x", "ADZUNA_APP_KEY": "y"
    }.get(k, d)):
        with patch.object(
            job_apis.requests, "get",
            return_value=_fake_response(404, {"some": "other error"}),
        ):
            result = fetch_adzuna(["AI Engineer"], ["Germany"])
            assert result == []
            print("PASS: Adzuna generic 404 does not raise a special error, just skips")


# ── Apify (shared classifier across all three sources) ──────────────────


def test_apify_credential_error_confirmed_shape():
    with patch.object(job_apis.os, "getenv", return_value="bad_token"):
        with patch.object(
            job_apis.requests, "post",
            return_value=_fake_response(
                401, {"error": {"type": "user-or-token-not-found", "message": "..."}}
            ),
        ):
            try:
                fetch_apify_linkedin("AI engineer intern", ["Germany"])
                assert False, "expected CredentialExpiredError"
            except CredentialExpiredError as e:
                assert e.confirmed is True
                assert e.source == "Apify"
                print(f"PASS: Apify credential error raised correctly: {e}")


def test_apify_quota_error_heuristic_unverified():
    # This body is NOT a confirmed real Apify response -- it's a plausible
    # guess. The test asserts the heuristic fires AND is labeled
    # confirmed=False, so nobody mistakes this passing for real-world proof.
    with patch.object(job_apis.os, "getenv", return_value="some_token"):
        with patch.object(
            job_apis.requests, "post",
            return_value=_fake_response(
                403, {"error": {"type": "usage-hard-limit-exceeded",
                                 "message": "Monthly usage limit exceeded"}}
            ),
        ):
            try:
                fetch_apify_wttj("stage intelligence artificielle")
                assert False, "expected QuotaExhaustedError"
            except QuotaExhaustedError as e:
                assert e.confirmed is False, (
                    "quota detection is a heuristic -- must never claim confirmed=True"
                )
                print(f"PASS: Apify quota heuristic fired, correctly labeled unverified: {e}")


def test_apify_429_alone_triggers_quota_heuristic():
    # 429 is Apify's one DOCUMENTED error type (rate-limit-exceeded) --
    # distinct from monthly-cap exhaustion, but still routed to
    # QuotaExhaustedError since both mean "back off / needs attention",
    # not a code bug.
    with patch.object(job_apis.os, "getenv", return_value="some_token"):
        with patch.object(
            job_apis.requests, "post",
            return_value=_fake_response(
                429, {"error": {"type": "rate-limit-exceeded", "message": "too many"}}
            ),
        ):
            try:
                fetch_apify_indeed("AI engineer intern", ["Germany"])
                assert False, "expected QuotaExhaustedError"
            except QuotaExhaustedError as e:
                assert e.confirmed is False
                print(f"PASS: Apify 429 routed to QuotaExhaustedError: {e}")


def test_apify_missing_token_skips_silently():
    # No token configured at all is a "not set up", not a "failed" state --
    # must not raise, must just return empty, same as every free source's
    # missing-credential guard.
    with patch.object(job_apis.os, "getenv", return_value=None):
        result = fetch_apify_linkedin("AI engineer intern", ["Germany"])
        assert result == []
        print("PASS: missing APIFY_API_TOKEN skips silently, no exception")


if __name__ == "__main__":
    test_adzuna_credential_error_confirmed_shape()
    test_adzuna_generic_4xx_does_not_raise_special_error()
    test_apify_credential_error_confirmed_shape()
    test_apify_quota_error_heuristic_unverified()
    test_apify_429_alone_triggers_quota_heuristic()
    test_apify_missing_token_skips_silently()
    print("\nALL PASS")
