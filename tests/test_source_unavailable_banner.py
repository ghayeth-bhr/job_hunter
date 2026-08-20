"""
End-to-end test: a source-unavailable condition must survive all the way
through to write_opportunities_report's rendered Markdown/JSON -- not just
that the exception gets raised (already covered by
test_source_unavailable_errors.py), but that it's actually VISIBLE in the
real generated report, the same discipline used for FIX 1's UNRANKED
banner and the DISPUTED banner.

Deliberately does NOT drive this through fetch_all_free_apis with a
globally-patched requests.get -- that patches every free source's HTTP
calls at once (Arbeitnow, RemoteOK, etc.), not just Adzuna's, and several
of those don't defend against a malformed/unexpected response shape (e.g.
RemoteOK does `data[1:]` assuming a list). That cross-contamination is
exactly the kind of test-only bug this project has caught before by
insisting on real, isolated conditions -- so this test isolates the two
real concerns instead: (1) fetch_adzuna's classifier already proven in
test_source_unavailable_errors.py, and (2) whether
write_opportunities_report correctly renders a sources_unavailable list
into a real Markdown/JSON report, using the exact shape that classifier
actually produces.

Run directly: python tests/test_source_unavailable_banner.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


def _fake_llm_valid_ranking(prompt, max_tokens=2000, temperature=0.2):
    """Ranking isn't what this test is about -- return a real, valid,
    minimal ranking so the test stays fast and free instead of making a
    real paid/slow LLM call just to reach the report-writing code."""
    return json.dumps([
        {
            "id": 1,
            "score": 7,
            "tier": "GOOD FITS",
            "title": "Real Job From Serper",
            "company": "Acme",
            "source_url": "https://example.com/job/1",
            "match_reasons": ["Matches profile"],
            "concerns": [],
        }
    ])


def test_sources_unavailable_reaches_the_rendered_report():
    cv_terms = {
        "candidate_name": "Test Candidate",
        "seniority_level": "intern",
        "job_titles": ["AI Engineer Intern"],
        "skills_technical": ["Python"],
    }
    # Exact shape _call_source_safely actually produces on a confirmed
    # CredentialExpiredError (see tools/job_apis.py) -- not a hand-invented
    # dict, the real one.
    sources_unavailable = [
        {
            "source": "Adzuna",
            "kind": "CREDENTIAL EXPIRED",
            "reason": "app_id/app_key rejected (confirmed AUTH_FAIL)",
            "confirmed": True,
        }
    ]
    opportunities = {
        "total_results": 1,
        "queries_executed": 1,
        "opportunities": [
            {
                "title": "Real Job From Serper",
                "company": "Acme",
                "source_url": "https://example.com/job/1",
                "platform": "unknown",
                "snippet": "A real job.",
            }
        ],
        "sources_unavailable": sources_unavailable,
    }

    original_llm = main._llm
    main._llm = _fake_llm_valid_ranking
    try:
        result_str = main.write_opportunities_report(
            json.dumps(cv_terms), json.dumps(opportunities)
        )
    finally:
        main._llm = original_llm

    assert "SOURCE_UNAVAILABLE" in result_str, "return string must carry the marker"
    print("PASS: return string carries SOURCE_UNAVAILABLE marker")

    md_m = re.search(r"Markdown\s*:\s*(.+\.md)", result_str)
    json_m = re.search(r"JSON\s*:\s*(.+\.json)", result_str)
    assert md_m and json_m
    md_path = Path(md_m.group(1).strip())
    json_path = Path(json_m.group(1).strip())

    md_content = md_path.read_text(encoding="utf-8")
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert "SOURCE(S) UNAVAILABLE THIS RUN" in md_content
    assert "Adzuna" in md_content
    assert "CREDENTIAL EXPIRED" in md_content
    assert "confirmed" in md_content
    print("PASS: real rendered Markdown contains the SOURCE_UNAVAILABLE banner naming Adzuna")

    assert report["meta"]["sources_unavailable"] == sources_unavailable
    print("PASS: real written JSON meta carries sources_unavailable exactly")

    print("\n=== Rendered banner excerpt ===")
    start = md_content.find("SOURCE(S) UNAVAILABLE")
    print(md_content[max(0, start - 3): start + 350])


if __name__ == "__main__":
    test_sources_unavailable_reaches_the_rendered_report()
    print("\nALL PASS")
