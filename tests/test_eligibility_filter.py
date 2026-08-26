"""
Tests for the eligibility check (main.py): a deterministic pre-filter for
confirmed disqualifiers (ITAR/US-person/no-sponsorship language), plus the
LLM-assisted INELIGIBLE tier that flows through write_opportunities_report.

Real feedback (2026-08-22): a merged report included postings the candidate
(Tunisian, no other citizenship/permit) could never actually apply to --
a US defense-contractor internship requiring ITAR/US-person status, and
roles explicitly stating no visa sponsorship. Nothing checked this before.

Run directly: python tests/test_eligibility_filter.py
"""

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


# ── 1. Deterministic pre-filter -- confirmed cases only ──────────────────


def test_confirmed_disqualifiers_are_caught():
    cases = [
        ("Software Engineer Intern", "Must be a U.S. citizen due to ITAR restrictions on this program."),
        ("Defense Systems Intern", "This role is open to U.S. persons only."),
        ("AI Intern", "We are unable to sponsor visas for this internship."),
        ("Backend Intern", "Candidates must already have the right to work in Germany."),
        ("Research Intern", "Note: this position is export-controlled."),
    ]
    for title, raw_content in cases:
        reason = main._eligibility_disqualified_match(title, raw_content)
        assert reason is not None, f"Missed a confirmed disqualifier: {title!r} / {raw_content!r}"
        print(f"PASS: {title!r} correctly caught (matched {reason!r})")


def test_ordinary_silent_postings_are_not_caught_by_the_deterministic_filter():
    # Plain silence on sponsorship (the common case) is NOT a job for the
    # deterministic filter -- that needs actual reasoning about country,
    # French stage/PFE exceptions, etc., which only the ranking LLM can do.
    # A false positive here would silently exclude postings before the LLM
    # ever gets a chance to apply the nuanced eligibility check.
    cases = [
        ("Software Engineer Intern", "Join our team in Warsaw working on backend systems."),
        ("Stage PFE Ingenieur IA", "Rejoignez notre equipe basee a Paris pour un stage de fin d'etudes."),
        ("Working Student", "Munich-based, flexible hours, great team culture."),
    ]
    for title, raw_content in cases:
        reason = main._eligibility_disqualified_match(title, raw_content)
        assert reason is None, (
            f"False positive: {title!r} / {raw_content!r} was excluded by the "
            f"deterministic filter (matched {reason!r}) despite no explicit "
            f"disqualifying language -- this should be left to the LLM."
        )
    print("PASS: ordinary silent-on-sponsorship postings are NOT excluded by the deterministic filter")


# ── 2. Deterministic auto-resolve exceptions skip the LLM entirely ───────


def test_auto_resolved_program_track_skips_llm():
    resolved = main._eligibility_auto_resolved({"track": "program", "title": "Mitacs Globalink Research Internship"})
    assert resolved is not None and resolved[0] is True
    print("PASS: a funded-program-track item auto-resolves eligible, no LLM needed")


def test_auto_resolved_french_stage_pfe_skips_llm():
    resolved = main._eligibility_auto_resolved({"title": "Stage PFE Ingenieur Vision"})
    assert resolved is not None and resolved[0] is True
    print("PASS: a French stage/PFE title auto-resolves eligible, no LLM needed")


def test_auto_resolved_positive_sponsorship_language_skips_llm():
    resolved = main._eligibility_auto_resolved({"title": "AI Intern", "raw_content": "We sponsor work visas for international candidates."})
    assert resolved is not None and resolved[0] is True
    print("PASS: explicit positive sponsorship language auto-resolves eligible, no LLM needed")


def test_auto_resolved_fully_remote_skips_llm():
    resolved = main._eligibility_auto_resolved({"title": "Backend Intern", "remote_policy": "fully remote"})
    assert resolved is not None and resolved[0] is True
    print("PASS: a fully-remote posting auto-resolves eligible, no LLM needed")


def test_ambiguous_onsite_posting_is_not_auto_resolved():
    # No program track, no French marker, no positive/negative intl
    # language, not remote -- this genuinely needs the LLM's judgment.
    resolved = main._eligibility_auto_resolved({
        "title": "AI Engineer Intern", "raw_content": "Onsite in Berlin, Germany. Great team.",
    })
    assert resolved is None
    print("PASS: an ordinary ambiguous onsite posting is correctly left for LLM judgment")


def test_eligibility_prompt_includes_candidate_profile():
    terms = {
        "candidate_name": "Test Candidate", "seniority_level": "intern",
        "citizenship": "Tunisian", "work_authorization": "No EU/US/Canada status.",
        "availability_start": "2027-01-01", "availability_duration_months": 6,
    }
    batch = [{"id": 1, "title": "AI Intern", "source_url": "https://x/1", "raw_content": ""}]
    prompt = main._build_eligibility_prompt(batch, terms)
    assert "Tunisian" in prompt and "2027-01-01" in prompt
    print("PASS: eligibility prompt includes the candidate's citizenship/work-authorization/availability")


# ── 3. LLM-judged ineligibility flows through write_opportunities_report ─
#
#  Only the genuinely ambiguous posting (onsite, no program/remote/French/
#  positive-sponsorship marker) reaches the mocked LLM here -- the other
#  three postings below are resolved deterministically and never call it,
#  which this test also verifies via a call counter.


def test_eligibility_and_deterministic_scoring_flow_through_report_correctly():
    cv_terms = {
        "candidate_name": "Test Candidate",
        "seniority_level": "intern",
        "citizenship": "Tunisian",
        "work_authorization": "No EU/US/Canada status; only sponsored/exception/remote/program postings work.",
        "job_titles": ["AI Engineer Intern"],
        "skills_technical": ["Python", "PyTorch"],
    }
    opportunities = {
        "total_results": 3,
        "queries_executed": 1,
        "opportunities": [
            # Ambiguous -- must go to the LLM, which will say ineligible.
            {
                "title": "AI Engineer Intern", "company": "Acme", "source_url": "https://example.com/1",
                "platform": "linkedin", "raw_content": "Onsite in Boston, MA, USA. Great Python team.",
            },
            # Auto-resolved eligible (fully remote) -- never reaches the LLM.
            {
                "title": "AI Engineer Intern", "company": "Beta", "source_url": "https://example.com/2",
                "platform": "linkedin", "remote_policy": "fully remote",
                "raw_content": "Python, PyTorch, great team, fully remote.",
            },
            # Auto-resolved eligible (French stage/PFE) -- never reaches the LLM.
            {
                "title": "Stage PFE Ingenieur IA", "company": "Gamma", "source_url": "https://example.com/3",
                "platform": "welcometothejungle", "raw_content": "Python, PyTorch. Stage de fin d'etudes, Paris.",
            },
        ],
    }

    call_count = {"n": 0}

    def _fake_llm(prompt, *args, **kwargs):
        call_count["n"] += 1
        return json.dumps([{"id": 1, "eligible": False, "reason": "Requires US work authorization; no sponsorship stated"}])

    original_llm = main._llm
    main._llm = _fake_llm
    try:
        result_str = main.write_opportunities_report(json.dumps(cv_terms), json.dumps(opportunities))
    finally:
        main._llm = original_llm

    print("=== RETURNED STRING ===")
    print(result_str)
    assert call_count["n"] == 1, f"expected exactly 1 LLM call (only the ambiguous posting), got {call_count['n']}"
    assert "1 INELIGIBLE excluded" in result_str

    md_m = re.search(r"Markdown\s*:\s*(.+\.md)", result_str)
    json_m = re.search(r"JSON\s*:\s*(.+\.json)", result_str)
    assert md_m and json_m
    md_content = Path(md_m.group(1).strip()).read_text(encoding="utf-8")
    report = json.loads(Path(json_m.group(1).strip()).read_text(encoding="utf-8"))

    assert report["meta"]["jobs_ineligible"] == 1
    assert report["meta"]["scoring"] == "deterministic (tools/scoring.py)"
    assert len(report["ranked_opportunities"]) == 2
    for r in report["ranked_opportunities"]:
        assert isinstance(r["score"], (int, float)), "deterministic scoring must always produce a real number"
    assert len(report["ineligible_opportunities"]) == 1
    assert "US work authorization" in report["ineligible_opportunities"][0]["reason"][0]
    print("PASS: only the ambiguous posting called the LLM; the other two resolved deterministically; "
          "all scoring is deterministic; ineligible entry excluded and reason preserved")

    assert "NOT ELIGIBLE" in md_content and "Acme" in md_content
    print("PASS: rendered Markdown shows the ineligible entry in its own section")


if __name__ == "__main__":
    test_confirmed_disqualifiers_are_caught()
    test_ordinary_silent_postings_are_not_caught_by_the_deterministic_filter()
    test_auto_resolved_program_track_skips_llm()
    test_auto_resolved_french_stage_pfe_skips_llm()
    test_auto_resolved_positive_sponsorship_language_skips_llm()
    test_auto_resolved_fully_remote_skips_llm()
    test_ambiguous_onsite_posting_is_not_auto_resolved()
    test_eligibility_prompt_includes_candidate_profile()
    test_eligibility_and_deterministic_scoring_flow_through_report_correctly()
    print("\nALL PASS")
