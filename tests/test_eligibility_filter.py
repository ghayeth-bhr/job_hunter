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


# ── 2. Ranking prompt includes/omits the eligibility section correctly ───


def test_ranking_prompt_includes_eligibility_section_when_known():
    terms = {
        "candidate_name": "Test Candidate",
        "seniority_level": "intern",
        "citizenship": "Tunisian",
        "work_authorization": "No EU/US/Canada status.",
        "availability_start": "2027-01-01",
        "availability_duration_months": 6,
    }
    batch = [{"id": 1, "title": "AI Intern", "source_url": "https://x/1"}]
    prompt = main._build_ranking_prompt(batch, terms)
    # The section HEADER (with colon) is only injected when known -- the
    # bare phrase "CANDIDATE ELIGIBILITY" also appears in SCORING_SYSTEM_PROMPT's
    # own instructions regardless, so that alone can't distinguish the two.
    assert "CANDIDATE ELIGIBILITY:" in prompt
    assert "Tunisian" in prompt
    assert "2027-01-01" in prompt
    print("PASS: eligibility section present in the ranking prompt when citizenship/work_authorization are known")


def test_ranking_prompt_omits_eligibility_section_when_unknown():
    terms = {"candidate_name": "Test Candidate", "seniority_level": "intern"}
    batch = [{"id": 1, "title": "AI Intern", "source_url": "https://x/1"}]
    prompt = main._build_ranking_prompt(batch, terms)
    assert "CANDIDATE ELIGIBILITY:" not in prompt
    print("PASS: eligibility section correctly omitted when citizenship/work_authorization are unknown")


# ── 3. LLM-tagged INELIGIBLE flows through write_opportunities_report ────


def test_llm_ineligible_tier_flows_through_report_correctly():
    cv_terms = {
        "candidate_name": "Test Candidate",
        "seniority_level": "intern",
        "citizenship": "Tunisian",
        "work_authorization": "No EU/US/Canada status; only sponsored/exception postings work.",
        "job_titles": ["AI Engineer Intern"],
        "skills_technical": ["Python"],
    }
    opportunities = {
        "total_results": 2,
        "queries_executed": 1,
        "opportunities": [
            {
                "title": "AI Engineer Intern",
                "company": "Acme",
                "source_url": "https://example.com/1",
                "platform": "linkedin",
                "raw_content": "Onsite in Boston, MA. Requires US work authorization; no sponsorship provided.",
            },
            {
                "title": "AI Engineer Intern",
                "company": "Beta",
                "source_url": "https://example.com/2",
                "platform": "linkedin",
                "raw_content": "Fully remote, hire-from-anywhere, great Python team.",
            },
        ],
    }

    def _fake_llm(prompt, *args, **kwargs):
        return json.dumps(
            [
                {
                    "id": 1,
                    "score": None,
                    "tier": "INELIGIBLE",
                    "title": "AI Engineer Intern",
                    "company": "Acme",
                    "concerns": ["Requires US work authorization; no sponsorship stated"],
                    "source_url": "https://example.com/1",
                },
                {
                    "id": 2,
                    "score": 8,
                    "tier": "TOP PICKS",
                    "title": "AI Engineer Intern",
                    "company": "Beta",
                    "match_reasons": ["Fully remote, no location restriction"],
                    "source_url": "https://example.com/2",
                },
            ]
        )

    original_llm = main._llm
    main._llm = _fake_llm
    try:
        result_str = main.write_opportunities_report(json.dumps(cv_terms), json.dumps(opportunities))
    finally:
        main._llm = original_llm

    print("=== RETURNED STRING ===")
    print(result_str)
    assert "1 INELIGIBLE excluded" in result_str

    md_m = re.search(r"Markdown\s*:\s*(.+\.md)", result_str)
    json_m = re.search(r"JSON\s*:\s*(.+\.json)", result_str)
    assert md_m and json_m
    md_content = Path(md_m.group(1).strip()).read_text(encoding="utf-8")
    report = json.loads(Path(json_m.group(1).strip()).read_text(encoding="utf-8"))

    assert report["meta"]["jobs_ineligible"] == 1
    assert len(report["ranked_opportunities"]) == 1
    assert report["ranked_opportunities"][0]["tier"] == "TOP PICKS"
    assert len(report["ineligible_opportunities"]) == 1
    assert "US work authorization" in report["ineligible_opportunities"][0]["reason"][0]
    print("PASS: INELIGIBLE entry excluded from ranked_opportunities, correctly counted, reason preserved")

    assert "NOT ELIGIBLE" in md_content and "Acme" in md_content
    assert "TOP PICKS" in md_content and "Beta" in md_content
    print("PASS: rendered Markdown shows the ineligible entry in its own section, eligible one in TOP PICKS")


if __name__ == "__main__":
    test_confirmed_disqualifiers_are_caught()
    test_ordinary_silent_postings_are_not_caught_by_the_deterministic_filter()
    test_ranking_prompt_includes_eligibility_section_when_known()
    test_ranking_prompt_omits_eligibility_section_when_unknown()
    test_llm_ineligible_tier_flows_through_report_correctly()
    print("\nALL PASS")
