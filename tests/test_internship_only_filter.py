"""
Tests for the internship-only backstop filter (main.py's _filter_internship_only).

Bug this guards against (found live, 2026-08-18): a Telegram-triggered run
reported regular full-time jobs ("Distributed Systems Engineer 2026",
"Computer Vision Engineer 2026", etc.) to a candidate who only wants a
final/graduation internship (stage PFE). Root cause was two-fold:
  1. The CV-analysis LLM's suggested_search_queries had no internship marker
     requirement (fixed separately, in the prompt).
  2. Nothing downstream re-checked titles against "does this look like an
     internship at all" for sources other than the free structured APIs
     (tools/job_apis.py already did this for its own sources only).
This tests fix #2: the deterministic backstop applied to the FULL merged
job set regardless of which source found it.

Run directly: python tests/test_internship_only_filter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


def test_non_internship_titles_are_dropped_for_intern_candidate():
    jobs = [
        {"title": "Distributed Systems Engineer", "source_url": "https://x/1"},
        {"title": "Computer Vision Engineer", "source_url": "https://x/2"},
        {"title": "AI Engineer Intern", "source_url": "https://x/3"},
        {"title": "Stage PFE Ingenieur Machine Learning", "source_url": "https://x/4"},
    ]
    result = main._filter_internship_only(jobs, "intern")
    titles = {j["title"] for j in result}
    assert titles == {"AI Engineer Intern", "Stage PFE Ingenieur Machine Learning"}, (
        f"expected only the two internship-marked titles to survive, got: {titles}"
    )
    print(f"PASS: {len(jobs) - len(result)}/{len(jobs)} non-internship titles dropped for an intern candidate")


def test_student_and_junior_seniority_also_filtered():
    jobs = [
        {"title": "Software Engineer", "source_url": "https://x/1"},
        {"title": "Working Student - Data Science", "source_url": "https://x/2"},
    ]
    for seniority in ("student", "junior", "Intern", "STUDENT"):
        result = main._filter_internship_only(jobs, seniority)
        assert len(result) == 1 and result[0]["title"] == "Working Student - Data Science", (
            f"seniority={seniority!r}: expected only the working-student title to survive"
        )
    print("PASS: 'student'/'junior' seniority (and case-insensitivity) also trigger the filter")


def test_mid_senior_lead_candidates_are_not_filtered():
    jobs = [
        {"title": "Distributed Systems Engineer", "source_url": "https://x/1"},
        {"title": "Staff Software Engineer", "source_url": "https://x/2"},
    ]
    for seniority in ("mid", "senior", "lead", ""):
        result = main._filter_internship_only(jobs, seniority)
        assert result == jobs, (
            f"seniority={seniority!r}: a non-intern candidate's job list must pass through untouched"
        )
    print("PASS: mid/senior/lead/unknown seniority leaves the job list untouched (no over-filtering)")


def test_boundary_international_is_not_a_false_positive_intern_match():
    # Same false-positive class already fixed for _senior_title_match: a bare
    # substring match on "intern" would wrongly treat "International" as an
    # internship marker.
    jobs = [{"title": "Fullstack Engineer (International)", "source_url": "https://x/1"}]
    result = main._filter_internship_only(jobs, "intern")
    assert result == [], (
        "'International' must NOT be treated as an internship marker via bare substring match"
    )
    print("PASS: 'Fullstack Engineer (International)' correctly dropped, not a false-positive intern match")


if __name__ == "__main__":
    test_non_internship_titles_are_dropped_for_intern_candidate()
    test_student_and_junior_seniority_also_filtered()
    test_mid_senior_lead_candidates_are_not_filtered()
    test_boundary_international_is_not_a_false_positive_intern_match()
    print("\nALL PASS")
