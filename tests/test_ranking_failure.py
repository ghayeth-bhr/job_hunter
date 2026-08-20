"""
Induced-failure test for the ranking JSON-failure fallback in main.py.

Drives the REAL, unmodified write_opportunities_report() end to end with
main._llm monkeypatched to return deliberately invalid JSON for the ranking
call. Asserts against its actual output -- the real returned string and the
real files it writes to disk -- not a separately hand-built "expected
UNRANKED list". This exact code path has never fired in a real run before,
so this proves the fallback survives, not just that it reads correctly.

Run directly: python tests/test_ranking_failure.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


def test_ranking_failure_produces_nonempty_unranked_report():
    cv_terms = {
        "candidate_name": "Ghayeth Ben Haj Rhouma",
        "seniority_level": "intern",
        "job_titles": [
            "Machine Learning Engineer Intern",
            "Computer Vision Engineer Intern",
        ],
        "skills_technical": ["Python", "PyTorch", "YOLOv8", "OpenCV"],
    }

    # Realistic opportunity data (same shape tools/job_apis.py produces,
    # includes two of the real Arbeitnow postings and one PFE-style listing
    # verified live in a prior session) -- this is just input, not a
    # hand-built "expected result" to compare against.
    opportunities = {
        "total_results": 3,
        "queries_executed": 9,
        "opportunities": [
            {
                "title": "Working Student: AI Engineer, Agentic Systems (m/f/d)",
                "company": "Retorio GmbH",
                "location": "Munich",
                "source_url": "https://www.arbeitnow.com/jobs/companies/retorio-gmbh/working-student-ai-engineer",
                "platform": "arbeitnow",
                "extraction_method": "api",
                "raw_content": "Join our agentic AI team building LLM-powered coaching tools.",
                "date": "2026-08-14T00:00:00+00:00",
                "salary": "",
                "remote_policy": "",
            },
            {
                "title": "AI Engineering Intern",
                "company": "Sanovio",
                "location": "Munich",
                "source_url": "https://www.arbeitnow.com/jobs/companies/sanovio/ai-engineering-intern",
                "platform": "arbeitnow",
                "extraction_method": "api",
                "raw_content": "Support our computer vision pipeline for medical imaging.",
                "date": "2026-08-15T00:00:00+00:00",
                "salary": "",
                "remote_policy": "",
            },
            {
                "title": "Stage Ingenieur IA en Computer Vision",
                "company": "Inria Startup Studio",
                "location": "France",
                "source_url": "https://jobs.inria.fr/public/classic/fr/offres/2026-10162",
                "platform": "inria",
                "extraction_method": "bm25_markdown",
                "raw_content": "Stage de 4 a 6 mois sur la detection d'anomalies par vision 3D.",
                "date": "2026-07-01",
                "salary": "",
                "remote_policy": "",
            },
        ],
    }

    original_llm = main._llm

    def fake_llm_returns_invalid_json(prompt, max_tokens=2000, temperature=0.2):
        # Deliberately truncated JSON -- the real failure mode this guards
        # against is a model cutting off mid-array under max_tokens=6000.
        return '[{"score": 8, "tier": "TOP PICKS", "title": "truncated mid-object'

    main._llm = fake_llm_returns_invalid_json
    try:
        result_str = main.write_opportunities_report(
            json.dumps(cv_terms), json.dumps(opportunities)
        )
    finally:
        main._llm = original_llm

    print("=== RETURNED STRING (from the real write_opportunities_report call) ===")
    print(result_str)

    assert "RANKING_FAILED" in result_str, "Return string must carry the failure marker"

    md_m = re.search(r"Markdown\s*:\s*(.+\.md)", result_str)
    json_m = re.search(r"JSON\s*:\s*(.+\.json)", result_str)
    assert md_m and json_m, "Report file paths must be present in the return string"
    md_path = Path(md_m.group(1).strip())
    json_path = Path(json_m.group(1).strip())

    # Files must actually exist on disk -- not just an in-memory claim.
    assert md_path.exists(), f"Markdown report was not written to {md_path}"
    assert json_path.exists(), f"JSON report was not written to {json_path}"

    md_content = md_path.read_text(encoding="utf-8")
    report = json.loads(json_path.read_text(encoding="utf-8"))

    print("\n=== WRITTEN MARKDOWN (real file read back from disk: %s) ===" % md_path)
    print(md_content)

    print("\n=== WRITTEN JSON meta (real file read back from disk: %s) ===" % json_path)
    print(json.dumps(report["meta"], indent=2))

    # The actual bug being guarded against: the report must NOT be silently
    # empty. All 3 input jobs must survive as UNRANKED.
    assert report["meta"]["ranking_failed"] is True
    assert len(report["ranked_opportunities"]) == 3, (
        f"Expected all 3 input jobs to survive as UNRANKED, got "
        f"{len(report['ranked_opportunities'])} -- this is exactly the "
        f"silent-empty-report bug if it regresses."
    )
    assert all(r["tier"] == "UNRANKED" for r in report["ranked_opportunities"])
    assert all(r["score"] is None for r in report["ranked_opportunities"])
    assert len(report["skipped_opportunities"]) == 0, (
        "None of these should have been routed to SKIP on a ranking failure"
    )

    assert "RANKING INCOMPLETE THIS RUN" in md_content
    assert "UNRANKED" in md_content
    for title in [
        "Working Student: AI Engineer",
        "AI Engineering Intern",
        "Stage Ingenieur IA",
    ]:
        assert title in md_content, (
            f"Expected job title '{title}' to appear in the rendered report"
        )

    print(
        "\nPASS: induced ranking failure produced a non-empty, "
        "clearly-labeled UNRANKED report on disk."
    )


if __name__ == "__main__":
    test_ranking_failure_produces_nonempty_unranked_report()
