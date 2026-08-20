"""
Boundary test for the deterministic seniority pre-filter (main.py).

The exact bug class this guards against: naive keyword matching produced a
false positive in the very first session ("intern" as a bare substring also
matched "International"). This checks the seniority pre-filter's regex
doesn't repeat that mistake -- a title that legitimately contains a senior
keyword but IS an internship (not a seniority mismatch) must NOT be excluded.

Run directly: python tests/test_seniority_prefilter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main


def test_boundary_internship_mentioning_director_is_not_excluded():
    # Legitimately contains "Director" but is an internship, not a
    # director-level role -- this is exactly the false-positive shape to
    # guard against.
    title = "Internship — Assistant to the Director of Engineering"
    result = main._senior_title_match(title)
    assert result is None, (
        f"False positive: {title!r} was excluded (matched {result!r}) despite "
        f"being a legitimate internship, not a real seniority mismatch."
    )
    print(f"PASS: {title!r} correctly NOT excluded (result={result!r})")


def test_genuine_senior_titles_are_excluded():
    cases = [
        ("Staff Engineer (Control Systems) – Fastly – Permanent contract", "staff"),
        ("Enterprise Architect (F/H) - Deloitte - CDI à Strasbourg", "enterprise architect"),
        ("VP of Engineering", "vp"),
        ("Director of Product", "director"),
    ]
    for title, expected_substring in cases:
        result = main._senior_title_match(title)
        assert result is not None, f"Missed a genuine seniority mismatch: {title!r}"
        print(f"PASS: {title!r} correctly excluded (matched {result!r})")


def test_unrelated_titles_are_not_excluded():
    cases = [
        "Software Engineer Intern",
        "Machine Learning Engineer Intern (TikTok-Data-Search)",
        "AI Engineering Intern",
    ]
    for title in cases:
        result = main._senior_title_match(title)
        assert result is None, f"False positive on unrelated title: {title!r} (matched {result!r})"
        print(f"PASS: {title!r} correctly NOT excluded")


if __name__ == "__main__":
    test_boundary_internship_mentioning_director_is_not_excluded()
    test_genuine_senior_titles_are_excluded()
    test_unrelated_titles_are_not_excluded()
    print("\nALL PASS")
