"""
Tests for tools/scoring.py -- the deterministic ranking engine ported from
pfzebi/PFE-Hunter. Every case here mirrors a bug that project's own commit
history documents finding on real data, plus two bugs this port introduced
and fixed during translation (OpenCV mis-casing, YOLO version-suffix
mismatch) -- caught by writing exactly this kind of test before wiring the
module into main.py.

Fully offline, no network, no LLM -- the whole point of this module.

Run directly: python tests/test_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import scoring


# ── classify_tier: leftmost marker, not highest weight ────────────────────


def test_leftmost_marker_decides_not_highest_weight():
    # The documented bug: a naive highest-weight match misreads this as
    # senior because "Director" outranks "Intern" by weight. Position must
    # decide instead.
    assert scoring.classify_tier("Summer Intern, Director of Product") == "intern"
    print("PASS: 'Summer Intern, Director of Product' classified as intern, not senior")


def test_intern_program_director_bridge_guard():
    # The inverse case the leftmost rule alone would get wrong: this manages
    # an internship program, it is not one.
    assert scoring.classify_tier("Intern Program Director") == "senior"
    print("PASS: 'Intern Program Director' correctly classified as senior (bridge guard)")


def test_associate_senior_guard():
    assert scoring.classify_tier("Associate Director of Engineering") == "senior"
    print("PASS: 'Associate Director of Engineering' correctly classified as senior")


def test_multilingual_intern_markers():
    cases = [
        ("Praktikum Machine Learning (m/w/d)", "intern"),
        ("Stagiaire Data Scientist", "intern"),
        ("Werkstudent Backend Developer", "intern"),
        ("Becario de Ingenieria", "intern"),
        ("Staff Software Engineer", "senior"),
        ("Software Engineer Intern", "intern"),
    ]
    for title, expected in cases:
        got = scoring.classify_tier(title)
        assert got == expected, f"{title!r}: expected {expected!r}, got {got!r}"
    print("PASS: multilingual intern markers (Praktikum/Stagiaire/Werkstudent/Becario) all classified correctly")


# ── skill_fit: posting-as-denominator, Bayesian smoothing, org exclusion ──


def test_skill_fit_denominator_is_the_posting_not_the_cv():
    # A posting naming one skill the candidate has is NOT a 100% match --
    # Bayesian smoothing with prior weight 2 gives 1/1 -> 0.667, not 1.0.
    r = scoring.skill_fit("We need Computer Vision experience.", {"Computer Vision", "Python", "PyTorch"})
    assert abs(r["score"] - 2 / 3) < 0.01, r
    print(f"PASS: 1/1 skill match smoothed to {r['score']:.3f}, not 1.0")


def test_skill_fit_unknown_when_posting_names_nothing_recognizable():
    r = scoring.skill_fit("Great team, great culture, apply now!", {"Python"})
    assert r["unknown"] is True and r["score"] == 0.5
    print("PASS: a posting naming no recognizable skill returns neutral 0.5, unknown=True")


def test_skill_fit_excludes_employer_name():
    # A Mistral posting says "Mistral" because that's who they are, not
    # because they're asking for it as a skill.
    r = scoring.skill_fit("Join Mistral to build the next generation of LLMs.", {"LLMs"}, org="Mistral")
    assert "Mistral" not in r["matched"] and "Mistral" not in r["missing"]
    print("PASS: employer's own name excluded from required-skills extraction")


def test_skill_fit_version_suffixes_match_bare_name():
    # Caught during this port: "YOLOv8"/"YOLO11" must canonicalize the same
    # as bare "YOLO", or a candidate with YOLOv8 experience never matches a
    # posting asking for plain "YOLO" (or vice versa).
    r = scoring.skill_fit("Need YOLOv8 and YOLO11 experience.", {"YOLO"})
    assert r["matched"] == ["YOLO"] and not r["missing"], r
    print("PASS: YOLOv8/YOLO11 both canonicalize to YOLO, matching a bare-YOLO profile skill")


def test_skill_fit_opencv_casing():
    # Caught during this port: OpenCV was mis-casing to "Opencv" via a naive
    # str.title() fallback, silently breaking matches between an extracted
    # mention and a declared profile skill.
    profile = scoring.profile_skill_set(["OpenCV"])
    assert "OpenCV" in profile, profile
    r = scoring.skill_fit("Experience with OpenCV required.", profile)
    assert r["matched"] == ["OpenCV"], r
    print("PASS: OpenCV canonicalizes with correct casing on both the profile and extraction side")


# ── role_fit: stopwords, seniority gating, discriminating overlap ────────


def test_role_fit_rejects_baseline_only_overlap():
    # "Software Engineer" vs "Data Engineer" share only "engineer" --
    # a baseline word every technical title has. Must not count as a match.
    r = scoring.role_fit("Data Engineer", ["Software Engineer"])
    assert r["score"] == 0.0, r
    print("PASS: overlap on only a baseline word ('engineer') scores 0, not a partial match")


def test_role_fit_seniority_gate_refuses_mismatched_levels():
    r = scoring.role_fit("Senior Machine Learning Engineer", ["Machine Learning Engineer Intern"])
    assert r["score"] == 0.0, r
    print("PASS: 'Senior X' vs 'X Intern' refused outright by the seniority gate, not just scored down")


def test_role_fit_matches_close_title():
    r = scoring.role_fit(
        "Computer Vision Engineering Intern (Fall 2026)",
        ["AI Engineer Intern", "Computer Vision Engineer Intern"],
    )
    assert r["best"] == "Computer Vision Engineer Intern" and r["score"] > 0.3, r
    print(f"PASS: close title match found (score={r['score']:.2f}, best={r['best']!r})")


# ── legitimacy flags, recency, confidence shrinkage ───────────────────────


def test_legitimacy_flags_catch_scam_shape():
    flags = scoring.legitimacy_flags("Great internship! Contact us on WhatsApp to apply and interview.")
    assert flags, "should have flagged WhatsApp hiring"
    print(f"PASS: scam-shaped posting flagged: {flags}")


def test_legitimacy_flags_silent_on_normal_posting():
    assert scoring.legitimacy_flags("Normal posting about Python and Docker, apply via our careers page.") == []
    print("PASS: ordinary posting raises no legitimacy flags")


def test_recency_score_decays_and_floors_at_zero():
    assert scoring.recency_score(None) is None
    from datetime import datetime, timezone
    now = datetime(2026, 8, 26, tzinfo=timezone.utc)
    assert scoring.recency_score("2026-08-26T00:00:00Z", now=now) == 1.0
    assert scoring.recency_score("2026-01-01T00:00:00Z", max_age_days=90, now=now) == 0.0
    print("PASS: recency_score is None with no date, 1.0 when fresh, floors at 0.0 when stale")


def test_combine_redistributes_null_weight_not_defaults_to_half():
    # Two known terms should behave as if the unknown one never existed --
    # NOT as if it contributed a 0.5 at full weight, which drags every
    # partially-known row toward the center.
    only_known = scoring.combine([(1.0, 0.5), (1.0, 0.5)])
    with_unknown = scoring.combine([(1.0, 0.5), (1.0, 0.5), (None, 1.0)])
    assert only_known == with_unknown == 1.0
    print("PASS: combine() redistributes a None term's weight instead of substituting 0.5")


def test_shrink_to_neutral_pulls_toward_center_both_directions():
    assert scoring.shrink_to_neutral(1.0, confidence=0.5) == 0.75
    assert scoring.shrink_to_neutral(0.0, confidence=0.5) == 0.25
    print("PASS: shrink_to_neutral pulls both a high and a low raw score toward 0.5")


# ── score_opportunity: end to end, both tracks ────────────────────────────


def test_score_opportunity_industry_track():
    profile_skills = scoring.profile_skill_set(["Python", "PyTorch", "OpenCV", "YOLOv8"])
    item = {
        "title": "Computer Vision Engineering Intern (Fall 2026)",
        "company": "Acme Robotics",
        "raw_content": "We need an intern with Python, PyTorch, OpenCV and YOLO experience, "
                        "working on real-time detection pipelines and dataset curation for "
                        "our robotics platform. Six-month internship, on-site in Berlin.",
        "date": "2026-08-15T00:00:00Z",
    }
    r = scoring.score_opportunity(item, profile_skills, ["Computer Vision Engineer Intern"])
    assert r["tier"] in ("TOP PICKS", "GOOD FITS"), r
    assert "OpenCV" in r["matched_skills"] and "YOLO" in r["matched_skills"]
    assert r["flags"] == []
    print(f"PASS: industry-track scoring produced tier={r['tier']} score={r['score']}")


def test_score_opportunity_penalizes_scam_multiplicatively():
    profile_skills = scoring.profile_skill_set(["Python", "PyTorch"])
    item = {
        "title": "AI Engineer Intern",
        "company": "TotallyLegit LLC",
        "raw_content": "Python and PyTorch role. Unpaid position, pay a small registration fee "
                        "to secure your spot, contact us on WhatsApp to interview.",
    }
    r = scoring.score_opportunity(item, profile_skills, ["AI Engineer Intern"])
    assert r["tier"] == "SKIP", r
    assert r["flags"], "expected legitimacy flags to have fired"
    print(f"PASS: scam-shaped posting multiplicatively penalized to tier={r['tier']} despite skill match")


def test_score_opportunity_program_track():
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    soon = (now + timedelta(days=10)).date().isoformat()
    item = {
        "title": "Mitacs Globalink Research Internship",
        "company": "Mitacs",
        "raw_content": "Undergraduate students in AI, computer vision, machine learning welcome to apply.",
        "date": soon,
        "track": "program",
    }
    profile_skills = scoring.profile_skill_set(["Python", "computer vision", "machine learning"])
    r = scoring.score_opportunity(item, profile_skills, ["AI Engineer Intern"])
    assert r["tier"] in ("TOP PICKS", "GOOD FITS"), r
    print(f"PASS: program-track (close deadline) scored tier={r['tier']} score={r['score']}")


def test_score_opportunity_program_track_expired_deadline_scores_low():
    item = {
        "title": "Some Expired Program",
        "company": "Org",
        "raw_content": "Undergraduate students welcome.",
        "date": "2020-01-01",
        "track": "program",
    }
    profile_skills = scoring.profile_skill_set(["Python"])
    r = scoring.score_opportunity(item, profile_skills, ["AI Engineer Intern"])
    assert r["tier"] == "SKIP", r
    print(f"PASS: program with a passed deadline scores {r['tier']} (deadline term floors at 0)")


if __name__ == "__main__":
    test_leftmost_marker_decides_not_highest_weight()
    test_intern_program_director_bridge_guard()
    test_associate_senior_guard()
    test_multilingual_intern_markers()
    test_skill_fit_denominator_is_the_posting_not_the_cv()
    test_skill_fit_unknown_when_posting_names_nothing_recognizable()
    test_skill_fit_excludes_employer_name()
    test_skill_fit_version_suffixes_match_bare_name()
    test_skill_fit_opencv_casing()
    test_role_fit_rejects_baseline_only_overlap()
    test_role_fit_seniority_gate_refuses_mismatched_levels()
    test_role_fit_matches_close_title()
    test_legitimacy_flags_catch_scam_shape()
    test_legitimacy_flags_silent_on_normal_posting()
    test_recency_score_decays_and_floors_at_zero()
    test_combine_redistributes_null_weight_not_defaults_to_half()
    test_shrink_to_neutral_pulls_toward_center_both_directions()
    test_score_opportunity_industry_track()
    test_score_opportunity_penalizes_scam_multiplicatively()
    test_score_opportunity_program_track()
    test_score_opportunity_program_track_expired_deadline_scores_low()
    print("\nALL PASS")
