"""
programs.py — hand-curated funded internship/research programs.

Ported approach from pfzebi/PFE-Hunter's config/programs.yml: deliberately
NOT crawled or search-discovered. This project verified live (2026-08-18)
that no query surfaces Mitacs Globalink's own page reliably -- even a query
naming it directly returns third-party mentions in Google's top 10, never
mitacs.ca itself. A scraper that silently mis-reads a funding deadline is
worse than no scraper, because you'd trust it. config/programs.yaml is read
directly and turned into opportunities by date arithmetic here -- no search
engine, no scraping, no LLM, and therefore no failure mode at all beyond a
malformed YAML file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROGRAMS_FILE = Path(__file__).resolve().parent.parent / "config" / "programs.yaml"


def load_funded_programs(path: Path = PROGRAMS_FILE) -> list[dict]:
    """Returns the raw program entries from config/programs.yaml, or an
    empty list if the file is missing/empty -- funded programs are a bonus
    source, never a hard dependency for the pipeline to run."""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        print(f"  [WARN] {path} is not valid YAML ({e}) -- skipping funded programs this run.")
        return []
    return data.get("programs", []) or []


def program_to_opportunity(program: dict) -> dict:
    """Converts one config/programs.yaml entry into the same opportunity
    dict shape every other source produces, so it flows through the exact
    same dedup/eligibility/scoring/report pipeline as a scraped posting --
    no parallel code path to maintain."""
    domains = program.get("domains", []) or []
    eligibility = program.get("eligibility", "") or ""
    return {
        "title": program.get("name", "Funded Program"),
        "company": program.get("org", ""),
        "location": program.get("country", ""),
        "source_url": program.get("apply_url", ""),
        "platform": "funded_program",
        "raw_content": (
            f"{eligibility}. Relevant domains: {', '.join(domains)}."
            f" Last verified: {program.get('last_verified', 'unknown')}."
        ),
        "date": program.get("deadline", ""),
        "deadline": program.get("deadline", ""),
        "remote_policy": "",
        "salary": "",
        "extraction_method": "curated_config",
        "track": "program",
    }
