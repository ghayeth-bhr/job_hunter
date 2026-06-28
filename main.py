"""
═══════════════════════════════════════════════════════════════════════════════
  EU JOB HUNTER — Multi-Agent Pipeline (OpenAI Agents SDK × OpenRouter)
═══════════════════════════════════════════════════════════════════════════════
  Architecture:
    1. CV Researcher Agent  → extracts skills, roles, keywords from CV (PDF/DOCX)
    2. Web Search Agent     → exhaustive Serper searches across EU job platforms
    3. Orchestrator Agent   → coordinates both, ranks results, writes final report

  LLM backend: OpenRouter (drop-in OpenAI-compatible API)
    • Agents SDK is wired to OpenRouter via AsyncOpenAI(base_url=...) +
      OpenAIChatCompletionsModel — zero SDK patching needed.
    • Inner tool LLM calls use the synchronous openai.OpenAI(base_url=...) client.
    • Swap MODEL to any OpenRouter model slug without touching logic.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import concurrent.futures
import json
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from dotenv import load_dotenv

load_dotenv()

from tools.crawl4ai_scraper import scrape_job_listings
from tools.company_enricher import enrich_companies, merge_enrichment_into_jobs
from tools.direct_board_crawler import discover_jobs_direct
from prompts.scoring_agent import SCORING_SYSTEM_PROMPT

import requests

# ── PDF / DOCX readers ───────────────────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

# ── OpenAI SDK (used for both sync inner calls and async Agents SDK) ─────────
import openai

# ── OpenAI Agents SDK ────────────────────────────────────────────────────────
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (set in .env or export before running)
# ════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "YOUR_SERPER_API_KEY_HERE")

# Any OpenRouter model slug — see https://openrouter.ai/models
# Great free/cheap options:
#   "mistralai/mistral-7b-instruct"          (free tier)
#   "meta-llama/llama-3.1-8b-instruct"       (free tier)
#   "google/gemma-3-27b-it"                  (free tier)
#   "anthropic/claude-3.5-haiku"             (paid, fast & smart)
#   "openai/gpt-4o"                          (paid, most capable)
MODEL = os.getenv("MODEL", "anthropic/claude-3.5-haiku")

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Site metadata sent to OpenRouter (shows up in your dashboard)
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "https://github.com/eu-job-hunter")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "EU Job Hunter")

# EU countries to target
EU_LOCATIONS = os.getenv(
    "EU_LOCATIONS",
    "France,Germany,Netherlands,Spain,Portugal,Poland,Czech Republic,Sweden,Switzerland",
).split(",")

# Output directory
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./Ali_out"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  OPENROUTER CLIENT SETUP
#  One async client  → wired into the Agents SDK
#  One sync client   → used inside @function_tool bodies
# ════════════════════════════════════════════════════════════════════════════

# Extra headers OpenRouter recommends for attribution / rate-limit tiers
_OR_HEADERS = {
    "HTTP-Referer": YOUR_SITE_URL,
    "X-Title": YOUR_SITE_NAME,
}

# Async client — handed to the Agents SDK
_async_or_client = openai.AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers=_OR_HEADERS,
)

# Sync client — used inside tool functions (they are normal sync Python)
_sync_or_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers=_OR_HEADERS,
)

# Tell the Agents SDK to use our OpenRouter async client globally
set_default_openai_client(_async_or_client, use_for_tracing=False)

# Disable Anthropic-platform tracing (not applicable here)
set_tracing_disabled(True)

# Build the model object the SDK uses for every agent
_OR_MODEL = OpenAIChatCompletionsModel(
    model=MODEL,
    openai_client=_async_or_client,
)


# ════════════════════════════════════════════════════════════════════════════
#  HELPER — synchronous LLM call via OpenRouter
# ════════════════════════════════════════════════════════════════════════════


def _llm(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """Fire a single synchronous chat completion through OpenRouter."""
    response = _sync_or_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    # Strip markdown fences if the model wraps JSON in them
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
#  CV TEXT EXTRACTION  (PDF / DOCX → plain text)
# ════════════════════════════════════════════════════════════════════════════


def _extract_pdf(path: str) -> str:
    if PdfReader is None:
        raise ImportError("Run: pip install PyPDF2")
    reader = PdfReader(path)
    return "\n\n".join(
        p.extract_text().strip() for p in reader.pages if p.extract_text()
    )


def _extract_docx(path: str) -> str:
    if DocxDocument is None:
        raise ImportError("Run: pip install python-docx")
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts)


def extract_cv_text(cv_path: str) -> str:
    p = Path(cv_path)
    if not p.exists():
        raise FileNotFoundError(f"CV not found: {cv_path}")
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(str(p))
    if ext in (".docx", ".doc"):
        return _extract_docx(str(p))
    raise ValueError(f"Unsupported format '{ext}'. Use .pdf or .docx")


# ════════════════════════════════════════════════════════════════════════════
#  SERPER WEB SEARCH HELPER
# ════════════════════════════════════════════════════════════════════════════


def serper_search(query: str, num_results: int = 10, tbs: str = "") -> list[dict]:
    """Single Google search via Serper.dev. Returns list of result dicts."""
    try:
        payload: dict = {
            "q": query,
            "num": num_results,
            "gl": "eu",
            "hl": "en",
        }
        if tbs:
            payload["tbs"] = tbs
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
                "date": r.get("date", ""),
            }
            for r in resp.json().get("organic", [])
        ]
    except requests.exceptions.RequestException as e:
        return [{"error": str(e), "query": query}]


# ════════════════════════════════════════════════════════════════════════════
#  AGENT TOOLS
# ════════════════════════════════════════════════════════════════════════════


def analyze_cv_and_extract_search_terms(cv_path: str) -> str:
    """
    Researcher tool: reads a CV (PDF or DOCX) via OpenRouter LLM and returns
    a structured JSON of all keywords needed for EU job searching.

    Args:
        cv_path: Path to the CV file (.pdf or .docx).

    Returns:
        JSON string with: job_titles, skills_technical, skills_soft,
        tools_frameworks, domains, certifications, languages_spoken,
        education, notable_achievements, suggested_search_queries.
    """
    # ── 1. Extract raw text ──────────────────────────────────────────────────
    try:
        raw_text = extract_cv_text(cv_path)
    except Exception as e:
        return json.dumps({"error": f"Could not read CV: {e}"})

    # ── 2. Ask the LLM (via OpenRouter) to extract structured terms ──────────
    prompt = f"""
You are an expert CV analyst. Read this CV and extract every piece of information
useful for searching European job opportunities.

Today is 27 June 2026. ONLY return postings from 2026 — ignore anything from 2025 or earlier.

CV TEXT:
{raw_text[:12000]}

Return ONLY a valid JSON object (no markdown, no explanation) with these exact keys:

{{
  "candidate_name": "...",
  "seniority_level": "intern|junior|mid|senior|lead",
  "job_titles": ["role titles matching this profile"],
  "skills_technical": ["hard technical skills"],
  "skills_soft": ["soft skills"],
  "tools_frameworks": ["specific tools, frameworks, libraries"],
  "domains": ["industry domains e.g. computer vision, NLP, fintech"],
  "certifications": ["certifications and courses"],
  "languages_spoken": ["human languages the candidate knows"],
  "education": ["degrees and institutions"],
  "notable_achievements": ["quantified or notable results from the CV"],
    "suggested_search_queries": [
    "Generate 15+ diverse Google search queries to find ONLY 2026 EU job
     postings for this profile (today is June 2026). Mix: site:linkedin.com,
     site:indeed.com, site:glassdoor.com, site:welcometothejungle.com, generic
     queries. Every query MUST include '2026'. Include queries in both English
     and French if the candidate speaks French."
  ]
}}
"""
    result = _llm(prompt, max_tokens=2500, temperature=0.2)

    try:
        parsed = json.loads(result)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return result  # return raw; orchestrator can still use it


def search_eu_job_opportunities(search_terms_json: str) -> str:
    """
    Web Search Agent tool: takes CV researcher JSON and fires an exhaustive
    set of Serper queries across EU job platforms.

    Args:
        search_terms_json: JSON string from analyze_cv_and_extract_search_terms.

    Returns:
        JSON string with all found opportunities, deduplicated and pre-scored.
    """
    try:
        terms = json.loads(search_terms_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON", "raw": search_terms_json[:400]})

    suggested = terms.get("suggested_search_queries", [])
    job_titles = terms.get("job_titles", [])
    tech_skills = terms.get("skills_technical", [])
    tools = terms.get("tools_frameworks", [])
    seniority = terms.get("seniority_level", "junior")

    # ── Build exhaustive query set ───────────────────────────────────────────
    queries: list[str] = list(suggested)

    # Location × role matrix — target 2026 only
    for title in job_titles[:4]:
        for country in EU_LOCATIONS[:5]:
            queries.append(f'"{title}" internship {country} 2026')
            queries.append(
                f'"{title}" {seniority} job {country} 2026 site:linkedin.com'
            )

    # Platform sweeps — 2026 only
    top_skills = " ".join(tech_skills[:3])
    quoted_titles = " OR ".join(f'"{t}"' for t in job_titles[:3])
    for platform in [
        "site:linkedin.com/jobs",
        "site:indeed.com",
        "site:glassdoor.com",
        "site:welcometothejungle.com",
        "site:jobs.eu",
    ]:
        queries.append(
            f"{quoted_titles} {top_skills} internship OR stage Europe 2026 {platform}"
        )

    # Tool-stack cluster
    if tools:
        queries.append(
            f"{' '.join(tools[:4])} internship Europe 2026 software engineer"
        )

    # Deduplicate preserving order
    seen_q: set[str] = set()
    unique_queries = [q for q in queries if not (q in seen_q or seen_q.add(q))]

    # ── Execute all queries via Serper with freshness (CONCURRENT) ──────────
    all_results: list[dict] = []
    seen_links: set[str] = set()
    errors: list[str] = []

    print(
        f"\n🔍 Executing {len(unique_queries)} search queries via Serper (past 3 months, concurrent)...\n"
    )

    def _search_one(query: str) -> list[dict]:
        return serper_search(query, num_results=10, tbs="qdr:m3")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        fut_to_query = {pool.submit(_search_one, q): q for q in unique_queries}
        done = 0
        for fut in concurrent.futures.as_completed(fut_to_query):
            done += 1
            query = fut_to_query[fut]
            print(f"  [{done:02d}/{len(unique_queries)}] {query[:80]}...")
            try:
                results = fut.result()
            except Exception as e:
                errors.append(f"Error on query '{query[:60]}': {e}")
                continue
            for r in results:
                if "error" in r:
                    errors.append(f"'{query[:40]}': {r['error']}")
                    continue
                link = r.get("link", "")
                if link and link not in seen_links:
                    seen_links.add(link)
                    r["source_query"] = query
                    all_results.append(r)

    # ── Keyword-match pre-scoring + date boost ───────────────────────────────
    kw_set = set(
        [t.lower() for t in job_titles]
        + [s.lower() for s in tech_skills[:8]]
        + [t.lower() for t in tools[:6]]
    )

    def _score(r: dict) -> int:
        blob = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        kw_score = sum(1 for kw in kw_set if kw in blob)
        date_str = r.get("date", "")
        date_boost = 10 if "2026" in date_str else 0
        return kw_score + date_boost

    all_results.sort(key=_score, reverse=True)

    print(
        f"\n✅ {len(all_results)} unique opportunities found across {len(unique_queries)} queries.\n"
    )
    return json.dumps(
        {
            "total_results": len(all_results),
            "queries_executed": len(unique_queries),
            "errors": errors[:5],
            "opportunities": all_results[:80],
        },
        ensure_ascii=False,
        indent=2,
    )


def write_opportunities_report(
    cv_terms_json: str,
    opportunities_json: str,
    output_format: str = "markdown",
) -> str:
    """
    Report Writer tool: LLM-ranks all opportunities via OpenRouter using the
    new weighted scoring criteria (Skills 40%, Seniority 20%, Remote 15%,
    Company 15%, Freshness 10%). Groups into TOP PICKS, GOOD FITS,
    WORTH EXPLORING. Excludes SKIP (score 1-2) from reports.

    Args:
        cv_terms_json:      JSON from analyze_cv_and_extract_search_terms.
        opportunities_json: JSON with full job data from Crawl4AI pipeline.
        output_format:      'markdown' (default) or 'json'.

    Returns:
        Paths to the written report files.
    """
    try:
        terms = json.loads(cv_terms_json)
        opps = json.loads(opportunities_json)
    except json.JSONDecodeError as e:
        return f"JSON parse error: {e}"

    candidate = terms.get("candidate_name", "Candidate")
    seniority = terms.get("seniority_level", "")
    job_titles = terms.get("job_titles", [])
    tech_skills = terms.get("skills_technical", [])

    # Build richer condensed format — preserve full JD + company context
    # so the LLM can score against the new weighted criteria
    raw_jobs = opps.get("opportunities", [])[:60]
    condensed = []
    for i, job in enumerate(raw_jobs):
        entry = {
            "id": i + 1,
            "title": (job.get("title") or "")[:120],
            "company": (job.get("company") or "")[:60],
            "location": (job.get("location") or "")[:80],
            "source_url": job.get("source_url") or job.get("link", ""),
            "platform": job.get("platform", "unknown"),
            "snippet": (job.get("snippet") or "")[:200],
            "raw_content": (job.get("raw_content") or job.get("snippet", ""))[:600],
            "company_context": (job.get("company_context") or "")[:300],
            "extraction_method": job.get("extraction_method", ""),
            "date": job.get("date", job.get("posted_date", "")),
            "salary": (job.get("salary") or "")[:80],
            "remote_policy": (job.get("remote_policy") or "")[:60],
        }
        condensed.append(entry)

    ranking_prompt = f"""{SCORING_SYSTEM_PROMPT}

CANDIDATE PROFILE:
- Name: {candidate} ({seniority})
- Target roles: {", ".join(job_titles[:5])}
- Key skills: {", ".join(tech_skills[:8])}
- Full profile: {json.dumps(terms, ensure_ascii=False)[:1500]}

OPPORTUNITIES TO RANK ({len(condensed)} total):
{json.dumps(condensed, ensure_ascii=False, indent=2)[:15000]}

IMPORTANT: Return ONLY a valid JSON array. No preamble, no markdown fences.
"""

    print("🤖 Ranking opportunities via OpenRouter (weighted scoring)...\n")
    ranked_text = _llm(ranking_prompt, max_tokens=6000, temperature=0.3)

    try:
        ranked = json.loads(ranked_text)
    except json.JSONDecodeError:
        print("  ⚠️  LLM returned invalid JSON — using unranked fallback")
        ranked = condensed

    # Normalize: handle both old format (fit_reason) and new format
    for r in ranked:
        if "fit_reason" in r and "match_reasons" not in r:
            r["match_reasons"] = [r["fit_reason"]]
            r["concerns"] = []
            r["recommended_angle"] = ""
        r.setdefault("match_reasons", [])
        r.setdefault("concerns", [])
        r.setdefault("recommended_angle", "")
        r.setdefault("company", "")
        r.setdefault("location", "")
        r.setdefault("salary", "")
        r.setdefault("remote_policy", "")
        r.setdefault("platform", "unknown")
        r.setdefault("apply_url", r.get("source_url", r.get("link", "#")))

    # Separate SKIP tier from those included in report
    ranked_valid = [
        r for r in ranked if r.get("tier") != "SKIP" and (r.get("score", 0) or 0) > 2
    ]
    ranked_skipped = [
        r for r in ranked if r.get("tier") == "SKIP" or (r.get("score", 0) or 0) <= 2
    ]
    if ranked_skipped:
        print(f"   Excluded {len(ranked_skipped)} SKIP-tier jobs from report")

    # ── Build output files ───────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"opportunities_{slug}.md"
    json_path = OUTPUT_DIR / f"opportunities_{slug}.json"

    def _tier_block(tier: str, emoji: str) -> str:
        items = [r for r in ranked_valid if r.get("tier") == tier]
        if not items:
            return ""
        lines = [f"\n## {emoji} {tier} ({len(items)} found)\n"]
        for item in items:
            title = item.get("title", "?")
            company = item.get("company", "")
            company_str = f" — {company}" if company else ""
            location = item.get("location", "")
            salary = item.get("salary", "")
            remote = item.get("remote_policy", "")
            meta_parts = [p for p in [location, remote, salary] if p]
            meta_str = f"  `{' | '.join(meta_parts)}`" if meta_parts else ""
            source = item.get("platform", "unknown")
            method = item.get("extraction_method", "")
            quality = (
                "📄 Full JD"
                if any(
                    k in method
                    for k in ["css_schema", "bm25", "direct_crawl", "stealth_crawl"]
                )
                else "📋 Snippet"
            )
            apply_url = item.get("apply_url", "#")
            score = item.get("score", "?")

            lines.append(
                f"### [{title}]({apply_url}){company_str}"
                f"  `[Score: {score}/10]`{meta_str}\n"
            )

            reasons = item.get("match_reasons", [])
            if reasons:
                lines.append(
                    "**Why it fits:**  \n"
                    + "  \n".join(f"- {r}" for r in reasons)
                    + "\n"
                )

            concerns = item.get("concerns", [])
            if concerns:
                lines.append(
                    "**⚠️ Concerns:**  \n"
                    + "  \n".join(f"- {c}" for c in concerns)
                    + "\n"
                )

            angle = item.get("recommended_angle", "")
            if angle:
                lines.append(f"**💡 Recommended angle:** {angle}\n")

            lines.append(f"`Source: {source} | Quality: {quality}`\n\n---\n")
        return "\n".join(lines)

    md = (
        textwrap.dedent(f"""
        # 🌍 European Job Opportunities Report
        **Generated:** {now}
        **Candidate:** {candidate} ({seniority})
        **Target roles:** {", ".join(job_titles[:5])}
        **Key skills:** {", ".join(tech_skills[:8])}
        **Total unique jobs found:** {opps.get("total_results", 0)}
        **Queries executed:** {opps.get("queries_executed", 0)}
        **Jobs scored:** {len(ranked_valid)} (SKIP excluded: {len(ranked_skipped)})
        **LLM backend:** {MODEL} via OpenRouter

        ---
    """).strip()
        + "\n"
    )

    md += _tier_block("TOP PICKS", "🔥")
    md += _tier_block("GOOD FITS", "✅")
    md += _tier_block("WORTH EXPLORING", "📌")

    md += textwrap.dedent(f"""
        ---

        ## 📊 Search Statistics
        | Metric | Value |
        |--------|-------|
        | Queries executed | {opps.get("queries_executed", 0)} |
        | Raw results found | {opps.get("total_results", 0)} |
        | Jobs scored | {len(ranked_valid)} |
        | SKIP (excluded) | {len(ranked_skipped)} |
        | Data sources | Serper + Crawl4AI full JD + Company enrichment |
        | Countries targeted | {", ".join(EU_LOCATIONS)} |
        | LLM model | {MODEL} |
        | Report generated | {now} |

        ## 🔑 Extracted Keywords
        **Roles:** {", ".join(terms.get("job_titles", []))}
        **Technical:** {", ".join(terms.get("skills_technical", []))}
        **Tools:** {", ".join(terms.get("tools_frameworks", []))}
        **Domains:** {", ".join(terms.get("domains", []))}
    """).strip()

    md_path.write_text(md, encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "meta": {
                    "candidate": candidate,
                    "seniority": seniority,
                    "target_roles": job_titles,
                    "key_skills": tech_skills,
                    "llm_model": MODEL,
                    "llm_backend": "OpenRouter",
                    "generated_at": now,
                    "total_raw": opps.get("total_results", 0),
                    "queries_executed": opps.get("queries_executed", 0),
                    "jobs_scored": len(ranked_valid),
                    "jobs_skipped": len(ranked_skipped),
                },
                "ranked_opportunities": ranked_valid,
                "skipped_opportunities": [
                    {
                        "title": r.get("title", "?"),
                        "score": r.get("score", 0),
                        "reason": r.get("concerns", []),
                    }
                    for r in ranked_skipped
                ],
                "search_errors": opps.get("errors", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        f"Reports written!\n"
        f"  📄 Markdown : {md_path}\n"
        f"  📦 JSON     : {json_path}\n"
        f"  📊 {len(ranked_valid)} opportunities scored | "
        f"{len(ranked_skipped)} SKIP excluded | "
        f"{opps.get('total_results', 0)} found in {opps.get('queries_executed', 0)} queries | "
        f"Model: {MODEL}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  AGENT DEFINITIONS  (all use _OR_MODEL → OpenRouter)
# ════════════════════════════════════════════════════════════════════════════

_cv_tool = function_tool(analyze_cv_and_extract_search_terms)
_search_tool = function_tool(search_eu_job_opportunities)
_report_tool = function_tool(write_opportunities_report)

CV_RESEARCHER_AGENT = Agent(
    name="CV Researcher Agent",
    model=_OR_MODEL,
    instructions=textwrap.dedent("""
        You are an expert CV analyst and technical recruiter.
        Call `analyze_cv_and_extract_search_terms` with the provided CV path
        and return the raw JSON result. No commentary.
    """),
    tools=[_cv_tool],
)

WEB_SEARCH_AGENT = Agent(
    name="EU Job Web Search Agent",
    model=_OR_MODEL,
    instructions=textwrap.dedent("""
        You are a European job market researcher.
        Call `search_eu_job_opportunities` with the CV terms JSON and return
        the full raw JSON result. Do not summarize — the orchestrator needs all data.
    """),
    tools=[_search_tool],
)

ORCHESTRATOR_AGENT = Agent(
    name="Job Hunt Orchestrator",
    model=_OR_MODEL,
    instructions=textwrap.dedent("""
        You are the main orchestrator of a European job-hunting pipeline.
        Execute the following three steps in strict order:

        STEP 1 — Call `analyze_cv_and_extract_search_terms` with the user's CV path.
                 Save the returned JSON as cv_terms.

        STEP 2 — Call `search_eu_job_opportunities` passing cv_terms as the argument.
                 Save the returned JSON as opportunities.

        STEP 3 — Call `write_opportunities_report` with:
                   cv_terms_json      = cv_terms (from Step 1)
                   opportunities_json = opportunities (from Step 2)
                   output_format      = "markdown"

        After Step 3, tell the user:
          • The report file paths
          • Total opportunities found
          • Top 3 recommended positions with their scores
          • 2-3 actionable next steps
        
        Never skip a step. Never invent results.
    """),
    tools=[
        _cv_tool,
        _search_tool,
        _report_tool,
    ],
)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

# ── URL deduplication ──────────────────────────────────────────────────────────

_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "ref",
    "source",
    "trk",
}


def _normalize_url(url: str) -> str:
    """Strip tracking params and trailing slash for deduplication."""
    try:
        p = urlparse(url)
        clean = {
            k: v
            for k, v in parse_qs(p.query).items()
            if k.lower() not in _TRACKING_PARAMS
        }
        return urlunparse(
            (
                p.scheme,
                p.netloc,
                p.path.rstrip("/"),
                "",
                urlencode(clean, doseq=True),
                "",
            )
        ).lower()
    except Exception:
        return url.lower()


def _deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs (same URL from Serper + direct crawl)."""
    seen: set[str] = set()
    out: list[dict] = []
    for job in jobs:
        key = _normalize_url(job.get("source_url", ""))
        if key not in seen:
            seen.add(key)
            out.append(job)
    return out


async def run_pipeline(cv_path: str) -> dict:
    print(
        textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   EU JOB HUNTER — Multi-Agent Pipeline (v2.0)           ║
    ║   Backend : OpenRouter ({MODEL:<28})║
    ║   CV      : {Path(cv_path).name:<46}║
    ╚══════════════════════════════════════════════════════════╝
    """),
        flush=True,
    )

    if not Path(cv_path).exists():
        print(f"❌  CV file not found: {cv_path}")
        sys.exit(1)

    # ── Step 1: CV Analysis (unchanged) ──────────────────────────────────────
    print("\n📄 Step 1/5: Analysing CV...", flush=True)
    cv_data_json = analyze_cv_and_extract_search_terms(cv_path)
    cv_data = json.loads(cv_data_json) if cv_data_json.startswith("{") else {}
    keywords = cv_data.get("skills_technical", []) + cv_data.get("tools_frameworks", [])
    roles = cv_data.get("job_titles", [])

    # ── Step 2: Serper Search (unchanged) ────────────────────────────────────
    print("\n🔍 Step 2/5: Searching job boards via Google...", flush=True)
    raw_search_json = search_eu_job_opportunities(cv_data_json)
    raw_search = json.loads(raw_search_json) if raw_search_json.startswith("{") else {}
    serper_results = raw_search.get("opportunities", [])
    serper_urls = [r.get("link", "") for r in serper_results if r.get("link")]
    serper_snippets = {
        r.get("link", ""): r.get("snippet", "") for r in serper_results if r.get("link")
    }
    print(f"   Found {len(serper_urls)} URLs from Serper", flush=True)

    # ── Step 3a: Crawl4AI — full JD scraping (NEW) ───────────────────────────
    print("\n🕷️  Step 3a: Crawling full job descriptions with Crawl4AI...", flush=True)
    scraped_jobs = await scrape_job_listings(
        urls=serper_urls,
        cv_keywords=keywords,
        snippets=serper_snippets,
        max_concurrent=int(os.getenv("CRAWL4AI_MAX_CONCURRENT", 10)),
    )

    # ── Step 3b: Crawl4AI — company enrichment (NEW) ─────────────────────────
    print(
        "\n🏢 Step 3b: Enriching company pages with Crawl4AI (concurrent)...",
        flush=True,
    )
    enriched = await enrich_companies(
        jobs=scraped_jobs,
        max_companies=int(os.getenv("COMPANY_ENRICH_MAX", 20)),
    )
    scraped_jobs = merge_enrichment_into_jobs(scraped_jobs, enriched)

    # ── Step 3c: Direct board discovery (NEW, optional) ──────────────────────
    direct_jobs: list[dict] = []
    if os.getenv("ENABLE_DIRECT_CRAWL", "true").lower() == "true":
        print("\n🌍 Step 3c: Direct EU board discovery with Crawl4AI...", flush=True)
        direct_jobs = await discover_jobs_direct(
            cv_keywords=keywords,
            target_roles=roles,
            max_per_board=int(os.getenv("DIRECT_CRAWL_MAX_PER_BOARD", 15)),
        )

    # Merge + deduplicate all sources before scoring
    # Without this, the same posting found by both Serper and direct crawl
    # would be scored twice with potentially different scores in the report.
    all_jobs = _deduplicate_jobs(scraped_jobs + direct_jobs)
    print(f"\n   Total unique jobs to score: {len(all_jobs)}", flush=True)

    # ── Step 4: Scoring agent — now has full JD context ──────────────────────
    print("\n🤖 Step 4/5: Scoring opportunities with LLM...", flush=True)
    scored_result = write_opportunities_report(
        cv_data_json,
        json.dumps(
            {
                "total_results": len(all_jobs),
                "queries_executed": raw_search.get("queries_executed", 0),
                "opportunities": all_jobs[:60],
            }
        ),
    )

    # ── Step 5: Report writer (unchanged — called inside write_opportunities_report)
    print("\n📝 Step 5/5: Report written to disk.")

    print("\n" + "═" * 62)
    print("✅  PIPELINE COMPLETE")
    print("═" * 62)
    print(scored_result)
    print("═" * 62)

    return {
        "output": scored_result,
        "total_jobs": len(all_jobs),
    }


def reconfigure(
    model: str | None = None,
    locations: str | None = None,
    output_dir: str | None = None,
):
    """Update global config at runtime (used by web frontend)."""
    global MODEL, _OR_MODEL, EU_LOCATIONS, OUTPUT_DIR
    if model:
        MODEL = model
        _OR_MODEL = OpenAIChatCompletionsModel(
            model=MODEL, openai_client=_async_or_client
        )
        CV_RESEARCHER_AGENT.model = _OR_MODEL
        WEB_SEARCH_AGENT.model = _OR_MODEL
        ORCHESTRATOR_AGENT.model = _OR_MODEL
    if locations:
        EU_LOCATIONS = [loc.strip() for loc in locations.split(",")]
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) < 2:
        print(
            textwrap.dedent("""
            EU Job Hunter — OpenRouter Edition

            Usage:
              python main.py <path/to/cv.pdf>
              python main.py <path/to/cv.docx>

            Required environment variables:
              OPENROUTER_API_KEY   — get one free at https://openrouter.ai
              SERPER_API_KEY       — free tier at https://serper.dev

            Optional:
              MODEL          default: anthropic/claude-3.5-haiku
              EU_LOCATIONS   comma-separated countries
              OUTPUT_DIR     default: ./output
              YOUR_SITE_URL  shown in OpenRouter dashboard
              YOUR_SITE_NAME shown in OpenRouter dashboard

            Example:
              export OPENROUTER_API_KEY=sk-or-...
              export SERPER_API_KEY=abc123
              python main.py ./my_cv.pdf
        """).strip()
        )
        sys.exit(0)

    asyncio.run(run_pipeline(sys.argv[1]))


if __name__ == "__main__":
    main()
