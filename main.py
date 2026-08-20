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

from dotenv import load_dotenv

load_dotenv()

# The pipeline's print statements are emoji-heavy; on Windows the default
# console codepage (cp1252) can't encode them and the whole run crashes
# mid-pipeline. Force UTF-8 stdout/stderr so an unattended daily run can't
# die on a print() call. No-op on platforms where stdout is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from tools.crawl4ai_scraper import scrape_job_listings
from tools.company_enricher import enrich_companies, merge_enrichment_into_jobs
from tools.direct_board_crawler import discover_jobs_direct
from tools.job_apis import (
    fetch_all_free_apis,
    fetch_all_paid_apis,
    _INTERN_SENIORITY,
    _INTERN_TITLE_HINT_PATTERN,
    SourceUnavailableError,
    CredentialExpiredError,
)
from tools.store import init_db, filter_new, mark_seen, normalize_url
from tools.email_sender import send_report_email
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
# "openai/gpt-oss-120b:free" was the default until a live run (2026-08-16)
# hit a 404 from OpenRouter: that slug was pulled from the free tier ("use
# this slug instead: openai/gpt-oss-120b" — the paid version). Confirmed via
# GET https://openrouter.ai/api/v1/models that it's genuinely gone from the
# free list; swapped the default to one confirmed still free at that time.
# Re-check https://openrouter.ai/models?max_price=0 if these stop working —
# the free roster rotates on its own schedule, not a fixed one.
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b:free")

# Fallback chain tried (in order) if MODEL 4xxs, rate-limits, or is pulled from
# the free tier. MODEL itself is always tried first.
FREE_MODEL_FALLBACKS = [
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
]

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Site metadata sent to OpenRouter (shows up in your dashboard)
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "https://github.com/eu-job-hunter")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "EU Job Hunter")

# Countries to target. Name is EU_LOCATIONS for historical/env-var-compat
# reasons, but the list also carries non-EU markets in scope (Canada) --
# it's consumed as a flat location list everywhere (Serper query matrix,
# Adzuna's per-country loop in tools/job_apis.py), so it's simplest to keep
# one list rather than thread a second "non-EU" list through every call site.
EU_LOCATIONS = os.getenv(
    "EU_LOCATIONS",
    "France,Germany,Netherlands,Spain,Portugal,Poland,Czech Republic,Sweden,Switzerland,Canada",
).split(",")

# Apify (paid) source scoping — trimmed for cost, confirmed 2026-08-17:
# Apify's Free plan has a hard $5/month cap with NO overage option (blocked
# until next billing cycle if exceeded — not a pay-a-bit-more situation).
# Real observed per-result costs: LinkedIn ~$0.002, WTTJ ~$0.003, Indeed
# ~$0.00017. Off by default (ENABLE_APIFY) since this costs real money,
# unlike every other source in this pipeline.
ENABLE_APIFY = os.getenv("ENABLE_APIFY", "false").lower() == "true"
# Fixed top-3 by expected AI/ML/CV job volume, not narrowed to
# French-speaking markets — candidate is open across all of EU_LOCATIONS +
# Canada; his CV being in French reflects his first language, not a market
# preference. Germany: largest EU tech/AI market by volume. France: surged
# AI hub (Mistral, Hugging Face presence) — confirmed via live smoke test.
# Canada: explicit target market, globally recognized AI research hub.
APIFY_LINKEDIN_COUNTRIES = os.getenv(
    "APIFY_LINKEDIN_COUNTRIES", "Germany,France,Canada"
).split(",")
# WTTJ itself is France/Francophone-market-focused by nature of the
# platform — this is not a candidate-targeting choice, unlike LinkedIn's
# scoping above.
APIFY_WTTJ_COUNTRIES = os.getenv("APIFY_WTTJ_COUNTRIES", "France").split(",")

# Output directory
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./reports"))
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
    """Fire a synchronous chat completion through OpenRouter.

    Tries MODEL first, then falls back through FREE_MODEL_FALLBACKS if it
    errors — free-tier model availability rotates, so a single hardcoded
    slug isn't reliable for an unattended daily run.
    """
    models_to_try = [MODEL] + [m for m in FREE_MODEL_FALLBACKS if m != MODEL]
    last_error: Exception | None = None

    for model_slug in models_to_try:
        try:
            response = _sync_or_client.chat.completions.create(
                model=model_slug,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise ValueError("empty completion")
            # Strip markdown fences if the model wraps JSON in them
            text = re.sub(r"^```(?:json)?\s*", "", text.strip())
            text = re.sub(r"\s*```$", "", text)
            return text.strip()
        except openai.AuthenticationError as e:
            # Confirmed live (2026-08-18): OpenRouter returns 401 with
            # body {"error": {"message": "User not found.", "code": 401}}
            # for an invalid/expired key -- the openai SDK surfaces this as
            # openai.AuthenticationError with .status_code == 401. This is a
            # credential problem, not per-model flakiness -- the SAME
            # OPENROUTER_API_KEY applies to every fallback model, so trying
            # the next one would just repeat the identical failure. Raise
            # immediately rather than burning through the whole chain.
            raise CredentialExpiredError(
                "OpenRouter", f"API key rejected (confirmed): {e.body}", confirmed=True
            ) from e
        except Exception as e:
            last_error = e
            print(f"  [WARN] Model '{model_slug}' failed ({e}); trying next fallback...")
            continue

    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")


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
    today = datetime.now()
    current_year = today.year
    prompt = f"""
You are an expert CV analyst. Read this CV and extract every piece of information
useful for searching European job opportunities.

Today is {today.strftime("%d %B %Y")}. ONLY return postings from {current_year} —
ignore anything postmarked {current_year - 1} or earlier.

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
    "Generate 8-10 diverse Google search queries to find ONLY {current_year} EU
     INTERNSHIP postings for this profile (today is {today.strftime("%B %Y")}) --
     this candidate wants a final/graduation internship (stage de fin d'etudes /
     PFE), NOT a full-time/permanent job. Every single query MUST contain an
     internship-marker word: 'internship', 'intern', 'stage', 'PFE', 'trainee',
     or 'working student' -- never generate a bare role-title query like
     'Machine Learning Engineer 2026' with no internship marker, since that
     surfaces full-time roles this candidate cannot apply to. Mix: site:linkedin.com,
     site:indeed.com, site:glassdoor.com, site:welcometothejungle.com, generic
     queries. Every query MUST include '{current_year}'. Include queries in both English
     and French if the candidate speaks French (French queries should use 'stage'
     or 'PFE', not 'internship')."
  ]
}}
"""
    # 2500 was too tight for this schema in practice — a live run (2026-08-16)
    # truncated mid-string on a free model's response (it produced verbose
    # output before finishing the requested fields), which then crashed a
    # downstream unguarded parse. Bumped for headroom; the parse site in
    # run_pipeline() is now also guarded regardless (see json.JSONDecodeError
    # handling around cv_data_json below).
    result = _llm(prompt, max_tokens=4000, temperature=0.2)

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
    domains = terms.get("domains", [])
    languages = terms.get("languages_spoken", [])
    speaks_french = any(
        "french" in lang.lower() or "français" in lang.lower() for lang in languages
    )
    current_year = datetime.now().year

    # ── Build a LEAN query set ───────────────────────────────────────────────
    # Kept intentionally small (~15-20 total): tools/job_apis.py now covers
    # the bulk of discovery via free structured APIs (Arbeitnow, Adzuna,
    # Remotive, RemoteOK, Jobicy, Bundesagentur). Serper is only for the gaps
    # those APIs miss — company career pages and platform-specific listings.
    queries: list[str] = list(suggested[:8])

    # Location × role matrix — small, targeted
    for title in job_titles[:2]:
        for country in EU_LOCATIONS[:3]:
            queries.append(f'"{title}" internship {country} {current_year}')

    # Canada gets an explicit query regardless of the [:3] slice above (it
    # sits later in EU_LOCATIONS) and regardless of whether Adzuna
    # credentials are configured (tools/job_apis.py's Adzuna fetch silently
    # no-ops without ADZUNA_APP_ID/KEY) — Serper is the channel guaranteed
    # to be live, so Canada coverage shouldn't depend on a key that may not
    # be set up yet.
    if "Canada" in EU_LOCATIONS:
        for title in job_titles[:2]:
            queries.append(f'"{title}" internship Canada {current_year}')

    # Platform sweeps — top 2 platforms only
    top_skills = " ".join(tech_skills[:3])
    quoted_titles = " OR ".join(f'"{t}"' for t in job_titles[:3])
    for platform in ["site:linkedin.com/jobs", "site:welcometothejungle.com"]:
        queries.append(
            f"{quoted_titles} {top_skills} internship OR stage Europe {current_year} {platform}"
        )

    # Tool-stack cluster
    if tools:
        queries.append(
            f"{' '.join(tools[:4])} internship Europe {current_year} software engineer"
        )

    # French-language "PFE"/"stage" pass — PRIMARY French institutional terms,
    # not an English query with "OR stage" bolted on as an afterthought.
    # "PFE" (Projet/Stage de Fin d'Études) is the actual vocabulary used by
    # Francophone engineering-school job markets — verified live in a prior
    # session: Inria, ALTEN Maroc, Sofrecom Tunisia, and Atos postings all
    # use this exact phrasing, and none surfaced from the English-only
    # queries above. Intentionally not site:-restricted or country-scoped —
    # a plain-text French query is what actually found those employers,
    # regardless of which country they're headquartered in.
    if speaks_french:
        for title in job_titles[:2]:
            queries.append(f"stage PFE {title} {current_year}")
            queries.append(f"stage de fin d'études ingénieur {title} {current_year}")
        for domain in domains[:2]:
            queries.append(f"stage PFE ingénieur {domain} {current_year}")

    # Funded research-internship PROGRAMS pass — distinct from job-board
    # postings. Investigation (2026-08-18): a real, highly-relevant match
    # (Mitacs Globalink Research Internship, Canada) never surfaced from ANY
    # job-title-locked query above ("AI Engineer Intern" internship Canada,
    # site:linkedin.com/indeed.com) -- verified live via direct Serper calls.
    # It only appeared for a generic "research internship ... international
    # students" query or the program's own name. These academic/fellowship
    # programs use program vocabulary ("Globalink", "fully funded", "research
    # internship"), not job titles, so the job-title-locked queries above
    # structurally can't reach them. Named-program query follows the same
    # precedent as the French PFE pass above: hardcoded, verified-live
    # vocabulary for a specific, well-known, highly-relevant target rather
    # than a generic keyword guess.
    for domain in domains[:2]:
        queries.append(f"funded research internship {domain} students {current_year}")
    if "Canada" in EU_LOCATIONS:
        queries.append(f"Mitacs Globalink Research Internship {current_year}")

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
        date_boost = 10 if str(current_year) in date_str else 0
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


# ════════════════════════════════════════════════════════════════════════════
#  BATCHED RANKING — retry + count-reconciliation
#
#  Investigation (2026-08-16, real induced tests against a live ~94-job
#  dataset) found the single-call ranking design has THREE distinct failure
#  modes, only one of which the JSON-parse check catches:
#    1. Large batch (60 jobs) hits the token ceiling and truncates mid-object
#       -> invalid JSON -> caught by the parse-failure fallback.
#    2. Even a small batch (15 jobs, well under the token budget) can come
#       back broken/short due to free-tier model flakiness unrelated to size.
#    3. The model returns a SYNTACTICALLY VALID, self-terminated array that
#       silently covers only a handful of the sent jobs (zero SKIPs, zero
#       errors) -- this is what runs 1 and 2 actually exhibited, and it
#       passes both "did json.loads() throw" and "was zero exceptions raised".
#  Batching alone only helps with #1. Retry alone doesn't catch #3, since a
#  short-but-valid response never trips a retry. Only comparing the actual
#  returned entries against what was SENT (by job identity, not position)
#  catches #3 -- that's what _reconcile_batch does.
# ════════════════════════════════════════════════════════════════════════════

RANKING_BATCH_SIZE = 10  # was 15 -- empirically too tight for the model's
# typical verbosity: successful full-batch responses ran 9,700-11,200 chars
# against a 6,000-token ceiling, so a majority of live calls truncated
# mid-object. A smaller batch needs less total output to cover every item.


# ════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC SENIORITY PRE-FILTER
#
#  Investigation (2026-08-17): the same two clearly-senior jobs (a Fastly
#  "Staff Engineer" and a Deloitte "Enterprise Architect") were correctly
#  SKIP-scored in 2 of 3 real LLM ranking passes, but scored a perfect
#  TOP PICKS 10/10 in the third -- a coin-flip-grade error on an
#  unambiguous seniority mismatch for a junior/intern candidate. A cheap,
#  deterministic keyword check on the title removes this class of job
#  before it ever reaches the LLM, so the outcome doesn't depend on the
#  model getting it right.
# ════════════════════════════════════════════════════════════════════════════

_SENIOR_TITLE_PATTERN = re.compile(
    r"\b(staff|principal|director|vp|vice president|head of|chief|"
    r"distinguished|executive|svp|evp|enterprise architect)\b",
    re.IGNORECASE,
)


def _senior_title_match(title: str) -> str | None:
    """Returns the matched senior keyword, or None if the title isn't a
    genuine seniority mismatch.

    A bare keyword hit isn't enough: "Internship -- Assistant to the
    Director of Engineering" contains "Director" but IS a legitimate
    internship, not a director role. Reusing tools/job_apis.py's
    intern-title-hint regex (rather than re-deriving the same "is this
    actually an internship" check) means a title that also reads as an
    internship/stage/trainee posting is never excluded, regardless of
    which senior keyword it happens to contain.
    """
    match = _SENIOR_TITLE_PATTERN.search(title or "")
    if not match:
        return None
    if _INTERN_TITLE_HINT_PATTERN.search(title or ""):
        return None
    return match.group(1)


def _filter_internship_only(jobs: list[dict], seniority: str) -> list[dict]:
    """Drops any job whose title has no internship marker, for an
    intern/student/junior candidate.

    tools/job_apis.py already applies this to its own free-API sources
    (fetch_all_free_apis), but Serper (scraped_jobs), direct board crawling
    (direct_jobs), and Apify (paid_api_jobs) have no such filter -- a query
    like "Machine Learning Engineer 2026" with no internship marker
    surfaces full-time roles a final/graduation-internship candidate can't
    apply to. Reuses the same word-boundary intern-hint regex as
    _senior_title_match for consistency (same false-positive risks already
    solved there -- e.g. "International" must never match).
    """
    if seniority.lower() not in _INTERN_SENIORITY:
        return jobs
    return [j for j in jobs if _INTERN_TITLE_HINT_PATTERN.search(j.get("title", "") or "")]


def _build_ranking_prompt(batch: list[dict], terms: dict) -> str:
    candidate = terms.get("candidate_name", "Candidate")
    seniority = terms.get("seniority_level", "")
    job_titles = terms.get("job_titles", [])
    tech_skills = terms.get("skills_technical", [])
    return f"""{SCORING_SYSTEM_PROMPT}

CANDIDATE PROFILE:
- Name: {candidate} ({seniority})
- Target roles: {", ".join(job_titles[:5])}
- Key skills: {", ".join(tech_skills[:8])}
- Full profile: {json.dumps(terms, ensure_ascii=False)[:1500]}

OPPORTUNITIES TO RANK ({len(batch)} total):
{json.dumps(batch, ensure_ascii=False, indent=2)}

IMPORTANT: Return ONLY a valid JSON array with exactly {len(batch)} entries —
one per opportunity listed above. Each entry MUST include the same "id" value
as its corresponding input opportunity, so it can be matched back. No
preamble, no markdown fences.
"""


def _reconcile_batch(
    batch: list[dict], parsed_entries
) -> tuple[dict[int, dict], list[dict]]:
    """Match parsed LLM output back to the sent batch by id/URL, not position.

    Returns (matched: {sent_item_id: parsed_entry}, missing: [sent_items]).
    A syntactically valid array that only covers some of `batch` is exactly
    the case this exists to catch — position-based matching would silently
    accept it.
    """
    if not isinstance(parsed_entries, list):
        return {}, list(batch)

    by_id = {item["id"]: item for item in batch}
    by_url = {
        normalize_url(item.get("source_url", "")): item
        for item in batch
        if item.get("source_url")
    }

    matched: dict[int, dict] = {}
    for parsed in parsed_entries:
        if not isinstance(parsed, dict):
            continue
        sent_item = None
        pid = parsed.get("id")
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            pid = None
        if pid is not None and pid in by_id:
            sent_item = by_id[pid]
        else:
            url = normalize_url(parsed.get("source_url") or parsed.get("apply_url") or "")
            if url and url in by_url:
                sent_item = by_url[url]
        if sent_item is not None:
            # Force the id onto the stored entry rather than trusting
            # whatever the model echoed back — some responses omit "id"
            # entirely even though they were correctly matched by URL, and a
            # downstream consumer (double-scoring's pass-A/pass-B alignment)
            # needs to be able to trust entry["id"] without re-deriving it.
            parsed["id"] = sent_item["id"]
            matched[sent_item["id"]] = parsed

    missing = [item for item in batch if item["id"] not in matched]
    return matched, missing


def _rank_batch_with_retry(batch: list[dict], terms: dict) -> tuple[list[dict], list[dict]]:
    """Ranks one batch, retrying the whole batch once on any gap.

    Returns (ranked_entries, unranked_fallback_entries) — every job in
    `batch` appears in exactly one of the two lists.
    """
    prompt = _build_ranking_prompt(batch, terms)

    def _attempt() -> tuple[dict[int, dict], list[dict], bool]:
        raw = _llm(prompt, max_tokens=8000, temperature=0.3)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}, list(batch), False
        matched, missing = _reconcile_batch(batch, parsed)
        return matched, missing, True

    matched1, missing1, parsed_ok1 = _attempt()

    if not missing1:
        return list(matched1.values()), []

    print(
        f"  [WARN] Ranking batch: {len(missing1)}/{len(batch)} missing after "
        f"attempt 1 ({'parse failed' if not parsed_ok1 else 'reconciliation gap'})"
        f" — retrying batch..."
    )
    matched2, missing2, parsed_ok2 = _attempt()

    # Union: a job matched in EITHER attempt is rescued. Retry's value wins
    # if matched in both (a fresher sample), but nothing found in attempt 1
    # is thrown away just because the retry didn't also find it.
    merged = {**matched1, **matched2}
    still_missing = [item for item in batch if item["id"] not in merged]

    unranked_fallback = []
    for item in still_missing:
        reason = (
            "parse failure both attempts"
            if not parsed_ok1 and not parsed_ok2
            else "missing after retry (batch parsed, item never appeared in output)"
        )
        unranked_fallback.append(
            {
                **item,
                "score": None,
                "tier": "UNRANKED",
                "concerns": [
                    f"Automated ranking could not score this job — {reason}."
                ],
                "_unranked_reason": reason,
            }
        )

    if unranked_fallback:
        print(
            f"  [WARN] {len(unranked_fallback)}/{len(batch)} still UNRANKED "
            f"after retry:"
        )
        for item in unranked_fallback:
            print(
                f"           - {item.get('title', '?')[:70]!r} | "
                f"{item.get('source_url', '')[:80]} | "
                f"reason: {item['_unranked_reason']}"
            )

    return list(merged.values()), unranked_fallback


# Investigation (2026-08-17): comparing independent scoring passes across
# today's runs found 60-62% of comparable jobs disagreed on TIER, not just
# the two senior-title cases the pre-filter already handles -- swings like
# WORTH EXPLORING(3) <-> TOP PICKS(10) on ordinary intern-appropriate jobs.
# Every batch is now scored twice; where the two passes disagree on tier,
# the job is tagged DISPUTED with both raw scores shown, rather than
# averaging (fabricates false confidence) or silently picking one.
DOUBLE_SCORE_RANKING = True


def _double_score_batch(
    batch: list[dict], terms: dict
) -> tuple[list[dict], dict]:
    """Ranks one batch twice independently and reconciles the two passes.

    Returns (results, batch_stats). `results` covers every job in `batch`
    exactly once: genuinely scored (both passes agreed), tier="DISPUTED"
    (both passes produced a real score but disagreed on tier -- both raw
    scores are preserved, not blended), or tier="UNRANKED" (neither pass
    could score it at all). `batch_stats` carries avg_delta/avg_abs_delta
    over jobs both passes actually scored, so a batch-wide calibration
    shift is visible instead of hiding inside per-job flags.
    """
    ranked_a, unranked_a = _rank_batch_with_retry(batch, terms)
    ranked_b, unranked_b = _rank_batch_with_retry(batch, terms)

    by_id_a = {r["id"]: r for r in ranked_a}
    by_id_b = {r["id"]: r for r in ranked_b}
    unranked_by_id_a = {r["id"]: r for r in unranked_a}
    unranked_by_id_b = {r["id"]: r for r in unranked_b}

    results: list[dict] = []
    deltas: list[float] = []
    disputed_count = 0

    for item in batch:
        id_ = item["id"]
        a = by_id_a.get(id_)
        b = by_id_b.get(id_)

        if a is not None and b is not None:
            score_a = a.get("score")
            score_b = b.get("score")
            if a.get("tier") == b.get("tier"):
                results.append(a)  # agreement -> pass A is canonical
            else:
                disputed_count += 1
                results.append(
                    {
                        **a,
                        "score": None,
                        "tier": "DISPUTED",
                        "match_reasons": [],
                        "concerns": [
                            f"Score disputed across two independent passes: "
                            f"{score_a}/10 ({a.get('tier')}) vs "
                            f"{score_b}/10 ({b.get('tier')}) — manual review "
                            f"recommended, not averaged or auto-resolved."
                        ],
                        "_pass_a": {"score": score_a, "tier": a.get("tier")},
                        "_pass_b": {"score": score_b, "tier": b.get("tier")},
                    }
                )
            if isinstance(score_a, (int, float)) and isinstance(score_b, (int, float)):
                deltas.append(score_b - score_a)
        elif a is not None:
            results.append(a)  # only pass A scored it -- nothing to compare
        elif b is not None:
            results.append(b)  # only pass B scored it
        else:
            # Neither pass could score it -- merge whichever fallback exists.
            fallback = unranked_by_id_a.get(id_) or unranked_by_id_b.get(id_) or {
                **item,
                "score": None,
                "tier": "UNRANKED",
                "concerns": ["Automated ranking could not score this job in either pass."],
            }
            results.append(fallback)

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    avg_abs_delta = sum(abs(d) for d in deltas) / len(deltas) if deltas else 0.0
    stats = {
        "compared": len(deltas),
        "disputed": disputed_count,
        "avg_delta": avg_delta,
        "avg_abs_delta": avg_abs_delta,
    }
    return results, stats


def _mark_batch_openrouter_unavailable(batch: list[dict]) -> list[dict]:
    return [
        {
            **item,
            "score": None,
            "tier": "UNRANKED",
            "concerns": [
                "OpenRouter API key invalid — ranking could not be attempted "
                "(not a parse failure, not model flakiness)."
            ],
            "_unranked_reason": "OpenRouter credential expired",
        }
        for item in batch
    ]


def _rank_opportunities_in_batches(
    condensed: list[dict], terms: dict
) -> tuple[list[dict], int, int, list[dict], bool]:
    """Ranks all condensed jobs in bounded-size batches with per-batch retry,
    count reconciliation, and (if DOUBLE_SCORE_RANKING) a second independent
    pass to catch score/tier disagreement that a single pass can't reveal.

    Returns (ranked, unranked_count, disputed_count, batch_stats,
    openrouter_unavailable). Every job in `condensed` is guaranteed to
    appear exactly once in `ranked` — genuinely scored, tier="UNRANKED", or
    tier="DISPUTED". Asserted, not just hoped for.

    A CredentialExpiredError from OpenRouter is a fundamentally different
    situation from a parse failure or model flakiness -- retrying the next
    batch won't help, the same OPENROUTER_API_KEY applies to all of them.
    Once confirmed dead, remaining batches are marked UNRANKED immediately
    with a distinct, clearly-labeled reason instead of each one
    independently re-discovering (and re-printing) the identical failure --
    that's exactly the "wall of noise with no clear cause" this is meant
    to avoid.
    """
    batches = [
        condensed[i : i + RANKING_BATCH_SIZE]
        for i in range(0, len(condensed), RANKING_BATCH_SIZE)
    ]
    all_ranked: list[dict] = []
    total_unranked = 0
    total_disputed = 0
    batch_stats: list[dict] = []
    openrouter_unavailable = False

    for batch_num, batch in enumerate(batches, 1):
        if openrouter_unavailable:
            all_ranked.extend(_mark_batch_openrouter_unavailable(batch))
            total_unranked += len(batch)
            continue

        print(f"🤖 Ranking batch {batch_num}/{len(batches)} ({len(batch)} jobs)"
              f"{' (double-scored)' if DOUBLE_SCORE_RANKING else ''}...")

        try:
            if DOUBLE_SCORE_RANKING:
                results, stats = _double_score_batch(batch, terms)
                stats["batch"] = batch_num
                batch_stats.append(stats)
                print(
                    f"   Batch {batch_num}: avg_delta(B-A)={stats['avg_delta']:+.2f} "
                    f"avg_abs_delta={stats['avg_abs_delta']:.2f} "
                    f"{stats['disputed']}/{len(batch)} disputed "
                    f"({stats['compared']} jobs comparable in both passes)"
                )
                unranked_entries = [r for r in results if r.get("tier") == "UNRANKED"]
            else:
                ranked_entries, unranked_entries = _rank_batch_with_retry(batch, terms)
                results = ranked_entries + unranked_entries
        except CredentialExpiredError as e:
            print(f"  [CREDENTIAL EXPIRED] {e.source}: {e.reason} — marking this "
                  f"and all remaining batches UNRANKED rather than retrying "
                  f"a guaranteed repeat failure.")
            openrouter_unavailable = True
            results = _mark_batch_openrouter_unavailable(batch)
            unranked_entries = results

        all_ranked.extend(results)
        total_unranked += len(unranked_entries)
        total_disputed += sum(1 for r in results if r.get("tier") == "DISPUTED")

    assert len(all_ranked) == len(condensed), (
        f"Ranking count mismatch: {len(all_ranked)} results for "
        f"{len(condensed)} input jobs — every job must be accounted for."
    )
    return all_ranked, total_unranked, total_disputed, batch_stats, openrouter_unavailable


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
    sources_unavailable = opps.get("sources_unavailable", [])

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

    # ── Deterministic seniority pre-filter ────────────────────────────────
    # Only for junior/intern/student candidates -- a senior/lead candidate
    # legitimately wants Staff/Principal postings, so this must not apply
    # universally.
    pre_filter_skipped: list[dict] = []
    to_rank = condensed
    if seniority.lower() in _INTERN_SENIORITY:
        to_rank = []
        for c in condensed:
            keyword = _senior_title_match(c.get("title", ""))
            if keyword:
                print(
                    f"  [PRE-FILTER] Excluding {c.get('title', '')[:70]!r} "
                    f"(matched keyword: {keyword!r}) | {c.get('source_url', '')}"
                )
                pre_filter_skipped.append(
                    {
                        **c,
                        "score": None,
                        "tier": "SKIP",
                        "concerns": [
                            f"Excluded by seniority pre-filter — title matched "
                            f"senior/exec keyword {keyword!r}, not LLM-scored."
                        ],
                    }
                )
            else:
                to_rank.append(c)
        if pre_filter_skipped:
            print(
                f"  [PRE-FILTER] {len(pre_filter_skipped)}/{len(condensed)} "
                f"jobs excluded before ranking (senior/exec title match)"
            )

    print(
        f"🤖 Ranking {len(to_rank)} opportunities via OpenRouter "
        f"(batches of {RANKING_BATCH_SIZE}, weighted scoring)...\n"
    )
    ranked, unranked_count, disputed_count, batch_stats, openrouter_unavailable = (
        _rank_opportunities_in_batches(to_rank, terms)
    )
    if openrouter_unavailable:
        sources_unavailable.append(
            {
                "source": "OpenRouter",
                "kind": "CREDENTIAL EXPIRED",
                "reason": "API key rejected during ranking (confirmed) — see console/logs for the exact call that failed",
                "confirmed": True,
            }
        )
    ranking_failed = unranked_count > 0
    if ranking_failed:
        print(
            f"  [ERROR] {unranked_count}/{len(to_rank)} opportunities could "
            f"not be ranked this run even after retry. Falling back to "
            f"unranked/unfiltered for those specific jobs so nothing is "
            f"silently dropped."
        )
    if disputed_count:
        print(
            f"  [WARN] {disputed_count}/{len(to_rank)} opportunities had "
            f"DISPUTED scores — two independent passes disagreed on tier. "
            f"Shown with both raw scores, not averaged."
        )

    # Pre-filter exclusions never reach the normalize loop below (they were
    # never LLM output), so give them the same defaults directly.
    for c in pre_filter_skipped:
        c.setdefault("match_reasons", [])
        c.setdefault("recommended_angle", "")
        c.setdefault("apply_url", c.get("source_url", "#"))

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

    # Separate SKIP tier from those included in report. Exclusion depends
    # ONLY on the model's own categorical tier judgment now, not a numeric
    # score floor. Measured against 95 real scored items from actual LLM
    # ranking output (2026-08-18): zero cases of a non-SKIP tier paired with
    # score <= 2 -- SKIP was 0-2, WORTH EXPLORING 3-5, GOOD FITS 5-7.6, TOP
    # PICKS 7-10, every single time. The old score>2 threshold was fully
    # redundant with tier=="SKIP" in observed practice; removing it is a
    # measured no-op, not a guessed simplification. (The old threshold also
    # needed special-case bypasses for UNRANKED/DISPUTED's score=None,
    # which are no longer needed now that tier is the only signal.)
    ranked_valid = [r for r in ranked if r.get("tier") != "SKIP"]
    ranked_skipped = [r for r in ranked if r.get("tier") == "SKIP"]
    ranked_skipped = pre_filter_skipped + ranked_skipped
    if ranked_skipped:
        print(f"   Excluded {len(ranked_skipped)} SKIP-tier jobs from report "
              f"({len(pre_filter_skipped)} by seniority pre-filter, "
              f"{len(ranked_skipped) - len(pre_filter_skipped)} by LLM)")

    # Every job in `condensed` must land in exactly one bucket: genuinely
    # ranked, LLM-tagged SKIP, pre-filter SKIP, or UNRANKED. This is the same
    # invariant _rank_opportunities_in_batches already asserts for `to_rank`
    # alone -- this checks it holds end-to-end, across the pre-filter split.
    assert len(ranked_valid) + len(ranked_skipped) == len(condensed), (
        f"Pre-filter + ranking accounting mismatch: "
        f"{len(ranked_valid)} valid + {len(ranked_skipped)} skipped != "
        f"{len(condensed)} total condensed jobs."
    )

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
            score = item.get("score")
            score_str = "N/A" if score is None else f"{score}/10"

            lines.append(
                f"### [{title}]({apply_url}){company_str}"
                f"  `[Score: {score_str}]`{meta_str}\n"
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

    # Banner is computed HERE, from the list actually being rendered — not
    # from a count captured back when batching happened — so it can't drift
    # out of sync with whatever ranked_valid actually contains by the time
    # the Markdown is built. Batching means failures are now per-batch/
    # partial rather than all-or-nothing, so this must say HOW MANY, not
    # just THAT it failed.
    unranked_in_report = len([r for r in ranked_valid if r.get("tier") == "UNRANKED"])
    disputed_in_report = len([r for r in ranked_valid if r.get("tier") == "DISPUTED"])
    total_this_run = len(ranked_valid) + len(ranked_skipped)
    if ranking_failed:
        md += (
            textwrap.dedent(f"""
            ## ⚠️ RANKING INCOMPLETE THIS RUN
            **{unranked_in_report}** of **{total_this_run}** opportunities could not
            be scored this run (LLM ranking failed or returned an incomplete
            response for one or more batches, even after a retry). They are
            shown **UNRANKED and UNFILTERED** below — no scoring, no SKIP
            exclusion applied. Review manually before trusting any as a strong
            match. Everything else in this report was ranked normally.

            ---
        """).strip()
            + "\n"
        )

    if disputed_in_report:
        md += (
            textwrap.dedent(f"""
            ## ⚖️ {disputed_in_report} OPPORTUNITIES HAD DISPUTED SCORES
            Every job below was ranked **twice, independently**. Where the two
            passes disagreed on tier, the job is shown here with **both raw
            scores** — not averaged, not auto-resolved to whichever was lower.
            Treat these as needing a manual look, not as confidently scored.

            ---
        """).strip()
            + "\n"
        )

    if sources_unavailable:
        source_lines = "\n".join(
            f"- **{s['source']}** — {s['kind']} ({'confirmed' if s['confirmed'] else 'heuristic, unverified'}): {s['reason']}"
            for s in sources_unavailable
        )
        md += (
            textwrap.dedent(f"""
            ## 🚫 {len(sources_unavailable)} SOURCE(S) UNAVAILABLE THIS RUN
            These sources were skipped this run — everything else in this
            report still ran normally, this just means less coverage from
            the affected source(s). A human needs to either add credit or
            renew a credential, not debug code.

            {source_lines}

            ---
        """).strip()
            + "\n"
        )

    md += _tier_block("UNRANKED", "⚠️")
    md += _tier_block("DISPUTED", "⚖️")
    md += _tier_block("TOP PICKS", "🔥")
    md += _tier_block("GOOD FITS", "✅")
    md += _tier_block("WORTH EXPLORING", "📌")

    ranking_status = "✅ OK"
    if unranked_in_report or disputed_in_report:
        ranking_status = (
            f"⚠️ {unranked_in_report} unranked, {disputed_in_report} disputed "
            f"of {total_this_run} — see sections above"
        )

    md += textwrap.dedent(f"""
        ---

        ## 📊 Search Statistics
        | Metric | Value |
        |--------|-------|
        | Queries executed | {opps.get("queries_executed", 0)} |
        | Raw results found | {opps.get("total_results", 0)} |
        | Jobs scored | {len(ranked_valid) - unranked_in_report - disputed_in_report} |
        | Disputed (two passes disagreed) | {disputed_in_report} |
        | Unranked (fallback) | {unranked_in_report} |
        | SKIP (excluded) | {len(ranked_skipped)} |
        | Ranking status | {ranking_status} |
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
                    "jobs_scored": len(ranked_valid) - unranked_in_report - disputed_in_report,
                    "jobs_skipped": len(ranked_skipped),
                    "jobs_unranked": unranked_in_report,
                    "jobs_disputed": disputed_in_report,
                    "ranking_failed": ranking_failed,
                    "double_score_batch_deltas": batch_stats,
                    "sources_unavailable": sources_unavailable,
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

    # RANKING_FAILED is a machine-checkable marker, not just decoration —
    # run_pipeline() greps this exact string to decide whether to flag the
    # email subject line, since an unattended cron run has no one watching
    # stdout for the "[ERROR]" print above.
    failure_prefix = ""
    if ranking_failed:
        failure_prefix += (
            f"⚠️ RANKING_FAILED — {unranked_in_report} of {total_this_run} "
            "opportunities could not be ranked this run (batch parse failure or "
            "incomplete response, even after retry); shown UNRANKED below.\n\n"
        )
    if disputed_in_report:
        failure_prefix += (
            f"⚖️ DISPUTED_SCORES — {disputed_in_report} of {total_this_run} "
            "opportunities got different tiers across two independent ranking "
            "passes; shown with both raw scores, not averaged.\n\n"
        )
    if sources_unavailable:
        names = ", ".join(s["source"] for s in sources_unavailable)
        failure_prefix += (
            f"🚫 SOURCE_UNAVAILABLE — {len(sources_unavailable)} source(s) skipped "
            f"this run ({names}); see report for credential/quota details.\n\n"
        )
    return (
        failure_prefix
        + f"Reports written!\n"
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
# URL normalization now lives in tools/store.py so the in-run dedup here and
# the cross-run "seen" tracking agree on what counts as "the same job".


def _deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    """Remove duplicate jobs (same URL from Serper, free APIs, direct crawl)."""
    seen: set[str] = set()
    out: list[dict] = []
    for job in jobs:
        key = normalize_url(job.get("source_url", ""))
        if key not in seen:
            seen.add(key)
            out.append(job)
    return out


async def run_pipeline(
    cv_path: str,
    from_email: str | None = None,
    to_email: str | None = None,
    email_subject: str | None = None,
) -> dict:
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

    # ── Step 1: CV Analysis ───────────────────────────────────────────────────
    print("\n📄 Step 1/5: Analysing CV...", flush=True)
    try:
        cv_data_json = analyze_cv_and_extract_search_terms(cv_path)
    except CredentialExpiredError as e:
        # If OpenRouter is dead, the run genuinely can't do much -- no CV
        # analysis means no job_titles/keywords to search with at all.
        # Fail loud and specific here rather than limping into a run that
        # would just produce a wall of confusing downstream noise with no
        # clear cause.
        print(f"  [CREDENTIAL EXPIRED] {e.source}: {e.reason}")
        return {
            "output": f"⚠️ Pipeline could not run — {e.source} credential expired: {e.reason}",
            "total_jobs": 0,
            "new_jobs": 0,
            "sources_unavailable": [
                {"source": e.source, "kind": "CREDENTIAL EXPIRED", "reason": e.reason, "confirmed": e.confirmed}
            ],
            "email": {},
        }
    # analyze_cv_and_extract_search_terms() already catches JSONDecodeError
    # internally and returns the raw (possibly invalid) text in that case —
    # but this second parse here was unguarded, so that same invalid text
    # crashed the whole run uncaught (hit live, 2026-08-16: a free model's
    # truncated response passed through as raw text and died here instead of
    # degrading gracefully).
    try:
        cv_data = json.loads(cv_data_json) if cv_data_json.startswith("{") else {}
    except json.JSONDecodeError:
        print(
            "  [WARN] CV analysis returned invalid/truncated JSON — "
            "proceeding with an empty profile. Search-term extraction below "
            "will have little/no keyword coverage this run."
        )
        cv_data = {}
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

    # ── Step 3c: Direct board discovery (optional) ───────────────────────────
    direct_jobs: list[dict] = []
    if os.getenv("ENABLE_DIRECT_CRAWL", "true").lower() == "true":
        print("\n🌍 Step 3c: Direct EU board discovery with Crawl4AI...", flush=True)
        direct_jobs = await discover_jobs_direct(
            cv_keywords=keywords,
            target_roles=roles,
            max_per_board=int(os.getenv("DIRECT_CRAWL_MAX_PER_BOARD", 15)),
        )

    # ── Step 3d: Free structured job APIs (NEW) ───────────────────────────────
    # These come back with full descriptions already — no scraping needed —
    # and cost nothing (Arbeitnow/RemoteOK/Remotive/Jobicy have no key at all;
    # Adzuna and Bundesagentur are free-tier/keyless).
    print("\n🆓 Step 3d: Querying free structured job APIs...", flush=True)
    free_api_jobs, sources_unavailable = fetch_all_free_apis(cv_data, EU_LOCATIONS)

    # ── Step 3e: Paid Apify sources (NEW, opt-in — costs real money) ─────────
    paid_api_jobs: list[dict] = []
    if ENABLE_APIFY:
        print(
            "\n💳 Step 3e: Querying paid Apify sources (LinkedIn/WTTJ/Indeed)...",
            flush=True,
        )
        paid_api_jobs, paid_sources_unavailable = fetch_all_paid_apis(
            cv_data,
            linkedin_countries=APIFY_LINKEDIN_COUNTRIES,
            wttj_countries=APIFY_WTTJ_COUNTRIES,
            indeed_countries=EU_LOCATIONS,
        )
        sources_unavailable += paid_sources_unavailable

    # Merge + deduplicate all sources before scoring
    # Without this, the same posting found by multiple sources would be
    # scored twice with potentially different scores in the report.
    all_jobs = _deduplicate_jobs(
        scraped_jobs + direct_jobs + free_api_jobs + paid_api_jobs
    )
    print(f"\n   Total unique jobs across all sources: {len(all_jobs)}", flush=True)

    # ── Internship-only backstop ──────────────────────────────────────────────
    # fetch_all_free_apis() already filters its own sources to internship-shaped
    # titles (tools/job_apis.py), and the CV-analysis prompt now asks for
    # internship-only search queries -- but Serper (scraped_jobs), direct board
    # crawling (direct_jobs), and Apify (paid_api_jobs) had no such filter, so a
    # query like "Machine Learning Engineer 2026" (a full-time role, no
    # internship marker) could still surface full-time postings for a candidate
    # who explicitly wants a final/graduation internship only. Applied here,
    # once, across the fully merged set, so no source can bypass it.
    seniority = cv_data.get("seniority_level", "")
    before_intern_filter = len(all_jobs)
    all_jobs = _filter_internship_only(all_jobs, seniority)
    if len(all_jobs) != before_intern_filter:
        print(
            f"   Internship-only filter: kept {len(all_jobs)}/{before_intern_filter} "
            f"(dropped non-internship titles -- candidate seniority_level={seniority!r})",
            flush=True,
        )

    # ── Persistence: only report jobs not already surfaced in a past run ─────
    db_conn = init_db()
    new_jobs = filter_new(all_jobs, conn=db_conn)
    already_seen = len(all_jobs) - len(new_jobs)
    print(
        f"   {len(new_jobs)} are NEW since the last run "
        f"({already_seen} already reported previously — skipped)",
        flush=True,
    )

    if not new_jobs:
        db_conn.close()
        print("\n✅ No new opportunities since the last run — nothing to report.")
        return {
            "output": "No new opportunities since the last run.",
            "total_jobs": 0,
            "new_jobs": 0,
            "sources_unavailable": sources_unavailable,
            "email": {},
        }

    # ── Step 4: Scoring agent — now has full JD context ──────────────────────
    print("\n🤖 Step 4/5: Scoring opportunities with LLM...", flush=True)
    scored_result = write_opportunities_report(
        cv_data_json,
        json.dumps(
            {
                "total_results": len(all_jobs),
                "queries_executed": raw_search.get("queries_executed", 0),
                "opportunities": new_jobs[:60],
                "sources_unavailable": sources_unavailable,
            }
        ),
    )

    # Only the jobs actually included in this run's report get marked seen —
    # so a scoring/write failure won't silently blackhole them from tomorrow.
    mark_seen(new_jobs, conn=db_conn)
    db_conn.close()

    # ── Step 5: Report writer
    print("\n📝 Step 5/5: Report written to disk.")

    # ── Step 6: Optional email report ─────────────────────────────────────────
    email_result = {}
    if os.getenv("ENABLE_EMAIL", "false").lower() == "true":
        print("\n📧 Step 6/6: Sending report via email...")
        json_match = re.search(r"JSON\s*:\s*(.+?\.json)", scored_result)
        if json_match:
            json_path = json_match.group(1).strip()
            from_addr = from_email or os.getenv("SENDGRID_FROM_EMAIL")
            to_addr = to_email or os.getenv("SENDGRID_TO_EMAIL")
            # Surface a ranking failure or unavailable source in the subject
            # line itself — an unattended cron run has no one watching
            # stdout, so the inbox is the only place this realistically
            # gets noticed.
            ranking_failed_this_run = "RANKING_FAILED" in scored_result
            source_unavailable_this_run = "SOURCE_UNAVAILABLE" in scored_result
            base_subject = (
                email_subject
                or f"EU Job Hunter Report — {datetime.now().strftime('%Y-%m-%d')}"
            )
            subject_flags = []
            if ranking_failed_this_run:
                subject_flags.append("⚠️ RANKING FAILED")
            if source_unavailable_this_run:
                subject_flags.append("🚫 SOURCE UNAVAILABLE")
            subject = (
                f"{' — '.join(subject_flags)} — {base_subject}"
                if subject_flags
                else base_subject
            )
            email_result = send_report_email(subject, json_path, from_addr, to_addr)
        else:
            print("   ⚠️  Could not find JSON report path")

    print("\n" + "═" * 62)
    print("✅  PIPELINE COMPLETE")
    print("═" * 62)
    print(scored_result)
    print("═" * 62)

    return {
        "output": scored_result,
        "total_jobs": len(all_jobs),
        "new_jobs": len(new_jobs),
        "email": email_result,
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
