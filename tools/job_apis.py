"""
Free, structured job-board APIs.

These replace fragile Google-search-then-scrape discovery for the sources
that already publish clean JSON: no browser, no blocking risk, no cost.

Each fetch_* function returns a list of normalized dicts:
    {source_url, title, company, location, platform, extraction_method,
     raw_content, date, salary, remote_policy}
so they can be merged directly into the same job list that
tools/crawl4ai_scraper.py and tools/direct_board_crawler.py produce.

Every function is defensive: a failed/changed endpoint returns an empty
list and prints a warning instead of breaking the whole pipeline run.
"""

import base64
import os
import re
from datetime import datetime, timezone

import requests

_TIMEOUT = 15


# ════════════════════════════════════════════════════════════════════════════
#  SOURCE-UNAVAILABLE ERRORS
#
#  Distinct from a generic request exception so a failed run is immediately
#  diagnosable as "needs a human to add credit/renew a key" vs. "something
#  broke in the code". Verification status differs by source (2026-08-17
#  research) -- see each classifier's docstring:
#    - Credential errors: CONFIRMED LIVE for both Adzuna and Apify (exact
#      status code + body shape reproduced and captured).
#    - Quota/credit-exhaustion errors: NOT reproduced live (would mean
#      deliberately burning through the real monthly cap for no benefit).
#      Adzuna documents nothing about this at all; Apify documents the
#      BEHAVIOR ("blocked until next monthly cycle") but not the exact wire
#      format. Detection below is a best-effort heuristic, not an exact
#      match -- flagged as such wherever it's used, not papered over.
# ════════════════════════════════════════════════════════════════════════════


class SourceUnavailableError(Exception):
    """Base class. Carries which source failed and a human-readable reason."""

    def __init__(self, source: str, reason: str, confirmed: bool):
        self.source = source
        self.reason = reason
        self.confirmed = confirmed  # True = exact verified match, False = heuristic
        super().__init__(f"{source}: {reason}")


class CredentialExpiredError(SourceUnavailableError):
    """Invalid/expired/revoked API credentials. Always confirmed=True here --
    both call sites only raise this on an exact, live-verified match."""


class QuotaExhaustedError(SourceUnavailableError):
    """Monthly usage cap / credit exhausted. confirmed=False unless a source
    ever documents (or we ever verify) an exact wire-level match."""


def _classify_adzuna_error(status_code: int, body: dict) -> Exception | None:
    """Confirmed live (2026-08-17): invalid app_id/app_key -> HTTP 401,
    body {"exception": "AUTH_FAIL", ...}.

    Adzuna's docs say nothing about the 1,000-calls/month quota's error
    response -- there is no confirmed or even documented shape for it, so
    quota exhaustion is NOT detected here at all rather than guessing at a
    shape with zero evidence behind it.
    """
    if status_code == 401 and body.get("exception") == "AUTH_FAIL":
        return CredentialExpiredError(
            "Adzuna", "app_id/app_key rejected (confirmed AUTH_FAIL)", confirmed=True
        )
    return None


def _classify_apify_error(status_code: int, body: dict) -> Exception | None:
    """Shared classifier for all three Apify-backed sources (same platform API).

    Confirmed live (2026-08-17): invalid/revoked token -> HTTP 401,
    body {"error": {"type": "user-or-token-not-found", ...}}.

    Quota/credit exhaustion is NOT confirmed at the wire level. Apify's docs
    only confirm the BEHAVIOR for Free-plan accounts ("blocked until the
    beginning of the next monthly cycle") -- no documented status code or
    error.type for it. This is a best-effort heuristic (4xx status commonly
    used for billing blocks, or "limit"/"quota"/"credit" in the message) --
    flagged via confirmed=False so callers/tests never mistake it for a
    verified match.
    """
    error = body.get("error") if isinstance(body, dict) else None
    error_type = (error or {}).get("type", "") if isinstance(error, dict) else ""
    message = (error or {}).get("message", "") if isinstance(error, dict) else ""

    if status_code == 401 and error_type == "user-or-token-not-found":
        return CredentialExpiredError(
            "Apify", "API token invalid/revoked (confirmed)", confirmed=True
        )
    if status_code in (402, 403, 429) or re.search(
        r"usage limit|monthly.*(limit|cap)|insufficient credit|quota", message, re.I
    ):
        return QuotaExhaustedError(
            "Apify",
            f"heuristic match on HTTP {status_code}"
            + (f' + message {message!r}' if message else ""),
            confirmed=False,
        )
    return None

# Best-effort ISO country name -> 2-letter code map for Adzuna.
# Adzuna's supported-country list has changed over time and isn't fully
# documented; we just try each mapped code and skip ones that 404/400.
_COUNTRY_CODES = {
    "france": "fr",
    "germany": "de",
    "netherlands": "nl",
    "spain": "es",
    "portugal": "pt",
    "poland": "pl",
    "czech republic": "cz",
    "sweden": "se",
    "switzerland": "ch",
    "belgium": "be",
    "italy": "it",
    "austria": "at",
    "ireland": "ie",
    "denmark": "dk",
    "finland": "fi",
    "luxembourg": "lu",
    "canada": "ca",
}


def _match(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    blob = text.lower()
    return any(kw.lower() in blob for kw in keywords if kw)


# Seniority levels where we should filter down to internship-shaped postings
# rather than the full firehose of (mostly senior/full-time) tech jobs.
_INTERN_SENIORITY = {"intern", "student", "junior"}
_INTERN_JOB_TYPES = {"intern", "student", "working student", "trainee", "student college"}

# Word-boundary regex, not a bare substring check — "intern" as a plain
# substring also matches "International", "Internet", "Internal", etc.
# (caught this live: an Affirm "Fullstack (International)" posting was
# slipping through the internship filter before this fix).
_INTERN_TITLE_HINT_PATTERN = re.compile(
    r"\b(intern|internship|stage|pfe|praktikum|praktikant|working student|"
    r"trainee|apprenti)\b",
    re.IGNORECASE,
)


def _looks_like_internship(job_types: list[str], title: str) -> bool:
    if any((t or "").strip().lower() in _INTERN_JOB_TYPES for t in job_types):
        return True
    return bool(_INTERN_TITLE_HINT_PATTERN.search(title or ""))


# ── Arbeitnow ──────────────────────────────────────────────────────────────
# Free, no key, no rate-limit documented. Returns latest EU/DACH tech jobs,
# newest first. No server-side keyword search, so we filter client-side.


def fetch_arbeitnow(
    keywords: list[str], seniority: str = "", max_pages: int = 2
) -> list[dict]:
    want_intern = seniority.lower() in _INTERN_SENIORITY
    # Intern/student-tagged postings are a small slice of the feed (roughly
    # 20% in spot checks) — crawl more pages so filtering down to them still
    # leaves a usable result set.
    if want_intern:
        max_pages = max(max_pages, 5)

    jobs: list[dict] = []
    try:
        for page in range(1, max_pages + 1):
            resp = requests.get(
                "https://www.arbeitnow.com/api/job-board-api",
                params={"page": page},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                break
            for job in data:
                title = job.get("title", "")
                blob = f"{title} {job.get('description', '')}"
                if not _match(blob, keywords):
                    continue
                if want_intern and not _looks_like_internship(
                    job.get("job_types", []), title
                ):
                    continue
                created = job.get("created_at")
                date_str = (
                    datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                    if created
                    else ""
                )
                jobs.append(
                    {
                        "source_url": job.get("url", ""),
                        "title": job.get("title", ""),
                        "company": job.get("company_name", ""),
                        "location": job.get("location", ""),
                        "platform": "arbeitnow",
                        "extraction_method": "api",
                        "raw_content": (job.get("description") or "")[:3000],
                        "date": date_str,
                        "salary": "",
                        "remote_policy": "remote" if job.get("remote") else "",
                    }
                )
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Arbeitnow API failed: {e}")
    return jobs


# ── Adzuna ───────────────────────────────────────────────────────────────────
# Free tier: ~1,000 calls/month. Needs ADZUNA_APP_ID + ADZUNA_APP_KEY
# (free signup at https://developer.adzuna.com). Skips silently if unset.


def fetch_adzuna(
    job_titles: list[str],
    eu_locations: list[str],
    max_days_old: int = 14,
    results_per_country: int = 15,
) -> list[dict]:
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        return []

    query = " ".join(job_titles[:3]) if job_titles else ""
    jobs: list[dict] = []

    for location in eu_locations:
        code = _COUNTRY_CODES.get(location.strip().lower())
        if not code:
            continue
        try:
            resp = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{code}/search/1",
                params={
                    "app_id": app_id,
                    "app_key": app_key,
                    "what": query,
                    "max_days_old": max_days_old,
                    "results_per_page": results_per_country,
                    "content-type": "application/json",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                # The same app_id/app_key is used for every country in this
                # loop, so a credential failure would repeat on every
                # remaining call -- classify once and stop immediately
                # rather than burning the rest of the loop on a guaranteed
                # repeat failure.
                try:
                    body = resp.json()
                except ValueError:
                    body = {}
                error = _classify_adzuna_error(resp.status_code, body)
                if error is not None:
                    raise error
                continue
            for job in resp.json().get("results", []):
                salary_min = job.get("salary_min")
                salary_max = job.get("salary_max")
                salary = (
                    f"{salary_min:.0f}-{salary_max:.0f} {job.get('salary_currency', 'EUR')}"
                    if salary_min and salary_max
                    else ""
                )
                jobs.append(
                    {
                        "source_url": job.get("redirect_url", ""),
                        "title": job.get("title", ""),
                        "company": (job.get("company") or {}).get("display_name", ""),
                        "location": (job.get("location") or {}).get(
                            "display_name", ""
                        ),
                        "platform": f"adzuna:{code}",
                        "extraction_method": "api",
                        "raw_content": (job.get("description") or "")[:3000],
                        "date": job.get("created", ""),
                        "salary": salary,
                        "remote_policy": "",
                    }
                )
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Adzuna ({code}) failed: {e}")
            continue

    return jobs


# ── Remotive ─────────────────────────────────────────────────────────────────
# Free, no key. Remote-first tech jobs; supports a basic `search` param.


def fetch_remotive(job_titles: list[str]) -> list[dict]:
    jobs: list[dict] = []
    search = job_titles[0] if job_titles else ""
    try:
        resp = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={"search": search} if search else {},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        for job in resp.json().get("jobs", []):
            jobs.append(
                {
                    "source_url": job.get("url", ""),
                    "title": job.get("title", ""),
                    "company": job.get("company_name", ""),
                    "location": job.get("candidate_required_location", ""),
                    "platform": "remotive",
                    "extraction_method": "api",
                    "raw_content": (job.get("description") or "")[:3000],
                    "date": job.get("publication_date", ""),
                    "salary": job.get("salary", "") or "",
                    "remote_policy": "remote",
                }
            )
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Remotive API failed: {e}")
    return jobs


# ── RemoteOK ─────────────────────────────────────────────────────────────────
# Free, no key. Requires a descriptive User-Agent or it may 403.


def fetch_remoteok(keywords: list[str]) -> list[dict]:
    jobs: list[dict] = []
    try:
        resp = requests.get(
            "https://remoteok.com/api",
            headers={"User-Agent": "eu-job-hunter (personal job search tool)"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # First element is a legal-notice object, not a job — skip it.
        for job in data[1:] if data else []:
            blob = f"{job.get('position', '')} {' '.join(job.get('tags', []))}"
            if not _match(blob, keywords):
                continue
            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")
            salary = (
                f"{salary_min}-{salary_max} USD" if salary_min and salary_max else ""
            )
            jobs.append(
                {
                    "source_url": job.get("url", ""),
                    "title": job.get("position", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", "") or "Remote",
                    "platform": "remoteok",
                    "extraction_method": "api",
                    "raw_content": (job.get("description") or "")[:3000],
                    "date": job.get("date", ""),
                    "salary": salary,
                    "remote_policy": "remote",
                }
            )
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  [WARN] RemoteOK API failed: {e}")
    return jobs


# ── Jobicy ───────────────────────────────────────────────────────────────────
# Free, no key. Remote jobs, supports geo/tag filters.


def fetch_jobicy(job_titles: list[str], geo: str = "europe", count: int = 50) -> list[dict]:
    jobs: list[dict] = []
    try:
        resp = requests.get(
            "https://jobicy.com/api/v2/remote-jobs",
            params={"count": count, "geo": geo},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        for job in resp.json().get("jobs", []):
            blob = f"{job.get('jobTitle', '')} {job.get('jobIndustry', '')}"
            if job_titles and not _match(blob, job_titles):
                continue
            jobs.append(
                {
                    "source_url": job.get("url", ""),
                    "title": job.get("jobTitle", ""),
                    "company": job.get("companyName", ""),
                    "location": job.get("jobGeo", ""),
                    "platform": "jobicy",
                    "extraction_method": "api",
                    "raw_content": (job.get("jobExcerpt") or "")[:3000],
                    "date": job.get("pubDate", ""),
                    "salary": job.get("annualSalaryMin", "") or "",
                    "remote_policy": "remote",
                }
            )
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Jobicy API failed: {e}")
    return jobs


# ── Bundesagentur für Arbeit (Jobsuche) ──────────────────────────────────────
# Germany's federal job agency has no *official* public API, but its own
# frontend calls a keyless endpoint (see bundesAPI/jobsuche-api on GitHub;
# verified live against pc/v6/jobs — v4 returns 403). Endpoint paths are
# reverse-engineered and may shift again — treat failures here as expected,
# not fatal. The search endpoint doesn't return a description or apply URL,
# so each hit needs one extra jobdetails call; the public job page itself is
# reconstructed from the reference number (also verified live).

_BA_HEADERS = {"X-API-Key": "jobboerse-jobsuche"}
_BA_BASE = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"


def fetch_bundesagentur(job_titles: list[str], max_results: int = 20) -> list[dict]:
    if not job_titles:
        return []
    jobs: list[dict] = []
    try:
        resp = requests.get(
            f"{_BA_BASE}/pc/v6/jobs",
            headers=_BA_HEADERS,
            params={"was": job_titles[0], "size": max_results},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        for job in resp.json().get("ergebnisliste", []):
            refnr = job.get("referenznummer", "")
            location = ""
            orte = job.get("stellenlokationen") or []
            if orte:
                location = (orte[0].get("adresse") or {}).get("ort", "")

            description = ""
            if refnr:
                try:
                    b64 = base64.b64encode(refnr.encode()).decode()
                    detail = requests.get(
                        f"{_BA_BASE}/pc/v4/jobdetails/{b64}",
                        headers=_BA_HEADERS,
                        timeout=_TIMEOUT,
                    )
                    if detail.status_code == 200:
                        description = detail.json().get(
                            "stellenangebotsBeschreibung", ""
                        )
                except requests.exceptions.RequestException:
                    pass

            jobs.append(
                {
                    "source_url": (
                        f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
                        if refnr
                        else ""
                    ),
                    "title": job.get("stellenangebotsTitel", ""),
                    "company": job.get("firma", ""),
                    "location": location,
                    "platform": "bundesagentur",
                    "extraction_method": "api",
                    "raw_content": (description or "")[:3000],
                    "date": job.get("datumErsteVeroeffentlichung", ""),
                    "salary": "",
                    "remote_policy": "",
                }
            )
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"  [WARN] Bundesagentur Jobsuche API failed (endpoint may have "
              f"changed, it's unofficial): {e}")
    return jobs


# ── Apify (paid, small per-result cost — NOT part of fetch_all_free_apis) ────
# Verified live 2026-08-17 against real actors with real (small) spend:
#   LinkedIn: curious_coder/linkedin-jobs-scraper  — ~$0.002/result observed
#   WTTJ:     clearpath/welcome-to-the-jungle-jobs-api — ~$0.003/result observed
#   Indeed:   valig/indeed-jobs-scraper — ~$0.00017/result observed
# All three use Apify's run-sync-get-dataset-items endpoint, which returns
# 201 (not 200) with the real dataset array on success -- do not gate on
# status_code == 200, that's a real bug this code base almost shipped.

_APIFY_TIMEOUT = 120


def _run_apify_actor_sync(actor_id: str, input_data: dict) -> list[dict]:
    """POSTs to run-sync-get-dataset-items and returns the dataset items.

    Raises CredentialExpiredError / QuotaExhaustedError (see
    _classify_apify_error) on a matching failure; raises
    requests.exceptions.RequestException for anything else, left for the
    caller to catch with the same defensive pattern as every other source.
    """
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        return []

    resp = requests.post(
        f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params={"timeout": _APIFY_TIMEOUT},
        json=input_data,
        timeout=_APIFY_TIMEOUT + 30,
    )
    # 201 Created on success (the body IS the dataset array), not 200 --
    # only treat it as an error path once neither success shape applies.
    if resp.status_code not in (200, 201):
        try:
            body = resp.json()
        except ValueError:
            body = {}
        error = _classify_apify_error(resp.status_code, body)
        if error is not None:
            raise error
        resp.raise_for_status()

    items = resp.json()
    return items if isinstance(items, list) else []


def fetch_apify_linkedin(
    query: str, countries: list[str], max_per_country: int = 15
) -> list[dict]:
    """Real, full LinkedIn job postings via Apify (paid, ~$0.002/result).

    `countries` should be a short, deliberately scoped list (2-3) -- this
    is the expensive-per-result source, unlike the free APIs above.
    """
    jobs: list[dict] = []
    for country in countries:
        items = _run_apify_actor_sync(
            "curious_coder~linkedin-jobs-scraper",
            {
                "keywords": query,
                "location": country,
                "limitPerSource": max_per_country,
                "scrapeCompany": False,
            },
        )
        for job in items:
            jobs.append(
                {
                    "source_url": job.get("link", ""),
                    "title": job.get("title", ""),
                    "company": job.get("companyName", ""),
                    "location": job.get("location", ""),
                    "platform": "apify:linkedin",
                    "extraction_method": "api",
                    "raw_content": (job.get("descriptionHtml") or job.get("description") or "")[:3000],
                    "date": job.get("postedAt", ""),
                    "salary": "",
                    "remote_policy": "",
                }
            )
    return jobs


def fetch_apify_wttj(query: str, countries: list[str] | None = None, max_items: int = 15) -> list[dict]:
    """Real Welcome to the Jungle postings via Apify (paid, ~$0.003/result).

    WTTJ itself is inherently France/Francophone-market-focused as a
    platform -- that's not a candidate-targeting choice, so `countries`
    defaults to just France. Use SHORT phrase queries: a specific long
    phrase (e.g. "stage PFE ingenieur intelligence artificielle") returned
    zero results in live testing; a short one ("stage intelligence
    artificielle") returned 15 real, current postings.
    """
    countries = countries or ["France"]
    jobs: list[dict] = []
    for country in countries:
        items = _run_apify_actor_sync(
            "clearpath~welcome-to-the-jungle-jobs-api",
            {
                "maxItems": max_items,
                "query": query,
                "location": country,
                "includeDetails": False,
            },
        )
        for job in items:
            jobs.append(
                {
                    "source_url": job.get("url", ""),
                    "title": job.get("name", ""),
                    "company": job.get("organizationName", ""),
                    "location": country,
                    "platform": "apify:wttj",
                    "extraction_method": "api",
                    "raw_content": (job.get("summary") or "")[:3000],
                    "date": job.get("publishedAt", ""),
                    "salary": "",
                    "remote_policy": job.get("remote", "") or "",
                }
            )
    return jobs


def fetch_apify_indeed(
    query: str, countries: list[str], max_per_country: int = 15
) -> list[dict]:
    """Real Indeed postings via Apify (paid, ~$0.00017/result -- by far the
    cheapest of the three, safe to call broadly across all target markets)."""
    jobs: list[dict] = []
    for country in countries:
        code = _COUNTRY_CODES.get(country.strip().lower())
        if not code:
            continue
        items = _run_apify_actor_sync(
            "valig~indeed-jobs-scraper",
            {
                "country": code,
                "title": query,
                "location": country,
                "limit": max_per_country,
            },
        )
        for job in items:
            employer = job.get("employer") or {}
            # description is a dict ({"text": ..., "html": ...}), not a
            # plain string -- caught live: a bare (x or "")[:3000] slice on
            # a dict raises KeyError(slice(...)), not a type error, so this
            # would NOT have been obvious from the traceback alone without
            # actually running it.
            description = job.get("description")
            description_text = (
                description.get("text", "") if isinstance(description, dict) else (description or "")
            )
            jobs.append(
                {
                    "source_url": job.get("url", ""),
                    "title": job.get("title", ""),
                    "company": employer.get("name", "") if isinstance(employer, dict) else "",
                    "location": country,
                    "platform": f"apify:indeed:{code}",
                    "extraction_method": "api",
                    "raw_content": description_text[:3000],
                    "date": job.get("datePublished", ""),
                    "salary": "",
                    "remote_policy": "",
                }
            )
    return jobs


def _call_source_safely(label: str, fn, *args, **kwargs) -> tuple[list[dict], dict | None]:
    """Calls a fetch_* function, catching SourceUnavailableError so one
    source's credential/quota failure never aborts the whole batch.

    Shared by fetch_all_free_apis (Adzuna) and fetch_all_paid_apis (the
    three Apify sources) so the "print + classify + continue" behavior
    is identical for both, not two hand-copied try/except blocks that
    could quietly drift apart.

    Returns (jobs, unavailable_info_or_None).
    """
    try:
        result = fn(*args, **kwargs)
        print(f"  [OK] {label}: {len(result)} jobs")
        return result, None
    except SourceUnavailableError as e:
        kind = "QUOTA EXHAUSTED" if isinstance(e, QuotaExhaustedError) else "CREDENTIAL EXPIRED"
        confidence = "confirmed" if e.confirmed else "heuristic, unverified"
        print(f"  [{kind}] {label}: {e.reason} ({confidence}) — skipping this "
              f"source for this run, continuing with the others.")
        return [], {"source": label, "kind": kind, "reason": e.reason, "confirmed": e.confirmed}
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] {label} failed (not a quota/credential issue): {e}")
        return [], None


def fetch_all_paid_apis(
    cv_terms: dict,
    linkedin_countries: list[str],
    wttj_countries: list[str],
    indeed_countries: list[str],
) -> tuple[list[dict], list[dict]]:
    """Calls the three paid Apify sources. Kept entirely separate from
    fetch_all_free_apis -- these cost real money and must stay opt-in and
    clearly scoped, not silently bundled into the $0 tier.

    Returns (jobs, sources_unavailable). sources_unavailable is a list of
    {"source", "reason", "confirmed"} dicts for anything that raised
    CredentialExpiredError/QuotaExhaustedError -- one source failing never
    aborts the others.
    """
    job_titles = cv_terms.get("job_titles", [])
    query = job_titles[0] if job_titles else "AI engineer intern"

    jobs: list[dict] = []
    sources_unavailable: list[dict] = []

    for label, fn, args in [
        ("Apify LinkedIn", fetch_apify_linkedin, (query, linkedin_countries)),
        ("Apify WTTJ", fetch_apify_wttj, (query, wttj_countries)),
        ("Apify Indeed", fetch_apify_indeed, (query, indeed_countries)),
    ]:
        result, unavailable = _call_source_safely(label, fn, *args)
        jobs.extend(result)
        if unavailable:
            sources_unavailable.append(unavailable)

    return jobs, sources_unavailable


# ── Orchestrator ─────────────────────────────────────────────────────────────


def fetch_all_free_apis(
    cv_terms: dict, eu_locations: list[str]
) -> tuple[list[dict], list[dict]]:
    """Calls every free structured job API and merges the results.

    Args:
        cv_terms: Parsed CV JSON (job_titles, skills_technical, tools_frameworks).
        eu_locations: Countries to target (from EU_LOCATIONS).

    Returns:
        (jobs, sources_unavailable) — jobs is the merged list (no dedup —
        caller handles that); sources_unavailable carries any
        CredentialExpiredError from Adzuna (the only source here with real
        credentials that can actually expire — the rest are keyless).
    """
    job_titles = cv_terms.get("job_titles", [])
    seniority = cv_terms.get("seniority_level", "")
    keywords = (
        job_titles
        + cv_terms.get("skills_technical", [])[:6]
        + cv_terms.get("tools_frameworks", [])[:4]
    )

    print("  Fetching free structured job APIs (Arbeitnow, Adzuna, "
          "Remotive, RemoteOK, Jobicy, Bundesagentur)...")

    all_jobs: list[dict] = []
    sources_unavailable: list[dict] = []
    all_jobs += fetch_arbeitnow(keywords, seniority=seniority)
    adzuna_jobs, adzuna_unavailable = _call_source_safely(
        "Adzuna", fetch_adzuna, job_titles, eu_locations
    )
    all_jobs += adzuna_jobs
    if adzuna_unavailable:
        sources_unavailable.append(adzuna_unavailable)
    all_jobs += fetch_remotive(job_titles)
    all_jobs += fetch_remoteok(keywords)
    all_jobs += fetch_jobicy(job_titles)
    all_jobs += fetch_bundesagentur(job_titles)

    # Adzuna/Remotive/RemoteOK/Jobicy/Bundesagentur have no internship-type
    # field to filter on server-side — for a student/intern profile, apply a
    # title-hint pass so a CV like this doesn't get buried under senior
    # full-time roles that merely happen to share tech keywords.
    if seniority.lower() in _INTERN_SENIORITY:
        before = len(all_jobs)
        all_jobs = [
            j for j in all_jobs if _looks_like_internship([], j.get("title", ""))
        ]
        print(f"  Filtered to internship-shaped titles: {len(all_jobs)}/{before}")

    print(f"  Free APIs returned {len(all_jobs)} candidate jobs "
          f"(before global dedup)")
    return all_jobs, sources_unavailable
