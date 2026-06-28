"""
Company enrichment tool.

For each unique company, crawls 3-4 pages of their website
(About, Careers, Tech Blog) using BestFirstCrawlingStrategy,
scoring pages by relevance to candidate and culture keywords.

Returns a dict {company_url: {pages_crawled, company_context}} for
the scoring agent to consume.
"""

import asyncio
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# Keywords that signal high-value company pages
ENRICHMENT_KEYWORDS = [
    "remote",
    "hybrid",
    "culture",
    "team",
    "values",
    "mission",
    "stack",
    "technology",
    "engineering",
    "backend",
    "frontend",
    "funding",
    "series",
    "growth",
    "benefits",
    "equity",
    "salary",
    "open source",
    "autonomy",
    "work life",
    "diversity",
]

# Only follow links that look like company culture/tech pages
VALUABLE_PATH_PATTERNS = [
    "*/about*",
    "*/team*",
    "*/culture*",
    "*/values*",
    "*/careers*",
    "*/jobs*",
    "*/tech*",
    "*/blog*",
    "*/engineering*",
    "*/life-at*",
    "*/work-with-us*",
]


def extract_company_homepage(job: dict) -> str | None:
    """
    Extract the company homepage URL from a scraped job dict.
    Handles LinkedIn, Indeed, and direct company URLs.
    """
    company_url = job.get("company_url") or job.get("company_website")
    if company_url:
        return company_url

    # Try to derive from source URL (skip aggregators)
    source = job.get("source_url", "")
    skip_domains = [
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
        "welcometothejungle.com",
        "remoteok.com",
    ]
    if not any(d in source for d in skip_domains):
        # Likely a direct company page
        from urllib.parse import urlparse

        parsed = urlparse(source)
        return f"{parsed.scheme}://{parsed.netloc}"

    return None


async def _enrich_one_company(
    crawler: AsyncWebCrawler,
    company_url: str,
    max_pages: int,
    scorer: KeywordRelevanceScorer,
) -> tuple[str, dict] | None:
    """Crawl one company site. Returns (url, enrichment_dict) or None on failure."""
    try:
        config = CrawlerRunConfig(
            deep_crawl_strategy=BestFirstCrawlingStrategy(
                max_depth=1,
                include_external=False,
                url_scorer=scorer,
                max_pages=max_pages,
                filter_chain=FilterChain(
                    [URLPatternFilter(patterns=VALUABLE_PATH_PATTERNS)]
                ),
            ),
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=BM25ContentFilter(
                    user_query="remote work culture team values tech stack engineering benefits"
                )
            ),
            cache_mode=CacheMode.ENABLED,
            page_timeout=20000,
            stream=False,
        )

        pages = await crawler.arun(url=company_url, config=config)

        if not pages:
            return None

        # Merge all pages into one context block (capped at 4000 chars)
        context_chunks = []
        total_chars = 0
        for page in pages:
            if page.success and page.markdown.fit_markdown:
                chunk = page.markdown.fit_markdown[:1200]
                context_chunks.append(f"[Source: {page.url}]\n{chunk}")
                total_chars += len(chunk)
                if total_chars >= 4000:
                    break

        if context_chunks:
            return company_url, {
                "pages_crawled": len(pages),
                "company_context": "\n\n---\n\n".join(context_chunks),
            }
        return None

    except Exception as e:
        print(f"    ⚠️  Failed to enrich {company_url}: {e}")
        return None


async def enrich_companies(
    jobs: list[dict],
    max_companies: int = 20,
    max_pages_per_company: int = 4,
) -> dict[str, dict]:
    """
    Crawls company websites to extract culture and tech stack signals.
    Runs concurrently across all companies (capped at max_companies).

    Args:
        jobs:                 List of scraped job dicts
        max_companies:        Cap to control runtime (20 is a good default)
        max_pages_per_company: Max pages to crawl per company (3-5 recommended)

    Returns:
        {company_homepage_url: {pages_crawled: int, company_context: str}}
    """

    # Deduplicate company URLs
    seen: set[str] = set()
    company_urls: list[str] = []
    for job in jobs:
        url = extract_company_homepage(job)
        if url and url not in seen:
            seen.add(url)
            company_urls.append(url)
        if len(company_urls) >= max_companies:
            break

    if not company_urls:
        return {}

    print(f"  🏢 Enriching {len(company_urls)} company websites (concurrent)...")

    scorer = KeywordRelevanceScorer(keywords=ENRICHMENT_KEYWORDS, weight=1.0)

    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,
        viewport_width=1280,
        viewport_height=900,
    )

    enriched: dict[str, dict] = {}

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Concurrent enrichment — all companies in parallel, not sequentially.
        # Sequential would cost ~160s for 20 companies; concurrent costs ~32s.
        tasks = []
        for i, url in enumerate(company_urls, 1):
            print(f"     [{i}/{len(company_urls)}] Enriching {url}")
            tasks.append(
                _enrich_one_company(crawler, url, max_pages_per_company, scorer)
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if result and not isinstance(result, Exception):
                url, data = result
                pages = data.get("pages_crawled", "?")
                print(f"     ✓ {url} ({pages} pages)")
                enriched[url] = data

    print(f"  ✅ Enriched {len(enriched)} companies successfully")
    return enriched


def merge_enrichment_into_jobs(
    jobs: list[dict],
    enriched: dict[str, dict],
) -> list[dict]:
    """Attach company_context to each job dict that has a matching company URL."""
    for job in jobs:
        company_url = extract_company_homepage(job)
        if company_url and company_url in enriched:
            job["company_context"] = enriched[company_url]["company_context"]
    return jobs
