"""
Core Crawl4AI scraping tool.

Strategy by URL type:
  - Known platforms (LinkedIn, Indeed, WTTJ, Glassdoor)
    → JsonCssExtractionStrategy (free, instant, no LLM)
  - Unknown URLs
    → fit_markdown + BM25ContentFilter (free, query-focused)
  - LinkedIn (often blocked)
    → stealth attempt → Bing Cache fallback → snippet fallback

NOTE on Google Cache: Google shut down its public cache service in late 2024.
The _try_linkedin() fallback uses Bing Cache (cc.bingj.com) instead.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# ── Load all platform schemas once at module import ───────────────────────────

SCHEMAS: dict[str, JsonCssExtractionStrategy] = {}


def _load_schemas():
    schemas_dir = Path("schemas")
    if not schemas_dir.exists():
        return
    for schema_file in schemas_dir.glob("*.json"):
        platform = schema_file.stem
        with open(schema_file) as f:
            SCHEMAS[platform] = JsonCssExtractionStrategy(json.load(f))


_load_schemas()


# ── Platform detection ─────────────────────────────────────────────────────────

PLATFORM_MAP = {
    "linkedin.com": "linkedin",
    "indeed.com": "indeed",
    "welcometothejungle.com": "welcometothejungle",
    "glassdoor.com": "glassdoor",
    "remoteok.com": "remoteok",
}


def detect_platform(url: str) -> Optional[str]:
    for domain, platform in PLATFORM_MAP.items():
        if domain in url:
            return platform
    return None


# ── LinkedIn-specific fallback ─────────────────────────────────────────────────


async def _try_linkedin(crawler: AsyncWebCrawler, url: str, snippet: str) -> dict:
    """
    LinkedIn blocks most scrapers. Three-level fallback:

    1. Stealth crawl
       Success rate is highly variable — depends on IP reputation,
       datacenter vs. residential proxy, and LinkedIn's current detection.
       Do not rely on a fixed %; treat it as "sometimes works".
       If this fails consistently, go straight to Level 3 and invest
       in the Apify integration described at the end of Phase 7.

    2. Bing Cache
       Google Cache was shut down in late 2024 — never use it.
       Bing Cache (cc.bingj.com) is still operational as of mid-2026.
       Availability per URL is not guaranteed.

    3. Serper snippet (last resort)
       The LLM still scores it — just with less context.
    """

    # Attempt 1: stealth crawl with human-like behaviour
    stealth_config = CrawlerRunConfig(
        wait_until="networkidle",
        delay_before_return_html=3.0,
        js_code="window.scrollTo(0, 600);",
        page_timeout=25000,
    )
    r = await crawler.arun(url=url, config=stealth_config)
    if r.success and len(r.markdown.fit_markdown or "") > 500:
        return {
            "source_url": url,
            "platform": "linkedin",
            "extraction_method": "stealth_crawl",
            "raw_content": r.markdown.fit_markdown[:3000],
            "title": r.metadata.get("title", ""),
        }

    # Attempt 2: Bing Cache
    # NOTE: Google Cache (webcache.googleusercontent.com) was discontinued in late 2024.
    # Use Bing Cache instead. Availability per URL is not guaranteed.
    bing_cache_url = f"https://cc.bingj.com/cache.aspx?q={quote(url, safe='')}"
    r2 = await crawler.arun(
        url=bing_cache_url, config=CrawlerRunConfig(page_timeout=15000)
    )
    if r2.success and len(r2.markdown.fit_markdown or "") > 500:
        return {
            "source_url": url,
            "platform": "linkedin",
            "extraction_method": "bing_cache",
            "raw_content": r2.markdown.fit_markdown[:3000],
        }

    # Fallback: snippet from Serper
    return {
        "source_url": url,
        "platform": "linkedin",
        "extraction_method": "snippet_fallback",
        "raw_content": snippet,
    }


# ── Main scraping function ─────────────────────────────────────────────────────


async def scrape_job_listings(
    urls: list[str],
    cv_keywords: list[str],
    snippets: dict[str, str] = None,
    max_concurrent: int = 10,
) -> list[dict]:
    """
    Takes the URL list from Serper. Returns full job data for each URL.

    Args:
        urls:           All job URLs returned by Serper
        cv_keywords:    Flat list of skills + titles from CV (used for BM25)
        snippets:       {url: snippet_text} from Serper (used as fallback)
        max_concurrent: Max parallel browser tabs

    Returns:
        List of job dicts with full description and metadata
    """
    snippets = snippets or {}

    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,
        viewport_width=1280,
        viewport_height=900,
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=75,
        max_session_permit=max_concurrent,
        rate_limiter=RateLimiter(
            base_delay=(1.5, 3.5),  # Respectful — job boards ban aggressive crawlers
            max_delay=60.0,
            max_retries=2,
        ),
    )

    # Bucket URLs by platform
    linkedin_urls: list[str] = []
    platform_batches: dict[str, list[str]] = {}
    unknown_urls: list[str] = []

    for url in urls:
        platform = detect_platform(url)
        if platform == "linkedin":
            linkedin_urls.append(url)
        elif platform and platform in SCHEMAS:
            platform_batches.setdefault(platform, []).append(url)
        else:
            unknown_urls.append(url)

    all_results: list[dict] = []

    sem = asyncio.Semaphore(max_concurrent)

    async def _crawl_one(url: str, cfg: CrawlerRunConfig) -> object | None:
        async with sem:
            print(f"     ↓ {url[:100]}", flush=True)
            r = await crawler.arun(url=url, config=cfg)
            if r.success:
                print(f"     ✓ {url[:80]}", flush=True)
            else:
                print(f"     ✗ {url[:80]}", flush=True)
            return r

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # ── Batch A: Known platforms — CSS extraction (free, instant) ─────────
        for platform, batch_urls in platform_batches.items():
            print(
                f"  🕷️  Scraping {len(batch_urls)} URLs from {platform} (CSS)...",
                flush=True,
            )

            cfg = CrawlerRunConfig(
                extraction_strategy=SCHEMAS[platform],
                cache_mode=CacheMode.ENABLED,
                wait_until="domcontentloaded",
                page_timeout=20000,
            )
            tasks = [_crawl_one(url, cfg) for url in batch_urls]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is None or not r.success or not r.extracted_content:
                    continue
                try:
                    data = json.loads(r.extracted_content)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        item.update(
                            {
                                "source_url": r.url,
                                "platform": platform,
                                "extraction_method": "css_schema",
                            }
                        )
                    all_results.extend(items)
                except (json.JSONDecodeError, TypeError):
                    pass

        # ── Batch B: Unknown platforms — BM25 fit_markdown ───────────────────
        if unknown_urls:
            bm25_query = " ".join(cv_keywords[:12])
            print(
                f"  🕷️  Scraping {len(unknown_urls)} unknown URLs (BM25 markdown)...",
                flush=True,
            )

            cfg = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=BM25ContentFilter(user_query=bm25_query)
                ),
                cache_mode=CacheMode.ENABLED,
                wait_until="domcontentloaded",
                page_timeout=20000,
            )
            tasks = [_crawl_one(url, cfg) for url in unknown_urls]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is None or not r.success or not r.markdown.fit_markdown:
                    continue
                all_results.append(
                    {
                        "source_url": r.url,
                        "platform": "unknown",
                        "extraction_method": "bm25_markdown",
                        "title": r.metadata.get("title", ""),
                        "raw_content": r.markdown.fit_markdown[:3000],
                    }
                )

        # ── Batch C: LinkedIn — stealth + fallbacks (sequential) ─────────────
        if linkedin_urls:
            print(
                f"  🔐 Handling {len(linkedin_urls)} LinkedIn URLs (stealth mode)...",
                flush=True,
            )
            for i, url in enumerate(linkedin_urls, 1):
                result = await _try_linkedin(crawler, url, snippets.get(url, ""))
                method = result.get("extraction_method", "?")
                print(
                    f"     [{i}/{len(linkedin_urls)}] {method}  {url[:60]}", flush=True
                )
                all_results.append(result)

    print(f"  ✅ Total jobs scraped: {len(all_results)}", flush=True)
    return all_results
