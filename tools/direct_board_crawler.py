"""
Direct job board crawler — bypasses Google search entirely.

Uses BestFirstCrawlingStrategy so the most relevant listings
are visited first within the page budget per board.
This discovers opportunities that Serper misses.
"""

import asyncio
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
)
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy  # NOT BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


# EU-friendly job boards with public crawlable listings
EU_JOB_BOARDS = [
    {
        "url": "https://remoteok.com",
        "patterns": ["*/remote-*", "*/job-*"],
        "name": "RemoteOK",
    },
    {
        "url": "https://www.welcometothejungle.com/en/jobs",
        "patterns": ["*/jobs/*"],
        "name": "Welcome to the Jungle",
    },
    {
        "url": "https://euremotejobs.com",
        "patterns": ["*/jobs/*", "*/position/*"],
        "name": "EU Remote Jobs",
    },
    {
        "url": "https://remote.com/jobs",
        "patterns": ["*/jobs/*"],
        "name": "Remote.com",
    },
    {
        "url": "https://jobgether.com/offers",
        "patterns": ["*/offer/*", "*/offers/*"],
        "name": "Jobgether",
    },
]


async def discover_jobs_direct(
    cv_keywords: list[str],
    target_roles: list[str],
    max_per_board: int = 20,
    min_relevance_score: float = 0.3,
) -> list[dict]:
    """
    Crawls EU job boards directly and returns relevant job listings.

    Args:
        cv_keywords:         Skills and tools from CV
        target_roles:        Job titles the candidate is targeting
        max_per_board:       Max pages per board (controls runtime)
        min_relevance_score: Only keep pages scoring above this threshold

    Returns:
        List of job dicts with source_url, board_name, raw_content, score
    """
    all_keywords = cv_keywords + target_roles
    scorer = KeywordRelevanceScorer(keywords=all_keywords, weight=1.0)
    bm25_query = " ".join(all_keywords[:15])

    browser_config = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=900,
    )

    all_jobs: list[dict] = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for board in EU_JOB_BOARDS:
            print(f"  🌍 Crawling {board['name']} directly...", flush=True)

            try:
                config = CrawlerRunConfig(
                    deep_crawl_strategy=BestFirstCrawlingStrategy(
                        max_depth=2,
                        include_external=False,
                        url_scorer=scorer,
                        max_pages=max_per_board,
                        filter_chain=FilterChain(
                            [URLPatternFilter(patterns=board["patterns"])]
                        ),
                    ),
                    markdown_generator=DefaultMarkdownGenerator(
                        content_filter=BM25ContentFilter(user_query=bm25_query)
                    ),
                    cache_mode=CacheMode.ENABLED,
                    # stream=False: arun() returns a list — iterate normally.
                    # Do NOT use `async for result in await crawler.arun(...)` with
                    # stream=False; that raises TypeError on most Crawl4AI 0.9.x versions.
                    # If you upgrade to a version that changes this behavior, test first.
                    stream=False,
                )

                # arun() with stream=False returns a list, even for deep crawls
                results = await crawler.arun(url=board["url"], config=config)
                page_count = 0
                for result in results:
                    if not result.success:
                        continue

                    score = result.metadata.get("score", 0)
                    if score < min_relevance_score:
                        continue

                    content = result.markdown.fit_markdown or ""
                    if len(content) < 200:
                        continue

                    page_count += 1
                    title = result.metadata.get("title", "") or "untitled"
                    print(f"       ✓ [{score:.2f}] {title[:60]}", flush=True)

                    all_jobs.append(
                        {
                            "source_url": result.url,
                            "platform": board["name"],
                            "extraction_method": "direct_crawl",
                            "raw_content": content[:3000],
                            "relevance_score": round(score, 3),
                            "title": title,
                        }
                    )

                print(
                    f"     → {page_count} relevant pages from {board['name']}",
                    flush=True,
                )

            except Exception as e:
                print(f"    ⚠️  Failed on {board['name']}: {e}")
                continue

    print(f"  ✅ Direct crawl found {len(all_jobs)} relevant listings")
    return all_jobs
