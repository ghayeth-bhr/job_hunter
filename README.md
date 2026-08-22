<div align="center">

# EU Job Hunter 🎯

**Finds internship / stage-PFE opportunities across Europe and Canada that match your CV, on demand, via Telegram.**

</div>

---

## What it does

You send `/search` to a Telegram bot. It:

1. Reads your CV once (cached — no need to re-analyze it every time).
2. Searches for **internship / stage-PFE postings only** (not full-time jobs) across free job APIs, Google search, and direct job-board crawling.
3. Filters out anything already reported in a previous run.
4. Ranks what's left with an LLM (free-tier model) into tiers: **TOP PICKS / GOOD FITS / WORTH EXPLORING**.
5. Sends you back a summary + the full report (Markdown + JSON) on Telegram.

Runs for $0/month by default — no scheduled cron, no server to keep alive, just message the bot when you want a fresh search.

## How it works

```
Telegram /search
      │
      ▼
1. CV analysis (cached JSON — skipped after the first run)
      │
      ▼
2. Search for opportunities, in parallel:
     • Free job APIs   — Arbeitnow, Adzuna, Remotive, RemoteOK, Jobicy, Bundesagentur
     • Google search    — Serper.dev, internship/stage-scoped queries
     • Full-text scrape — Crawl4AI on the resulting URLs
     • Direct crawling  — a handful of EU job boards
     • (optional, paid) — Apify actors for LinkedIn / Welcome to the Jungle / Indeed
      │
      ▼
3. Deduplicate + drop anything already seen in a past run (SQLite)
      │
      ▼
4. Drop anything that isn't shaped like an internship
   (deterministic filter — catches what a query might have missed)
      │
      ▼
5. LLM ranks what's left, in small batches, twice per batch
   (disagreements are shown as DISPUTED, never silently averaged)
      │
      ▼
6. Report written (.md + .json) and sent back over Telegram
```

Along the way, the pipeline is deliberately loud about anything it *can't* do — an expired API key, an exhausted free quota, a ranking batch that failed — rather than silently producing a thinner report. Those show up in the report as clearly labeled banners, not missing rows.

## Setup

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo> && cd work
uv sync
cp .env.example .env   # fill in your keys — see below
```

### Minimum config (`.env`)

| Variable | Required | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | LLM backend — free-tier models work fine |
| `SERPER_API_KEY` | ✅ | Google Search API (2,500 free queries, one-time) |
| `TELEGRAM_BOT_TOKEN` | ✅ | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | ✅ | Your Telegram user ID — only this ID can trigger a search |
| `TELEGRAM_CV_PATH` | ✅ | Path to the CV the bot searches against |

Everything else (`ADZUNA_APP_ID`, `APIFY_API_TOKEN`, `EU_LOCATIONS`, ...) is optional and documented with cost/behavior notes in `.env.example`. Paid sources (Apify) are off by default (`ENABLE_APIFY=false`).

### Run it

```bash
uv run python run_telegram_bot.py
```

Leave this running in a terminal you control (not something that gets killed after a few minutes). Then message your bot:

- `/search` — run a search, get the report back on Telegram
- `/status` — check if a search is currently running

### CV analysis cache

The first `/search` needs a CV-analysis pass (an LLM call). If a `data/cv_analysis_cache.json` file exists, it's used directly and that step is skipped entirely — since your CV doesn't change run to run, there's no reason to re-analyze it (and re-analyzing it means depending on a free-tier model that has, in practice, occasionally been unreliable). Delete that file to force a fresh analysis next time.

### Web UI (alternative to Telegram)

```bash
uv run uvicorn app:app --port 8000
```

Opens a simple upload-a-CV-and-run web page at `http://localhost:8000`, useful for a one-off run without going through Telegram.

## Project structure

```
main.py                    # the pipeline itself (search, filter, rank, report)
run_telegram_bot.py         # entry point for the Telegram bot
app.py                       # entry point for the web UI
tools/
  job_apis.py                # free (+ optional paid) job-board API clients
  crawl4ai_scraper.py         # full job-description scraping
  direct_board_crawler.py     # direct EU board discovery
  company_enricher.py         # company-page enrichment
  store.py                    # SQLite "already reported" tracking
  telegram_bot.py             # bot commands, auth, run lock
  email_sender.py              # optional SendGrid email delivery
prompts/scoring_agent.py     # the ranking prompt
data/                         # local state — seen-jobs DB, CV analysis cache (gitignored)
report_output/                # generated reports (gitignored)
tests/                        # real, runnable tests — no mocked-away load-bearing logic
```

## Cost

Free by default: Arbeitnow/Remotive/RemoteOK/Jobicy/Bundesagentur need no key, Adzuna's free tier covers normal usage, Serper's one-time free allotment lasts months at this query volume, and the default LLM model is free-tier. The only thing that costs real money is Apify (LinkedIn/WTTJ/Indeed scraping), which is opt-in and off unless you explicitly enable it.
