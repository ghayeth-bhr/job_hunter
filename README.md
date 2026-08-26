<div align="center">

# EU Job Hunter 🎯

**Finds internship / stage-PFE opportunities across Europe and Canada that match your CV, on demand, via Telegram.**

</div>

---

## What it does

You send `/search` to a Telegram bot. It:

1. Reads your CV once (cached — no need to re-analyze it every time).
2. Searches for **internship / stage-PFE postings only** (not full-time jobs) across free job APIs, Google search, direct job-board crawling, and a curated list of funded programs (Mitacs, DAAD RISE, Erasmus+, ...).
3. Filters out anything already reported in a previous run, and anything the candidate isn't actually eligible to apply to (citizenship/visa/work-authorization).
4. **Scores what's left deterministically** — a pure function, not an LLM — into tiers: **TOP PICKS / GOOD FITS / WORTH EXPLORING**.
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
     • Funded programs  — config/programs.yaml, hand-curated, not discovered
     • (optional, paid) — Apify actors for LinkedIn / Welcome to the Jungle / Indeed
      │
      ▼
3. Deduplicate + drop anything already seen in a past run (SQLite)
   (funded programs are exempt — their urgency changes as the deadline nears)
      │
      ▼
4. Drop anything that isn't shaped like an internship
   (deterministic filter — catches what a query might have missed)
      │
      ▼
5. Eligibility check — deterministic exceptions first (funded program /
   fully remote / French stage-PFE / explicit sponsorship language), an
   LLM only for what's genuinely ambiguous after that
      │
      ▼
6. Deterministic scoring (tools/scoring.py) — skill/role/level/recency,
   no LLM, no network, cannot fail
      │
      ▼
7. Report written (.md + .json) and sent back over Telegram
```

Along the way, the pipeline is deliberately loud about anything it *can't* do — an expired API key, an exhausted free quota, an eligibility check that couldn't be verified — rather than silently producing a thinner report. Those show up in the report as clearly labeled banners, not missing rows.

### Why scoring isn't an LLM call

It used to be. Batched, retried, scored twice per batch to catch disagreement — because the same posting, sent to the same free-tier model twice, disagreed on tier something like 60% of the time. That's not judgment, it's noise, and no amount of retry/reconciliation machinery fixes a task that didn't need a model in the first place. `tools/scoring.py` replaces it with a pure function (ported from [pfzebi/PFE-Hunter](https://github.com/OussemaBenAmeur/pfzebi), a sibling project solving the identical problem, which reached the same conclusion independently): skill-fit measured against what the *posting* asks for, seniority read from the *leftmost* marker in the title (not the highest-weight one — "Summer Intern, Director of Product" is an internship), role-fit on discriminating token overlap, and a scam/unpaid-posting flag that multiplies the score down rather than averaging it away. Same input, same output, every time — no batching, no truncation, no retries, no cost, and a class of report-writing bug (silent gaps from a broken LLM response) that no longer exists because there's nothing left in that path to break.

The one thing that still needs actual reasoning — *can this specific candidate realistically apply here, given citizenship/visa/work authorization* — still uses an LLM, but only for the postings deterministic rules can't already resolve.

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
| `OPENROUTER_API_KEY` | ✅ | LLM backend — only used for CV analysis (once) and eligibility judgment on ambiguous postings |
| `SERPER_API_KEY` | ✅ | Google Search API (2,500 free queries, one-time) |
| `TELEGRAM_BOT_TOKEN` | ✅ | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | ✅ | Your Telegram user ID — only this ID can trigger a search |
| `TELEGRAM_CV_PATH` | ✅ | Path to the CV the bot searches against |
| `CANDIDATE_CITIZENSHIP` / `CANDIDATE_WORK_AUTHORIZATION` / `CANDIDATE_AVAILABILITY_START` / `CANDIDATE_AVAILABILITY_MONTHS` | recommended | Drives the eligibility check. Leave blank to skip eligibility filtering entirely. |

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

### Funded programs

`config/programs.yaml` is a small, hand-curated list of funded internship/research programs (Mitacs Globalink, DAAD RISE, Erasmus+, ...), each with a real deadline. It's deliberately *not* discovered by search or scraping — verified live that even a query naming a program directly doesn't reliably surface the program's own page in Google's results, only third-party mentions. Add your own entries as you find them; `last_verified` is there so you know when to double-check a deadline hasn't gone stale.

### Web UI (alternative to Telegram)

```bash
uv run uvicorn app:app --port 8000
```

Opens a simple upload-a-CV-and-run web page at `http://localhost:8000`, useful for a one-off run without going through Telegram.

## Project structure

```
main.py                    # the pipeline itself (search, filter, eligibility, score, report)
run_telegram_bot.py         # entry point for the Telegram bot
app.py                       # entry point for the web UI
tools/
  scoring.py                  # deterministic ranking engine — no LLM, no network
  programs.py                 # loads config/programs.yaml into opportunities
  job_apis.py                # free (+ optional paid) job-board API clients
  crawl4ai_scraper.py         # full job-description scraping
  direct_board_crawler.py     # direct EU board discovery
  company_enricher.py         # company-page enrichment
  store.py                    # SQLite "already reported" tracking
  telegram_bot.py             # bot commands, auth, run lock
  email_sender.py              # optional SendGrid email delivery
config/programs.yaml         # hand-curated funded programs (Mitacs, DAAD RISE, Erasmus+, ...)
data/                         # local state — seen-jobs DB, CV analysis cache (gitignored)
report_output/                # generated reports
tests/                        # real, runnable tests — no mocked-away load-bearing logic
```

## Cost

Free by default: Arbeitnow/Remotive/RemoteOK/Jobicy/Bundesagentur need no key, Adzuna's free tier covers normal usage, Serper's one-time free allotment lasts months at this query volume, and the eligibility-check LLM model is free-tier and only called for ambiguous postings (scoring itself is free — a pure function). The only thing that costs real money is Apify (LinkedIn/WTTJ/Indeed scraping), which is opt-in and off unless you explicitly enable it.

## Docs

- [`docs/PFZEBI_COMPARISON.md`](docs/PFZEBI_COMPARISON.md) — what changed and why: the deterministic scoring engine and funded-programs file were ported from a sibling project, replacing this project's LLM-based ranking entirely.
