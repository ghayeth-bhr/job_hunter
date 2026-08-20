<div align="center">

# EU Job Hunter 🎯

**Multi-Agent AI Pipeline — CV Analysis & European Job Search**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB) ![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI_Agents_SDK-0.17-412991) ![OpenRouter](https://img.shields.io/badge/OpenRouter-API-FF6B35) ![FastAPI](https://img.shields.io/badge/FastAPI-Web-009688)

</div>

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Orchestrator Agent      │
                    │  (coordinates all 3 steps)  │
                    └──────────┬──────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
   ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐
   │ CV Researcher  │ │ Web Search    │ │ Report Writer    │
   │ Agent          │ │ Agent         │ │ Tool             │
   │ (Step 1)       │ │ (Step 2)      │ │ (Step 3)         │
   └───────┬────────┘ └───────┬────────┘ └──────────────────┘
           │                  │
           ▼                  ▼
   ┌────────────────┐ ┌────────────────┐
   │  PyPDF2 /      │ │  Serper.dev    │
   │  python-docx   │ │  Google Search │
   │  + LLM parse   │ │  API           │
   └────────────────┘ └────────────────┘
```

## How It Works

The app uses the **OpenAI Agents SDK** with **3 specialized agents** running on **OpenRouter** (any LLM model). The entire pipeline is a single `Runner.run()` call to the orchestrator.

### Step 1 — CV Researcher Agent

- Reads your CV (PDF or DOCX) using `PyPDF2` or `python-docx`
- Sends raw text to the LLM which extracts structured data:
  - Job titles, technical skills, soft skills
  - Tools & frameworks, domains, certifications
  - Languages, education, notable achievements
  - **15+ suggested search queries** tailored to your profile
- Returns a clean JSON object with all extracted terms

### Step 2 — Web Search Agent

- Takes the JSON from Step 1 and generates **62+ targeted Google search queries**
- Queries target **2026 postings only** across EU countries
- Searches 5 platforms: LinkedIn, Indeed, Glassdoor, Welcome to the Jungle, jobs.eu
- **10 concurrent** Serper.dev API calls (ThreadPoolExecutor) for speed
- Uses `tbs=qdr:m3` (past 3 months) freshness filter
- Deduplicates results, pre-scores by keyword match + date boost

### Step 3 — Report Writer Tool

- LLM ranks all opportunities (score 1-10) based on profile fit
- Assigns tiers: **TOP PICKS** (8-10), **GOOD FITS** (5-7), **WORTH EXPLORING** (1-4)
- Heavily penalizes old/irrelevant postings
- Writes two files to the output directory:
  - `opportunities_<timestamp>.md` — Formatted Markdown report
  - `opportunities_<timestamp>.json` — Structured JSON data

### Step 4 (Optional) — Email Report

- Sends the structured JSON report as a styled HTML email via **SendGrid**
- Can be enabled from the frontend checkbox or via `ENABLE_EMAIL=true`
- Customizable sender, recipient, and subject line (avoids spam classification)
- HTML template includes tiered cards (TOP PICKS / GOOD FITS / WORTH EXPLORING) with scores, match reasons, and apply links

### LLM Backend

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Your Code   │ ──▶ │  OpenRouter  │ ──▶ │  Any LLM Model   │
│  (Agents SDK)│     │  API Gateway │     │  (GPT-4o, Claude,│
│              │     │              │     │   Llama, etc.)   │
└──────────────┘     └──────────────┘     └──────────────────┘
```

The Agents SDK connects to **OpenRouter** (an OpenAI-compatible API) instead of OpenAI directly. This lets you swap between 200+ models by changing one env variable — no code changes needed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Framework | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) |
| LLM Backend | [OpenRouter](https://openrouter.ai) (multi-model gateway) |
| Search API | [Serper.dev](https://serper.dev) (Google Search API) |
| Web Framework | [FastAPI](https://fastapi.tiangolo.com) |
| CV Parsing | PyPDF2, python-docx |
| Concurrency | ThreadPoolExecutor (10 parallel searches) |
| Streaming | Server-Sent Events (SSE) for live logs |
| Email Delivery | [SendGrid](https://sendgrid.com) (transactional email API) |

## Project Structure

```
├── main.py              # Core multi-agent pipeline (CLI)
├── app.py               # FastAPI web server
├── tools/
│   ├── crawl4ai_scraper.py   # Full JD scraping with Crawl4AI
│   ├── company_enricher.py   # Company website enrichment
│   ├── direct_board_crawler.py # Direct EU board discovery
│   └── email_sender.py       # SendGrid email report delivery
├── prompts/
│   └── scoring_agent.py      # Weighted scoring prompt
├── frontend/
│   ├── index.html       # Standalone SPA frontend
│   ├── nginx.conf       # Nginx reverse-proxy config
│   └── Dockerfile       # Nginx container build
├── templates/
│   └── index.html       # FastAPI-served frontend
├── docker-compose.yml   # Backend + frontend orchestration
├── Dockerfile           # Backend container build
├── uploads/             # Uploaded CVs (auto-created)
├── reports/             # Generated reports (configurable via OUTPUT_DIR)
├── .env                 # API keys + configuration
├── .env.example         # Environment variable template
└── pyproject.toml       # Python dependencies (uv)
```

## Data Flow

```
User uploads CV (PDF/DOCX)
        │
        ▼
FastAPI saves file → returns task_id
        │
        ▼
Background task starts pipeline:
  ┌──────────────────────────────────────────────────┐
  │  1. Orchestrator Agent receives task             │
  │     │                                            │
  │     ▼                                            │
  │  2. CV Researcher Agent                          │
  │     → calls analyze_cv_and_extract_terms()       │
  │     → LLM extracts skills, roles, queries        │
  │     │                                            │
  │     ▼                                            │
  │  3. Web Search Agent                             │
  │     → calls search_eu_job_opportunities()        │
  │     → 62 concurrent Serper API calls             │
  │     → Crawl4AI scrapes full job descriptions     │
  │     → Company enrichment via website crawl       │
  │     → Direct EU board discovery (optional)       │
  │     → dedup, pre-score results                   │
  │     │                                            │
  │     ▼                                            │
  │  4. Report Writer Tool                           │
  │     → calls write_opportunities_report()         │
  │     → LLM ranks & tiers opportunities            │
  │     → writes .md + .json files                   │
  │     │                                            │
  │     ▼                                            │
  │  5. Email Report (optional)                      │
  │     → Builds styled HTML from JSON report        │
  │     → Sends via SendGrid API                     │
  │     → Custom FROM/TO/subject fields              │
  └──────────────────────────────────────────────────┘
        │
        ▼
SSE streams logs → Frontend displays results
        │
        ▼
User downloads Markdown / JSON reports
```

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- API keys: [OpenRouter](https://openrouter.ai) + [Serper.dev](https://serper.dev)

### Setup

```bash
# Clone & enter directory
git clone <repo> && cd eu-job-hunter

# Install dependencies
uv sync

# Configure API keys (copy template)
cp .env.example .env
# Edit .env with your keys

# Run CLI (direct)
uv run python main.py my_cv.pdf

# Run web app
uv run python app.py
# → Open http://localhost:8000
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key |
| `SERPER_API_KEY` | ✅ | — | Serper.dev API key |
| `MODEL` | ❌ | `openai/gpt-4o-mini` | Any OpenRouter model slug |
| `EU_LOCATIONS` | ❌ | 10 EU countries | Comma-separated target countries |
| `OUTPUT_DIR` | ❌ | `./reports` | Report output directory |
| `CRAWL4AI_MAX_CONCURRENT` | ❌ | `10` | Parallel browser tabs for crawling |
| `COMPANY_ENRICH_MAX` | ❌ | `20` | Max companies to enrich per run |
| `ENABLE_DIRECT_CRAWL` | ❌ | `true` | Set `false` to skip direct board crawling |
| `DIRECT_CRAWL_MAX_PER_BOARD` | ❌ | `15` | Pages per direct board crawl |
| `SENDGRID_API_KEY` | ❌* | — | SendGrid API key (for email reports) |
| `SENDGRID_FROM_EMAIL` | ❌* | — | Default sender email address |
| `SENDGRID_TO_EMAIL` | ❌* | — | Default recipient email address |
| `ENABLE_EMAIL` | ❌ | `false` | Set `true` to send email after each run |

\* Required only if `ENABLE_EMAIL=true` or when using the email checkbox in the frontend.

## Email Reports

The pipeline can optionally email the ranked results as a styled HTML report via **SendGrid**.

### How to enable

**Via frontend:** Check "Send report via email", fill in the FROM/TO/Subject fields, and run the pipeline.

**Via CLI/env:** Set these in `.env`:
```env
ENABLE_EMAIL=true
SENDGRID_API_KEY=SG.your_api_key_here
SENDGRID_FROM_EMAIL=you@example.com
SENDGRID_TO_EMAIL=you@example.com
```

### What the email looks like

- Header with candidate name, opportunity count, and query stats
- Tiered result cards: **🔥 TOP PICKS**, **✅ GOOD FITS**, **📌 WORTH EXPLORING**
- Each card shows: title, company, score, snippet, match reasons, and concerns
- Apply links direct you to the job posting
- Clean, mobile-friendly HTML with inline styles

### Spam prevention

The subject line is customizable from the frontend. Using a personalized subject (e.g., *"EU Job Report — John Doe — June 2026"*) improves deliverability. The default fallback is *"EU Job Hunter Report — YYYY-MM-DD"*.

### Programmatic usage

```python
from tools.email_sender import send_report_email

result = send_report_email(
    subject="EU Job Report — Custom Subject",
    report_path="reports/opportunities_20260628_120000.json",
    from_email="sender@example.com",
    to_email="recipient@example.com",
)
# Returns {"status": "success", "status_code": 202}
# Or {"status": "error", "message": "..."}
```

## Why OpenRouter?

OpenRouter provides a **unified API** for 200+ LLMs. This project uses it to:

- **Avoid vendor lock-in** — swap models via one env variable
- **Free tier access** — Llama 3.3 70B, Gemma 3, Mistral 7B at $0
- **Fallback routing** — if one provider is down, OpenRouter fails over
- **Cost optimization** — use cheap models for structured extraction, smart ones for ranking
