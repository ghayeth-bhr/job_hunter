# Comparison with pfzebi/PFE-Hunter, and what changed here

[pfzebi](https://github.com/OussemaBenAmeur/pfzebi) is a sibling project solving
the identical problem — a Tunisian ENISo engineering student hunting PFE and
research-lab internships — built independently by a classmate. This document
records what was compared, what was ported directly into this project, what
was deliberately left out, and the concrete before/after.

## Why this comparison happened

The user asked for a side-by-side review, then asked for the two projects to
be "tuned" into a single best version: take the strongest technique from each,
discard what's weaker, and implement the result directly rather than just
report findings.

## What pfzebi does differently

pfzebi is a TypeScript/Node monorepo (Hono API, React dashboard, Drizzle+SQLite)
with roughly 90 structured source connectors (ATS auto-detection, ~90% of
discovery) and one design decision that turned out to matter more than the
tech stack: **ranking has no LLM in it at all.** Its own commit history
documents why — with an LLM removed from an earlier version and only four
cheap terms left, every score across 328 real rows fell between 0.32 and 0.43,
with 125 rows tied at exactly 0.34. It replaced that with a pure scoring
function, and treats visa/sponsorship evidence as a ranking signal that can
only ever raise a score, never exclude a row.

## What was taken, and why

### 1. Deterministic scoring — replaces the LLM-ranking subsystem entirely

This was the headline change. Every mechanism this project built earlier in
the same session to cope with LLM ranking being unreliable — batching,
retry-on-gap, double-scoring, the `DISPUTED` tier for when two passes
disagreed, `UNRANKED` for when a batch's response didn't parse — existed
*because* the same posting, scored twice by the same free-tier model,
disagreed on tier roughly 60% of the time. That's not judgment being applied
inconsistently; it's noise. pfzebi's `score.ts` reached the same conclusion
independently and removed the model rather than building machinery to cope
with it.

`tools/scoring.py` is a direct, tested port of that approach:

- **`skill_fit`** — the fraction of what the *posting* asks for that the
  candidate has, not the fraction of the candidate's skill list a posting
  happens to mention (the latter is ~10% for every posting, since no job asks
  for a whole CV — it discriminates nothing). Bayesian-smoothed with a neutral
  prior so a posting naming one skill you have isn't scored as a 100% match.
- **`classify_tier`** — seniority read from the *leftmost* marker in a title,
  not the highest-weight one. A naive weighted match reads "Summer Intern,
  Director of Product" as senior because "Director" outranks "Intern" by
  weight; position-based matching gets it right, with two guards for the
  inverse case ("Intern Program Director" manages an internship — it is one).
  Extended with the multilingual intern markers (Praktikum, Werkstudent,
  stagiaire, becario, tirocinio) this project's candidate actually needs
  across EU boards.
- **`role_fit`** — stopword-stripped, seniority-gated title matching that
  requires the overlap to be *discriminating* (sharing only "engineer" is not
  a match; every technical title has that word).
- **Legitimacy flags** — a scam/unpaid-posting pattern that multiplies the
  score down rather than averaging it in, so a fraudulent posting that
  happens to keyword-match everything can't rank first.
- **Null-weight redistribution** — a term with no data (no posted date, no
  description) drops out of the weighted average instead of defaulting to
  0.5, which is exactly what collapsed pfzebi's own earlier scorer into a
  narrow, undiscriminating band.

Two real bugs surfaced during the Python port and are now covered by
regression tests: `OpenCV` was mis-casing to `Opencv` through a naive
`str.title()` fallback (silently breaking matches between a declared skill and
an extracted mention), and version-suffixed mentions (`YOLOv8`, `YOLO11`)
weren't canonicalizing to the same skill as a bare `YOLO` mention.

**What this project chose to do differently from pfzebi here:** eligibility
(citizenship/visa/work authorization) is kept as a separate concern from
score, not folded into it as a weighted "evidence" term. This project's own
users had already asked for eligibility as a hard, orthogonal gate (the
`INELIGIBLE` tier, built earlier in the same session) — folding it into the
score would blur two independent questions ("how good a fit" vs "can you
actually apply") this project already treats as separate.

### 2. Funded programs — a hand-curated file, not a search query

Earlier in this session, a user report flagged that Mitacs Globalink Research
Internship never appeared in results. The fix at the time was an extra Serper
query pass — which worked partially, but live testing proved even a query
naming Mitacs directly never returns mitacs.ca itself in Google's top 10, only
third-party mentions (Facebook posts, scholarship-aggregator sites).

pfzebi's `config/programs.yml` solves this differently: a small, hand-written
file with each program's real deadline, not discovered by any scraper or
search query at all, with an explicit comment explaining why — *"a scraper
that silently mis-reads a deadline is worse than no scraper, because you would
trust it."*

`config/programs.yaml` here ports that approach directly (with pfzebi's own
already-researched program data — Mitacs, DAAD RISE, Erasmus+, KAUST VSRP,
INRIA — real facts, not fabricated), converted into ordinary opportunity dicts
via `tools/programs.py` so they flow through the exact same
dedup/eligibility/scoring/report pipeline as every other source, with one
exception: they're exempt from "already reported" exclusion, since a
program's urgency genuinely increases as its deadline approaches, unlike a
job posting.

## What was deliberately not taken

- **The TypeScript rewrite.** Rewriting this project's Telegram bot, CV
  cache, eligibility check, and eight test files in another language would
  have thrown away validated, working infrastructure for a stack switch with
  no clear benefit to a single user's personal tool.
- **The ~90 structured source connectors.** A real coverage advantage, but a
  multi-week porting project on its own, not a redesign decision. Left as a
  known gap, not silently absorbed.
- **The React dashboard.** pfzebi is built for browse-and-triage; this
  project is built for on-demand notification via Telegram. Different use
  case, not a strictly-better/worse comparison.
- **"Visa is a signal, never a veto."** pfzebi's philosophy, and a
  legitimate one — real government sponsor-register data (UK Home Office,
  Dutch IND) used to *raise* a score, never to exclude. This project's user
  explicitly chose the opposite policy for their own use — strict exclusion
  unless sponsorship, a French stage/PFE exception, or a remote/funded-program
  exception is established — because a wasted application costs more than a
  hidden one, for their specific volume of search. Both are defensible; this
  project kept its own deliberate choice rather than importing pfzebi's.

## Before / after

| | Before | After |
|---|---|---|
| Scoring | LLM call, batched (10/batch), retried, scored twice per batch | Pure function, `tools/scoring.py`, one pass, cannot fail |
| Disagreement handling | `DISPUTED` tier when two passes differed (~60% of comparable jobs) | None needed — a pure function has no run-to-run disagreement |
| Parse/truncation failures | `UNRANKED` tier, retry-on-gap, reconciliation by id/URL | Not possible — nothing to parse |
| Cost per run | Up to ~2× batches × ranking calls | Zero LLM calls for scoring; LLM only for ambiguous eligibility cases |
| OpenRouter outage impact | Ranking degrades to `UNRANKED` for affected batches | No impact on ranking at all; only eligibility goes unverified (flagged, not excluded) |
| Funded programs (Mitacs, DAAD, Erasmus+) | Extra search queries, unreliable (verified: official page never in Google's top 10) | Hand-curated file, always found, always has a real deadline |
| Eligibility | Deterministic pre-filter + full LLM judgment on everything that passed it | Same deterministic pre-filter + deterministic exceptions (remote/French stage-PFE/program/explicit sponsorship) resolve most cases without any LLM call; only genuinely ambiguous postings reach the LLM |

## Test coverage added

`tests/test_scoring.py` — 20 tests, fully offline, covering the exact bugs
pfzebi's own history documents (leftmost-marker tier classification, Bayesian
skill-fit smoothing, discriminating role overlap) plus the two bugs this port
introduced and caught before shipping (OpenCV casing, YOLO version-suffix
matching).

`tests/test_ranking_failure.py` and `tests/test_openrouter_credential_expired.py`
were rewritten to prove the new, strictly better failure mode: a completely
broken LLM (or a dead OpenRouter key) no longer produces `UNRANKED` gaps —
every posting still gets a real, deterministic score, with only the narrower
eligibility judgment affected and clearly flagged. A prompt-injection test was
added per this project's own review criteria: a malicious posting cannot
inflate its own score, because there is no model in the scoring path left to
manipulate.
