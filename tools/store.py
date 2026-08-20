"""
Cross-run persistence for the daily job hunt.

Without this, every run rediscovers and re-reports the same postings —
which defeats the point of a *daily* digest. This module tracks which
postings have already been surfaced (by normalized URL) so a run only
reports what's genuinely NEW since the last one.

Also owns URL normalization (strip tracking params, trailing slash,
lowercase) so both this module and main.py's dedup logic agree on what
counts as "the same job".
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

DB_PATH = Path(os.getenv("SEEN_JOBS_DB", "./data/seen_jobs.db"))

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


def normalize_url(url: str) -> str:
    """Strip tracking params and trailing slash for dedup/seen-tracking."""
    if not url:
        return ""
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


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_jobs (
            url TEXT PRIMARY KEY,
            title TEXT,
            first_seen TEXT,
            last_score INTEGER
        )
        """
    )
    conn.commit()
    return conn


def filter_new(jobs: list[dict], conn: sqlite3.Connection | None = None) -> list[dict]:
    """Return only the jobs whose normalized URL isn't already in the DB."""
    own_conn = conn is None
    if own_conn:
        conn = init_db()
    try:
        cur = conn.cursor()
        new_jobs = []
        for job in jobs:
            raw_url = job.get("source_url") or job.get("url") or job.get("link", "")
            key = normalize_url(raw_url)
            if not key:
                new_jobs.append(job)
                continue
            row = cur.execute(
                "SELECT 1 FROM seen_jobs WHERE url = ?", (key,)
            ).fetchone()
            if row is None:
                new_jobs.append(job)
        return new_jobs
    finally:
        if own_conn:
            conn.close()


def mark_seen(jobs: list[dict], conn: sqlite3.Connection | None = None) -> None:
    """Record jobs as seen so future runs won't re-report them."""
    own_conn = conn is None
    if own_conn:
        conn = init_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        for job in jobs:
            raw_url = job.get("source_url") or job.get("url") or job.get("link", "")
            key = normalize_url(raw_url)
            if not key:
                continue
            conn.execute(
                """
                INSERT OR IGNORE INTO seen_jobs (url, title, first_seen, last_score)
                VALUES (?, ?, ?, ?)
                """,
                (key, job.get("title", ""), now, job.get("score", 0) or 0),
            )
        conn.commit()
    finally:
        if own_conn:
            conn.close()
