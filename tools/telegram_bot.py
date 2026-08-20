"""
On-demand Telegram trigger + delivery for the job-hunter pipeline.

Raw HTTP calls via `requests` + long-polling on getUpdates -- NOT the
python-telegram-bot library. The actual surface area needed is exactly
three Bot API calls (getUpdates, sendMessage, sendDocument) for two
commands and one authorized user; pulling in PTB's async
Application/handler framework for that is meaningfully more machinery
than the problem needs, and inconsistent with how every other source in
this project (tools/job_apis.py) already talks to APIs directly via
`requests`.

Entry point is tools/../run_telegram_bot.py (see that file for how this
actually gets started/kept running).
"""

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path

import requests

_API_TIMEOUT = 40  # long-poll timeout (30s) + margin
_POLL_TIMEOUT = 30


def _api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def send_message(token: str, chat_id: int, text: str) -> None:
    try:
        requests.post(
            _api_url(token, "sendMessage"),
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Telegram sendMessage failed: {e}")


def send_document(token: str, chat_id: int, file_path: str, caption: str = "") -> None:
    path = Path(file_path)
    if not path.exists():
        send_message(token, chat_id, f"⚠️ Report file missing on disk: {file_path}")
        return
    try:
        with open(path, "rb") as f:
            requests.post(
                _api_url(token, "sendDocument"),
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"document": (path.name, f)},
                timeout=60,
            )
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Telegram sendDocument failed: {e}")


class _RunLock:
    """Single-process in-memory lock -- one bot process, so no file lock
    needed. Released in a finally block by the caller so a crash never
    leaves it stuck."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._started_at: float | None = None

    def try_acquire(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._started_at = time.time()
            return True

    def release(self) -> None:
        with self._lock:
            self._running = False
            self._started_at = None

    def status(self) -> tuple[bool, float | None]:
        with self._lock:
            return self._running, self._started_at


_run_lock = _RunLock()


def _count_top_picks(json_report_path: str) -> int:
    try:
        report = json.loads(Path(json_report_path).read_text(encoding="utf-8"))
        return sum(
            1
            for r in report.get("ranked_opportunities", [])
            if r.get("tier") == "TOP PICKS"
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _read_json_meta(json_report_path: str) -> dict:
    try:
        report = json.loads(Path(json_report_path).read_text(encoding="utf-8"))
        return report.get("meta", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _format_sources_unavailable(sources: list[dict]) -> str:
    """e.g. '🚫 2 source(s) unavailable: Adzuna (credential expired),
    OpenRouter (credential expired)' -- matches the exact format asked
    for, not just a generic count buried in the file."""
    if not sources:
        return ""
    parts = ", ".join(
        f"{s.get('source', '?')} ({s.get('kind', '?').lower().replace('_', ' ')})"
        for s in sources
    )
    return f"🚫 {len(sources)} source(s) unavailable: {parts}"


def _send_completion(token: str, chat_id: int, result: dict) -> None:
    total = result.get("total_jobs", 0)
    new = result.get("new_jobs", 0)
    output = result.get("output", "") or ""

    # sources_unavailable can come from two places: the result dict directly
    # (the early-return paths -- Step 1 credential failure, no new jobs --
    # where no report file was ever written), or the JSON report's own
    # meta field (the normal completion path, where write_opportunities_report
    # may ALSO have appended an OpenRouter-during-ranking failure that
    # run_pipeline's own dict never sees, since it only gets a return
    # STRING from that call). The JSON file is the authoritative, complete
    # picture whenever it exists.
    sources_unavailable = list(result.get("sources_unavailable") or [])

    if total == 0:
        text = "✅ Search complete — no new opportunities since the last run."
        sources_line = _format_sources_unavailable(sources_unavailable)
        if sources_line:
            text += f"\n{sources_line}"
        send_message(token, chat_id, text)
        return

    md_match = re.search(r"Markdown\s*:\s*(.+\.md)", output)
    json_match = re.search(r"JSON\s*:\s*(.+\.json)", output)
    top_picks = 0
    if json_match:
        json_path = json_match.group(1).strip()
        top_picks = _count_top_picks(json_path)
        meta = _read_json_meta(json_path)
        # File's list is the complete one when present -- prefer it over
        # whatever the early-return dict carried (which can't know about a
        # failure that happened later, during ranking).
        if meta.get("sources_unavailable"):
            sources_unavailable = meta["sources_unavailable"]

    summary = (
        f"✅ Search complete!\n"
        f"📊 {total} total jobs | {new} new since last run | 🔥 {top_picks} TOP PICKS"
    )
    sources_line = _format_sources_unavailable(sources_unavailable)
    if sources_line:
        summary += f"\n{sources_line}"
    send_message(token, chat_id, summary)

    if md_match:
        send_document(token, chat_id, md_match.group(1).strip(), caption=summary[:1024])
    else:
        send_message(token, chat_id, "⚠️ Could not find the report file path in the pipeline output.")


def _run_pipeline_background(token: str, chat_id: int, cv_path: str) -> None:
    # Force email off regardless of what .env's global ENABLE_EMAIL is set
    # to for the CLI/web-app paths -- Telegram delivery replaces email for
    # this trigger entirely, deliberately, not just "off by today's default".
    os.environ["ENABLE_EMAIL"] = "false"
    import main as pipeline  # deferred import: main.py does real client setup at import time

    try:
        result = asyncio.run(pipeline.run_pipeline(cv_path))
        _send_completion(token, chat_id, result)
    except Exception as e:
        # Loud failure, same principle as everywhere else in this project --
        # the user must never just never hear back.
        send_message(
            token, chat_id,
            f"❌ Search failed: {type(e).__name__}: {e}\n"
            f"This needs a human look, not a silent retry.",
        )
    finally:
        _run_lock.release()


def handle_update(token: str, authorized_chat_id: int, cv_path: str, update: dict) -> None:
    message = update.get("message")
    if not message:
        return
    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()

    if chat_id != authorized_chat_id:
        send_message(token, chat_id, "🚫 Not authorized.")
        return

    if text == "/search":
        if not _run_lock.try_acquire():
            running, started_at = _run_lock.status()
            elapsed = int(time.time() - started_at) if started_at else 0
            send_message(
                token, chat_id,
                f"⏳ A search is already running ({elapsed}s so far) — please wait for it to finish.",
            )
            return
        send_message(
            token, chat_id,
            "🔍 Search started — this may take a while, I'll message you when it's done.",
        )
        thread = threading.Thread(
            target=_run_pipeline_background,
            args=(token, chat_id, cv_path),
            daemon=True,
        )
        thread.start()

    elif text == "/status":
        running, started_at = _run_lock.status()
        if running:
            elapsed = int(time.time() - started_at) if started_at else 0
            send_message(token, chat_id, f"🔄 Search in progress ({elapsed}s so far).")
        else:
            send_message(token, chat_id, "✅ Idle — no search running. Send /search to start one.")


def poll_forever(token: str, authorized_chat_id: int, cv_path: str) -> None:
    print(f"Telegram bot listening (long-polling)... authorized chat_id={authorized_chat_id}")
    offset = None
    while True:
        params = {"timeout": _POLL_TIMEOUT}
        if offset is not None:
            params["offset"] = offset
        try:
            resp = requests.get(
                _api_url(token, "getUpdates"), params=params, timeout=_API_TIMEOUT
            )
            resp.raise_for_status()
            updates = resp.json().get("result", [])
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] getUpdates failed, retrying: {e}")
            time.sleep(5)
            continue

        for update in updates:
            offset = update["update_id"] + 1
            try:
                handle_update(token, authorized_chat_id, cv_path, update)
            except Exception as e:
                print(f"  [WARN] handle_update raised (not fatal to the poll loop): {e}")
