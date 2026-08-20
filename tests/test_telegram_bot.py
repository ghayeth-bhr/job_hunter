"""
Tests for tools/telegram_bot.py:
  1. Authorization rejection -- an unauthorized chat_id must NEVER
     trigger the pipeline.
  2. Run-in-progress lock -- a second /search while one is active must
     not spawn a second pipeline run.
  3. Mocked end-to-end trigger -- an authorized /search results in the
     report file being "sent" (mocked Telegram calls) with correct
     headline numbers extracted from a canned run_pipeline result.

All Telegram network calls and the pipeline itself are mocked -- no real
bot token/network use, no real (costly) pipeline run.

Run directly: python tests/test_telegram_bot.py
"""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.telegram_bot as bot

AUTHORIZED_CHAT_ID = 111111
UNAUTHORIZED_CHAT_ID = 222222
FAKE_TOKEN = "fake-token"


def _reset_lock():
    # Tests run in one process and share bot._run_lock -- reset it between
    # tests so one test's lock state can't leak into the next.
    bot._run_lock.release()


# ── 1. Authorization rejection ────────────────────────────────────────────


def test_unauthorized_chat_id_never_triggers_pipeline():
    _reset_lock()
    update = {
        "update_id": 1,
        "message": {"chat": {"id": UNAUTHORIZED_CHAT_ID}, "text": "/search"},
    }
    with patch.object(bot, "send_message") as mock_send, \
         patch("threading.Thread") as mock_thread:
        bot.handle_update(FAKE_TOKEN, AUTHORIZED_CHAT_ID, "/fake/cv.pdf", update)

        mock_thread.assert_not_called()
        assert bot._run_lock.status()[0] is False, "lock must never be acquired for an unauthorized sender"
        mock_send.assert_called_once()
        sent_chat_id, sent_text = mock_send.call_args[0][1], mock_send.call_args[0][2]
        assert sent_chat_id == UNAUTHORIZED_CHAT_ID
        assert "not authorized" in sent_text.lower()
    print("PASS: unauthorized chat_id never spawns a thread, never acquires the lock")


def test_authorized_chat_id_does_trigger():
    _reset_lock()
    update = {
        "update_id": 2,
        "message": {"chat": {"id": AUTHORIZED_CHAT_ID}, "text": "/search"},
    }
    with patch.object(bot, "send_message") as mock_send, \
         patch("threading.Thread") as mock_thread_cls:
        mock_thread_instance = MagicMock()
        mock_thread_cls.return_value = mock_thread_instance

        bot.handle_update(FAKE_TOKEN, AUTHORIZED_CHAT_ID, "/fake/cv.pdf", update)

        mock_thread_cls.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        assert bot._run_lock.status()[0] is True, "lock should be held once an authorized /search starts"
    _reset_lock()
    print("PASS: authorized chat_id does spawn a background thread")


# ── 2. Run-in-progress lock ────────────────────────────────────────────────


def test_second_search_while_running_does_not_spawn_another_thread():
    _reset_lock()
    assert bot._run_lock.try_acquire() is True  # simulate a run already in progress

    update = {
        "update_id": 3,
        "message": {"chat": {"id": AUTHORIZED_CHAT_ID}, "text": "/search"},
    }
    with patch.object(bot, "send_message") as mock_send, \
         patch("threading.Thread") as mock_thread_cls:
        bot.handle_update(FAKE_TOKEN, AUTHORIZED_CHAT_ID, "/fake/cv.pdf", update)

        mock_thread_cls.assert_not_called()
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "already running" in sent_text.lower()
    _reset_lock()
    print("PASS: a second /search while one is running does not spawn another thread")


def test_lock_released_after_pipeline_failure():
    # A crash in the pipeline must not leave the lock stuck forever.
    _reset_lock()

    class FakePipeline:
        async def run_pipeline(self, cv_path):
            raise RuntimeError("boom")

    with patch.object(bot, "send_message") as mock_send, \
         patch.dict(sys.modules, {"main": FakePipeline()}):
        bot._run_pipeline_background(FAKE_TOKEN, AUTHORIZED_CHAT_ID, "/fake/cv.pdf")

    assert bot._run_lock.status()[0] is False, "lock must be released even after an uncaught exception"
    failure_texts = [c.args[2] for c in mock_send.call_args_list]
    assert any("failed" in t.lower() for t in failure_texts), "must send a loud failure message"
    print("PASS: lock is released after a pipeline crash, and a failure message is sent")


# ── 3. Mocked end-to-end trigger ──────────────────────────────────────────


def test_completion_message_and_document_have_correct_numbers(tmp_path=None):
    _reset_lock()
    tmp_dir = Path(sys.modules[__name__].__file__).resolve().parent / "_tmp_telegram_test"
    tmp_dir.mkdir(exist_ok=True)
    md_path = tmp_dir / "opportunities_test.md"
    json_path = tmp_dir / "opportunities_test.json"
    md_path.write_text("# Fake report", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "ranked_opportunities": [
                    {"tier": "TOP PICKS"},
                    {"tier": "TOP PICKS"},
                    {"tier": "GOOD FITS"},
                ]
            }
        ),
        encoding="utf-8",
    )

    fake_result = {
        "output": f"Reports written!\n  📄 Markdown : {md_path}\n  📦 JSON     : {json_path}\n",
        "total_jobs": 12,
        "new_jobs": 5,
        "email": {},
    }

    with patch.object(bot, "send_message") as mock_send_msg, \
         patch.object(bot, "send_document") as mock_send_doc:
        bot._send_completion(FAKE_TOKEN, AUTHORIZED_CHAT_ID, fake_result)

        mock_send_msg.assert_called_once()
        summary_text = mock_send_msg.call_args[0][2]
        assert "12 total jobs" in summary_text
        assert "5 new" in summary_text
        assert "2 " in summary_text and "TOP PICKS" in summary_text

        mock_send_doc.assert_called_once()
        sent_path = mock_send_doc.call_args[0][2]
        assert sent_path == str(md_path)

    md_path.unlink()
    json_path.unlink()
    tmp_dir.rmdir()
    print(f"PASS: completion message correct ({summary_text!r}) and document path correct")


if __name__ == "__main__":
    test_unauthorized_chat_id_never_triggers_pipeline()
    test_authorized_chat_id_does_trigger()
    test_second_search_while_running_does_not_spawn_another_thread()
    test_lock_released_after_pipeline_failure()
    test_completion_message_and_document_have_correct_numbers()
    print("\nALL PASS")
