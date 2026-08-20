"""
On-demand Telegram bot entry point for the job-hunter pipeline.

Run manually:
    python run_telegram_bot.py

Keeps polling until you Ctrl+C. For always-on availability without a
terminal open, run this via Windows Task Scheduler (trigger: "At log on",
action: pythonw.exe run_telegram_bot.py so no console window appears).
No scheduled cron for the pipeline itself -- it only runs when you send
/search in Telegram.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from tools.telegram_bot import poll_forever

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_CV_PATH = os.getenv("TELEGRAM_CV_PATH")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be set in .env")
    sys.exit(1)

if not TELEGRAM_CV_PATH or not Path(TELEGRAM_CV_PATH).exists():
    print(
        f"TELEGRAM_CV_PATH is not set or the file doesn't exist "
        f"(got: {TELEGRAM_CV_PATH!r}). Set it in .env to the CV file "
        f"this bot should run searches against."
    )
    sys.exit(1)

if __name__ == "__main__":
    poll_forever(TELEGRAM_BOT_TOKEN, int(TELEGRAM_CHAT_ID), TELEGRAM_CV_PATH)
