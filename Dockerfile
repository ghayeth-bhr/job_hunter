FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright
ENV CRAWL4AI_CACHE_DIR=/app/crawl4ai_cache
ENV OUTPUT_DIR=/app/reports

RUN apt-get update && apt-get install -y \
    wget gnupg libglib2.0-0 libnss3 libnspr4 libatk1.0-0t64 \
    libatk-bridge2.0-0t64 libcups2t64 libdrm2 libdbus-1-3 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    libasound2t64 libatspi2.0-0t64 libwayland-client0 \
    libwayland-egl1 libwayland-cursor0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    openai>=1.0 \
    openai-agents \
    PyPDF2 \
    python-docx \
    requests \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    "crawl4ai>=0.9.0,<1.0"

RUN python -m playwright install chromium

COPY . .

RUN mkdir -p /app/reports /app/uploads /app/crawl4ai_cache

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
