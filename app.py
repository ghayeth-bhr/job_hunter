"""
EU Job Hunter — Web Frontend (FastAPI)
"""

import asyncio
import io
import json
import os
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import main as pipeline

app = FastAPI(title="EU Job Hunter")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

tasks: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")


@app.post("/run")
async def run_endpoint(
    cv: UploadFile = File(...),
    model: str = Form("openai/gpt-4o-mini"),
    locations: str = Form(""),
    enable_email: str = Form("false"),
):
    task_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{task_id}_{cv.filename}"
    with open(save_path, "wb") as f:
        f.write(await cv.read())

    os.environ["ENABLE_EMAIL"] = enable_email

    pipeline.reconfigure(
        model=model,
        locations=locations if locations.strip() else None,
    )

    log_queue: asyncio.Queue = asyncio.Queue()
    result_holder: dict = {"result": None}

    asyncio.create_task(
        _run_pipeline(task_id, str(save_path), log_queue, result_holder)
    )

    tasks[task_id] = {
        "queue": log_queue,
        "result": result_holder,
    }

    return {"task_id": task_id}


@app.get("/stream/{task_id}")
async def stream(task_id: str):
    info = tasks.get(task_id)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)

    queue = info["queue"]

    async def generate():
        while True:
            msg = await queue.get()
            if msg == "__DONE__":
                result = info["result"]["result"]
                yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"
                break
            if msg == "__ERROR__":
                err = info["result"].get("error", "Unknown error")
                yield f"data: {json.dumps({'type': 'error', 'message': err})}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _run_pipeline(
    task_id: str,
    cv_path: str,
    queue: asyncio.Queue,
    result_holder: dict,
):
    try:

        class QueueWriter(io.StringIO):
            def write(self, s):
                stripped = s.rstrip()
                if stripped:
                    try:
                        queue.put_nowait({"type": "log", "message": stripped})
                    except Exception:
                        pass
                super().write(s)

            def flush(self):
                pass

        writer = QueueWriter()

        with redirect_stdout(writer):
            await pipeline.run_pipeline(cv_path)

        output = writer.getvalue()
        result_holder["result"] = {"output": output}
        await queue.put("__DONE__")
    except Exception as e:
        import traceback

        result_holder["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        await queue.put("__ERROR__")
    finally:
        # Keep result in tasks dict for 60s so SSE can still retrieve it
        # without racing the cleanup. The queue is consumed by the stream,
        # so the only data left is the result_holder dict.
        if task_id in tasks:
            tasks[task_id]["done"] = True

            # Clean up after 60 seconds
            async def _cleanup():
                await asyncio.sleep(60)
                tasks.pop(task_id, None)

            asyncio.create_task(_cleanup())


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./Ali_out"))
if OUTPUT_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(OUTPUT_DIR)), name="reports")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
