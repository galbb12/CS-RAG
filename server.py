"""
FastAPI server exposing the RAG system as an OpenAI-compatible API.

Run with:  uvicorn server:app --port 8000
"""

import json
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from rag_engine import RAGEngine


engine: RAGEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    print("Loading RAG engine …")
    engine = RAGEngine()
    print("RAG engine ready.")
    yield


app = FastAPI(title="HAIFA-RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "haifa-rag"

#Request struct
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False

# Create 
def _make_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# Fast api endpoints
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "haifa-rag",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream_response(messages),
            media_type="text/event-stream",
        )

    # Non-streaming
    try:
        rag_result = await engine.aquery(messages)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": _make_chat_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": rag_result.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "tool_outputs": rag_result.tool_outputs,
        "suggestions": rag_result.suggestions,
    }


async def _stream_response(messages: list[dict]):
    """SSE generator - runs the full pipeline then streams the result in chunks."""
    chat_id = _make_chat_id()
    created = int(time.time())

    try:
        rag_result = await engine.aquery(messages)
    except Exception as e:
        traceback.print_exc()
        error_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n[שגיאה: {e}]"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Stream the answer in chunks
    chunk_size = 8
    for i in range(0, len(rag_result.content), chunk_size):
        chunk = rag_result.content[i : i + chunk_size]
        data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    # tool_outputs + suggestions are sent as a metadata chunk
    if rag_result.tool_outputs or rag_result.suggestions:
        meta = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "tool_outputs": rag_result.tool_outputs,
            "suggestions": rag_result.suggestions,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    # Final chunk
    final = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}

# Serve static files
DIST_DIR = Path(__file__).parent / "frontend" / "dist"

if DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve the React SPA for any non-API route."""
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
