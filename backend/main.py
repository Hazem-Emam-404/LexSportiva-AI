import os
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

from utils import map_sport_to_filename, DEFAULT_FILES_DIR
from rag_pipeline import get_pipeline
import database as db

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    get_pipeline()
    yield


app = FastAPI(title="LexSportiva-AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-ID", "*"],
)


def resolve_user_id(request: Request, response: Response) -> str:
    user_id = request.headers.get("X-User-ID") or request.cookies.get("sports_user_id")
    if not user_id:
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        response.set_cookie(
            key="sports_user_id",
            value=user_id,
            max_age=365 * 24 * 3600,
            httponly=False,
            samesite="none",
            secure=True,
        )
    response.headers["X-User-ID"] = user_id
    return user_id



class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class CreateConversationRequest(BaseModel):
    title: Optional[str] = "New Chat"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/conversations")
def get_conversations(request: Request, response: Response):
    user_id = resolve_user_id(request, response)
    return db.list_conversations(user_id)


@app.post("/api/conversations")
def create_conversation(data: CreateConversationRequest, request: Request, response: Response):
    user_id = resolve_user_id(request, response)
    title = data.title.strip() if data.title else "New Chat"
    return db.create_conversation(user_id, title)


@app.get("/api/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str, request: Request, response: Response):
    user_id = resolve_user_id(request, response)
    messages = db.get_conversation_messages(conversation_id, user_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request, response: Response):
    user_id = resolve_user_id(request, response)
    success = db.delete_conversation(conversation_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}


@app.post("/api/chat")
async def chat(data: ChatRequest, request: Request, response: Response):
    user_id = resolve_user_id(request, response)
    message = data.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conv_id = data.conversation_id
    is_new = False
    if not conv_id:
        conv = db.create_conversation(user_id, title=message[:40])
        conv_id = conv["id"]
        is_new = True

    history_records = db.get_conversation_messages(conv_id, user_id)
    if history_records is None:
        conv = db.create_conversation(user_id, title=message[:40])
        conv_id = conv["id"]
        history_records = []


    chat_history = [{"role": m["role"], "content": m["content"]} for m in history_records]

    db.add_message(conv_id, "user", message)

    pipeline = get_pipeline()
    rag_res = await pipeline.query(message, chat_history=chat_history, return_sources=True)

    answer_text = rag_res["answer"] if isinstance(rag_res, dict) else str(rag_res)

    citations = []
    claims = rag_res.get("claims", []) if isinstance(rag_res, dict) else []
    for c in claims:
        fname = c.source.file_name
        pdf_file = map_sport_to_filename(fname) or f"{fname}.pdf"
        page = str(c.source.page_number)
        citations.append({
            "claim": c.claim,
            "sport": fname,
            "file_name": pdf_file,
            "page_number": page,
            "pdf_url": f"/api/pdf/{pdf_file}#page={page}",
        })

    db.add_message(conv_id, "assistant", answer_text, citations)

    if not is_new and len(history_records) == 0:
        db.update_conversation_title(conv_id, message[:40])

    return {
        "conversation_id": conv_id,
        "answer": answer_text,
        "citations": citations,
    }


@app.get("/api/pdf/{filename}")
def get_pdf(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(DEFAULT_FILES_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"PDF '{safe_name}' not found")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={safe_name}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
