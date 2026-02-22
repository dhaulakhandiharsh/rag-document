"""
FastAPI app: /upload and /query for the RAG demo.
"""
from __future__ import annotations

import os
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from dotenv import load_dotenv

from rag import chunk_text, add_to_store, retrieve, answer_with_llm, clear_store

load_dotenv()  # load variables from backend/.env if present

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177", "http://localhost:5178", "http://127.0.0.1:5173", "http://127.0.0.1:5178"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "RAG API is running. Use the frontend at http://localhost:5173 (or 5176 if that port is in use).", "docs": "http://localhost:8002/docs"}


@app.post("/upload")
async def upload(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    content = None
    if file and file.filename:
        body = await file.read()
        filename_lower = file.filename.lower()
        is_pdf = file.content_type == "application/pdf" or filename_lower.endswith(".pdf")
        if is_pdf:
            try:
                reader = PdfReader(io.BytesIO(body))
                pages = [(page.extract_text() or "") for page in reader.pages]
                content = "\n\n".join(pages).strip()
            except Exception:
                raise HTTPException(status_code=400, detail="Could not read PDF file.")
            if not content:
                raise HTTPException(status_code=400, detail="PDF seems to have no extractable text.")
        else:
            try:
                content = body.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File must be UTF-8 text.")
    if text and text.strip():
        content = text.strip() if not content else content + "\n\n" + text.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Provide either a file or pasted text.")

    chunks = chunk_text(content)
    add_to_store(chunks)
    return {"status": "ok", "chunks_added": len(chunks)}


@app.post("/query")
async def query(question: str = Form(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    use_openai = os.environ.get("OPENAI_API_KEY") is not None
    chunks = retrieve(question.strip(), top_k=3)
    answer, full_context = answer_with_llm(question.strip(), chunks, use_openai=use_openai)

    return {
        "answer": answer,
        "sources": chunks,
    }


@app.post("/clear")
async def clear():
    clear_store()
    return {"status": "ok"}
