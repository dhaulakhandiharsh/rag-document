from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from rag import (
    chunk_text,
    add_to_store,
    retrieve,
    answer_with_llm,
    clear_store,
)

app = FastAPI(title="RAG API")

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "RAG API is running.",
        "docs": "/docs",
    }


@app.post("/upload")
async def upload(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    content = None

    # --- Handle file upload ---
    if file and file.filename:
        body = await file.read()
        filename_lower = file.filename.lower()
        is_pdf = (
            file.content_type == "application/pdf"
            or filename_lower.endswith(".pdf")
        )

        if is_pdf:
            try:
                reader = PdfReader(io.BytesIO(body))
                pages = [
                    (page.extract_text() or "")
                    for page in reader.pages
                ]
                content = "\n\n".join(pages).strip()
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read PDF file.",
                )

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="PDF has no extractable text.",
                )

        else:
            try:
                content = body.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="File must be UTF-8 text.",
                )

    # --- Handle pasted text ---
    if text and text.strip():
        content = (
            text.strip()
            if not content
            else content + "\n\n" + text.strip()
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file or pasted text.",
        )

    # Clear previous store (single-document design)
    clear_store()

    chunks = chunk_text(content)
    add_to_store(chunks)

    return {
        "status": "ok",
        "chunks_added": len(chunks),
    }


@app.post("/query")
async def query(question: str = Form(...)):
    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question is required.",
        )

    chunks = retrieve(question.strip(), top_k=3)

    answer, sources = answer_with_llm(
        question.strip(),
        chunks,
    )

    return {
        "answer": answer,
        "sources": sources,
    }


@app.post("/clear")
async def clear():
    clear_store()
    return {"status": "ok"}