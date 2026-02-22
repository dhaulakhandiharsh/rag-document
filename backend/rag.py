"""
Simple RAG: chunk text, embed, store in memory, retrieve and answer.
Keeping it minimal - one file for the logic.
"""
from __future__ import annotations

import re
import os
from typing import List, Optional

# we'll init the model lazily so server starts even if st is slow to load
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # small model, runs on CPU fine
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split into overlapping chunks so we don't cut sentences in half."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if sum(len(s) for s in current) + len(p) > chunk_size and current:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            current = []
        if len(p) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = []
            for s in sentences:
                if sum(len(x) for x in buf) + len(s) > chunk_size and buf:
                    chunks.append(" ".join(buf))
                    buf = []
                buf.append(s)
            if buf:
                current = buf
            continue
        current.append(p)

    if current:
        chunks.append("\n\n".join(current))
    return chunks if chunks else [text] if text.strip() else []


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    model = get_embedder()
    return model.encode(chunks).tolist()


_store: List[dict] = []


def add_to_store(chunks: List[str]) -> None:
    if not chunks:
        return
    embs = embed_chunks(chunks)
    for t, e in zip(chunks, embs):
        _store.append({"text": t, "embedding": e})


def clear_store() -> None:
    global _store
    _store = []


def cosine_sim(a: List[float], b: List[float]) -> float:
    import numpy as np
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-9))


def retrieve(query: str, top_k: int = 3) -> List[str]:
    if not _store:
        return []
    model = get_embedder()
    q_emb = model.encode([query]).tolist()[0]
    scored = [(cosine_sim(q_emb, d["embedding"]), d["text"]) for d in _store]
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]


def answer_with_llm(question: str, context_chunks: List[str], use_openai: bool = False) -> tuple[str, str]:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(No relevant text found.)"

    if use_openai:
        try:
            from openai import OpenAI
            client = OpenAI() 

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer only using the provided context. If the context doesn't contain the answer, say Error finding answer."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )

            msg = resp.choices[0].message.content
            return msg, context

        except Exception as e:
            return f"OpenAI error: {e}", context

    mock_answer = (
        f"Based on the uploaded document ({len(context_chunks)} relevant chunk(s)):\n\n"
        f"The relevant text says:\n\n{context[:800]}{'...' if len(context) > 800 else ''}\n\n"
       
    )
    return mock_answer, context