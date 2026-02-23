from __future__ import annotations

import re
import os
from typing import List
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
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
    model_local = get_embedder()
    return model_local.encode(chunks).tolist()


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
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-9))


def retrieve(query: str, top_k: int = 3) -> List[str]:
    if not _store:
        return []

    model_local = get_embedder()
    q_emb = model_local.encode([query]).tolist()[0]

    scored = [(cosine_sim(q_emb, d["embedding"]), d["text"]) for d in _store]
    scored.sort(key=lambda x: -x[0])

    return [text for _, text in scored[:top_k]]

def answer_with_llm(question: str, context_chunks: List[str]) -> tuple[str, str]:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(No relevant text found.)"

    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
If the context does not contain the answer, say: "Error finding answer."

Context:
{context}

Question:
{question}
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text, context

    except Exception:
        # Fallback if Gemini fails
        if context_chunks:
            fallback_answer = (
                "LLM unavailable. Based on the document:\n\n"
                + context[:800]
            )
            return fallback_answer, context
        else:
            return "No relevant information found.", context