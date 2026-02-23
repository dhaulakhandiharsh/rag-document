from typing import List
import os
from dotenv import load_dotenv
from google import genai
import math

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_store: List[dict] = []

def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def embed_text(text: str):
    response = client.models.embed_content(
        model="models/embedding-001",
        contents=text
    )
    return response.embedding

def add_to_store(chunks: List[str]) -> None:
    for chunk in chunks:
        embedding = embed_text(chunk)
        _store.append({"text": chunk, "embedding": embedding})

def clear_store():
    global _store
    _store = []

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def retrieve(query: str, top_k: int = 3) -> List[str]:
    if not _store:
        return []

    query_embedding = embed_text(query)

    scored = []
    for item in _store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]

def answer_with_llm(question: str, context_chunks: List[str]):
    context = "\n\n".join(context_chunks)

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=f"""
Answer ONLY using the provided context.
If answer not found, say: Error finding answer.

Context:
{context}

Question:
{question}
"""
        )
        return response.text, context
    except Exception:
        return "LLM unavailable. Showing retrieved context.", context