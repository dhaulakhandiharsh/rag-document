from typing import List
import math
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_store: List[dict] = []

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def embed(text: str):
    return embedder.encode(text).tolist()

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    return dot / (norm_a*norm_b + 1e-9)

def add_to_store(chunks: List[str]):
    for chunk in chunks:
        _store.append({
            "text": chunk,
            "embedding": embed(chunk)
        })

def clear_store():
    global _store
    _store = []

def retrieve(query: str, top_k: int = 3):
    if not _store:
        return []
    q_emb = embed(query)
    scored = [(cosine(q_emb, item["embedding"]), item["text"]) for item in _store]
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

def answer_with_llm(question: str, chunks: List[str]):
    context = "\n\n".join(chunks)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
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