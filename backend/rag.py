from typing import List
import os
import math
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
import numpy as np

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_store: List[str] = []
_vectorizer = TfidfVectorizer()
_vectors = None

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def add_to_store(chunks: List[str]):
    global _store, _vectors
    _store = chunks
    _vectors = _vectorizer.fit_transform(_store)

def clear_store():
    global _store, _vectors
    _store = []
    _vectors = None

def retrieve(query: str, top_k: int = 3):
    global _vectors
    if not _store or _vectors is None:
        return []

    query_vec = _vectorizer.transform([query])
    scores = (_vectors @ query_vec.T).toarray().flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_store[i] for i in top_indices]

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