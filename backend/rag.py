from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# In-memory store
_store: List[str] = []
_vectorizer = TfidfVectorizer()
_vectors = None


def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def add_to_store(chunks: List[str]):
    global _store, _vectors
    _store = chunks
    _vectors = _vectorizer.fit_transform(_store)


def clear_store():
    global _store, _vectors
    _store = []
    _vectors = None


def retrieve(query: str, top_k: int = 3) -> List[str]:
    global _vectors

    if not _store or _vectors is None:
        return []

    query_vec = _vectorizer.transform([query])

    # Cosine similarity (TF-IDF vectors are normalized)
    scores = (_vectors @ query_vec.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [_store[i] for i in top_indices]


def answer_with_llm(question: str, chunks: List[str]):
    if not chunks:
        return "No relevant information found.", []

    best_chunk = chunks[0]

    return (
        f"Based on the retrieved context:\n\n{best_chunk}",
        chunks
    )