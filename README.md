# 📄 RAG Document Question Answering System

A fullstack Retrieval Augmented Generation (RAG) application that allows users to upload a document and ask contextual questions about it.

The system retrieves relevant document chunks using vector embeddings and generates grounded responses using a language model.

---

## 🚀 Overview

This project implements a basic RAG pipeline using:

- **Frontend:** React + Vite
- **Backend:** FastAPI (Python)
- **Embeddings:** OpenAI Embedding Model
- **LLM:** OpenAI Chat Model
- **Vector Storage:** In memory list
- **Similarity Metric:** Cosine similarity

The goal is to answer user questions based only on the uploaded document context.

---

## 🏗 Architecture

```
User → React Frontend → FastAPI Backend
        ↓
    Upload Document
        ↓
Text Extraction → Chunking → Embedding Generation
        ↓
Store Embeddings (In-Memory)
        ↓
User Question → Query Embedding
        ↓
Cosine Similarity Search
        ↓
Top-K Relevant Chunks
        ↓
LLM Response Generation
        ↓
Return Answer to Frontend
```

---

## 🧠 How the RAG Flow Works

This application follows the Retrieval Augmented Generation approach:

### 1️⃣ Document Upload
- The user uploads a document.
- The backend extracts raw text.
- The text is split into smaller overlapping chunks.

### 2️⃣ Embedding Generation
- Each chunk is converted into a vector embedding using an OpenAI embedding model.
- These embeddings are stored in memory.

### 3️⃣ User Question
- The user submits a question.
- The question is also converted into an embedding.

### 4️⃣ Retrieval
- Cosine similarity is calculated between the question embedding and all stored document embeddings.
- The top k most relevant chunks are selected.

### 5️⃣ Generation
- The retrieved chunks are provided as context to the LLM.
- The LLM generates an answer grounded in the document content.

This ensures the model answers based on relevant document context rather than general knowledge.

---

## ⚙ Backend (FastAPI)

Key responsibilities:

- File upload handling
- Text chunking
- Embedding generation
- Cosine similarity computation
- LLM response generation

FastAPI was chosen because:
- Strong typing with Pydantic
- Built in validation
- Automatic Swagger documentation
- Strong ecosystem support for AI tooling

---

## 💻 Frontend (React + Vite)

The frontend provides:

- Document upload interface
- Question input field
- Answer display section
- API communication with backend

Vite was used for fast development and hot module reloading.

---

## 🛠 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone git@github.com:dhaulakhandiharsh/rag-document.git
cd rag-document
```

---

### 2️⃣ Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file inside `backend/`:

```
OPENAI_API_KEY=your_api_key_here
```

Run backend:

```bash
uvicorn main:app --reload
```

Backend runs on:
```
http://localhost:8000
```

---

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:
```
http://localhost:5173
```

---

## 📌 Assumptions & Shortcuts

- Embeddings are stored in memory (not persistent).
- No database integration.
- No authentication layer.
- Basic text extraction (no advanced parsing).
- Not optimized for large scale production workloads.

This implementation focuses on demonstrating core RAG concepts clearly.

---

## 🔮 Future Improvements

- Integrate FAISS or Pinecone for scalable vector storage
- Add document persistence
- Support multiple documents
- Implement streaming responses
- Add authentication
- Improve chunking strategy (dynamic chunk sizing)

---

## 🎯 Key Learnings

- Implemented end to end RAG pipeline
- Learned FastAPI while transitioning from Node.js backend experience
- Gained hands on experience with embeddings and retrieval based AI systems
- Understood trade offs between chunk size and retrieval accuracy

---

## 📜 License

This project is built for educational and evaluation purposes.
