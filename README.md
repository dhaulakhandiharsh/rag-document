# RAG Document Question Answering System

A full stack Retrieval Augmented Generation application that allows users to upload a document and ask contextual questions about it.

The system retrieves relevant document chunks using semantic embeddings and generates grounded responses using a language model. If the LLM is unavailable due to quota or API issues, the system gracefully falls back to document based retrieval so the application continues working.

---

## What This Project Does

This application implements a complete RAG pipeline:

1. User uploads a document  
2. Backend extracts and chunks the text  
3. Each chunk is converted into a semantic embedding  
4. Embeddings are stored in memory  
5. User question is embedded  
6. Top relevant chunks are retrieved using cosine similarity  
7. Context is passed to the language model  
8. Answer is returned to the frontend  

The response is grounded strictly in the uploaded document.

---

## Tech Stack

### Frontend
React  
Vite  
Axios  

### Backend
FastAPI  
Uvicorn  
Sentence Transformers  
NumPy  
Python dotenv  

### Language Model
Google Gemini API  

### Retrieval
In memory vector store  
Cosine similarity search  

---

## How The RAG Pipeline Works

### Document Upload
The user uploads a PDF or pastes text.  
The backend extracts raw text from the file.  
The text is split into manageable chunks.

### Embedding Generation
Each chunk is converted into a dense vector using a sentence transformer model.  
These embeddings are stored in memory for similarity search.

### Question Processing
The user submits a question.  
The question is embedded using the same embedding model.

### Retrieval
Cosine similarity is calculated between the question embedding and stored document embeddings.  
Top relevant chunks are selected.

### Answer Generation
The retrieved chunks are provided as context to Gemini.  
The LLM generates a response strictly based on the provided context.

If the Gemini API is unavailable due to quota or key issues, the system falls back to returning the most relevant document content instead of failing.

---

## Why This Project Matters

This project demonstrates:

- Understanding of Retrieval Augmented Generation  
- Semantic search using embeddings  
- Vector similarity scoring  
- Full stack integration between React and FastAPI  
- Graceful API failure handling  
- Clean modular backend design  

---

## Running Locally

### Clone Repository

```bash
git clone git@github.com:dhaulakhandiharsh/rag-document.git
cd rag-document


---

## Running Locally

### Clone Repository

```bash
git clone git@github.com:dhaulakhandiharsh/rag-document.git
cd rag-document
```

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file inside backend:

```
GEMINI_API_KEY=your_api_key_here
```

Run backend:

```bash
uvicorn main:app --reload --port 8002
```

Backend runs at:
http://localhost:8002

Swagger documentation:
http://localhost:8002/docs

---

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
http://localhost:5173

