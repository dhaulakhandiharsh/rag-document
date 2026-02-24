🚀 RAG Document QA System

A full stack Retrieval Based Question Answering application that allows users to upload a document and ask contextual questions about it.

The system retrieves relevant document chunks using vector similarity and returns answers strictly grounded in the uploaded content.

🌐 Frontend deployed on Vercel
⚙️ Backend deployed on Render

⸻

📌 What This Project Does

This application implements a simplified RAG style pipeline:

1️⃣ User uploads a document (Text or PDF)
2️⃣ Backend extracts and chunks the text
3️⃣ Each chunk is converted into a TF IDF vector
4️⃣ Vectors are stored in memory
5️⃣ User submits a question
6️⃣ Question is vectorized using the same model
7️⃣ Cosine similarity is calculated
8️⃣ Most relevant chunk is returned as the answer

✅ The response is strictly based on the uploaded document
✅ No hallucinations
✅ No external LLM dependency

⸻

🛠 Tech Stack

🎨 Frontend

React
Vite
Fetch API

⸻

⚙️ Backend

FastAPI
Uvicorn
pypdf
scikit learn
NumPy
Python 3

⸻

🔎 Retrieval

In memory vector store
Cosine similarity search

⸻

☁️ Deployment

Vercel for frontend
Render for backend

⸻

🧠 How The Retrieval Works

📄 Document is split into chunks of approximately 300 words

📊 Each chunk is converted into a TF IDF vector representation

❓ When a question is asked:

• The question is converted into a vector
• Cosine similarity is computed between the question vector and stored chunk vectors
• The top matching chunk is returned

🎯 This ensures the answer always comes directly from the uploaded document.

⸻

💻 Running Locally
Clone the repository:
git clone git@github.com:dhaulakhandiharsh/rag-document.git
cd rag-document

⚙️ Backend Setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8003
Backend runs at
http://localhost:8003

Swagger documentation
http://localhost:8003/docs

⸻

🎨 Frontend Setup

In a new terminal:
cd frontend
npm install
npm run dev
Frontend runs at
http://localhost:5173

⸻

🧪 Usage

1️⃣ Open the frontend in your browser
2️⃣ Upload a text or PDF document
3️⃣ Ask a question related to the document
4️⃣ The system retrieves the most relevant chunk and displays it along with the source text used

⸻

⭐ Key Highlights

✔ Full stack architecture using React and FastAPI
✔ Custom vector similarity implementation
✔ Clean modular backend design
✔ Production deployment on Vercel and Render
✔ Grounded answers without relying on external LLM APIs

⸻

🎯 What This Demonstrates

• Understanding of Retrieval Augmented Generation concepts
• Practical implementation of vector search
• Backend API design with FastAPI
• Frontend backend integration
• Real world deployment experience

