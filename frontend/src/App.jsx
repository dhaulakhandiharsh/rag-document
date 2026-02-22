import { useState, useEffect } from 'react'

// Use proxy so we don't hit CORS – see vite.config.js (backend must match target port)
const API = '/api'

export default function App() {
  const [pastedText, setPastedText] = useState('')
  const [file, setFile] = useState(null)
  const [uploadStatus, setUploadStatus] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [backendOk, setBackendOk] = useState(null)

  useEffect(() => {
    fetch(`${API}/`)
      .then((r) => r.json())
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false))
  }, [])

  async function handleUpload(e) {
    e.preventDefault()
    setError('')
    setUploadStatus('')
    const form = new FormData()
    if (file) form.append('file', file)
    if (pastedText.trim()) form.append('text', pastedText.trim())
    if (!file && !pastedText.trim()) {
      setError('Add a file or paste some text.')
      return
    }
    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setUploadStatus(`Uploaded. ${data.chunks_added} chunks added.`)
      setFile(null)
      setPastedText('')
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleQuery(e) {
    e.preventDefault()
    setError('')
    setAnswer('')
    setSources([])
    if (!question.trim()) return
    setLoading(true)
    try {
      const form = new FormData()
      form.append('question', question.trim())
      const res = await fetch(`${API}/query`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Query failed')
      setAnswer(data.answer)
      setSources(data.sources || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <h1>RAG Demo – Ask questions about your document</h1>
      {backendOk === true && <p className="backend-status ok">Backend connected.</p>}
      {backendOk === false && (
        <p className="backend-status fail">
          Cannot reach backend. Start it: in <code>backend/</code> run <code>source venv/bin/activate && uvicorn main:app --reload --port 8003</code>
        </p>
      )}

      <section>
        <h2>1. Upload or paste text</h2>
        <form onSubmit={handleUpload}>
          <label>Paste text here</label>
          <textarea
            value={pastedText}
            onChange={(e) => setPastedText(e.target.value)}
            rows={4}
            placeholder="Or upload a .txt file below"
          />
          <label>Or choose a text/PDF file</label>
          <input
            type="file"
            accept=".txt,.pdf,text/plain,application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button type="submit">Upload</button>
        </form>
        {uploadStatus && <p>{uploadStatus}</p>}
      </section>

      <section>
        <h2>2. Ask a question</h2>
        <form onSubmit={handleQuery}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What is the main idea?"
            disabled={loading}
          />
          <button type="submit" disabled={loading}>{loading ? '...' : 'Ask'}</button>
        </form>
      </section>

      {error && <p className="error">{error}</p>}

      {answer && (
        <section>
          <h2>Answer</h2>
          <div id="answer">{answer}</div>
        </section>
      )}

      {sources.length > 0 && (
        <section>
          <h2>Text used (sources)</h2>
          <div id="sources">
            {sources.map((src, i) => (
              <div key={i} className="source-block">
                <strong>Chunk {i + 1}:</strong><br />
                {src}
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  )
}
