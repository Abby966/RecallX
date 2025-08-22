# RAG Chatbot Backend (FastAPI, local embeddings)

## Features
- Upload PDF or .txt
- Extract text (pypdf for PDFs), chunk with overlap
- Local embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Cosine similarity search (pure NumPy, no FAISS required)
- Simple in-memory+on-disk store (JSON + .npy) — portable and easy

## Run (Windows/Mac/Linux)
```bash
cd backend
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Mac/Linux
# source .venv/bin/activate

pip install -r requirements.txt
# (First run downloads the embedding model ~90MB)

# Start API
uvicorn app:app --reload --port 8000
```

API will be at: http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs
```
