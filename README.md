# Employee Assistant — Internal Knowledge Chat

A RAG (Retrieval-Augmented Generation) chat application that lets employees query internal company documents in natural language. Built with FastAPI, ChromaDB, and the OpenAI API.

## What this project demonstrates

- End-to-end RAG pipeline design: document ingestion, vector embedding, semantic retrieval, and grounded generation
- REST API development with FastAPI
- Integration with OpenAI's embedding and chat completion APIs
- Local vector database management with ChromaDB
- Lightweight frontend wired to a live backend

## How it works

1. Internal documents are chunked and embedded using OpenAI's `text-embedding-3-small`
2. Embeddings are stored in a local ChromaDB vector database
3. When a user asks a question, it is embedded and the most semantically relevant chunks are retrieved
4. Retrieved chunks are injected as context into a GPT-4o prompt
5. The grounded answer and source references are returned to the user

```
User question → Embed question → Search ChromaDB → Retrieve top 5 chunks
      → Inject chunks into GPT-4o prompt → Return answer + sources
```

## Project structure

```
Organization_Chat/
├── main.py                    # FastAPI backend — RAG pipeline and API endpoints
├── ingest.py                  # One-time ingestion script: chunking, embedding, indexing
├── index.html                 # Frontend chat interface
├── requirements.txt           # Python dependencies
├── .env                       # API key (not committed)
├── .env.example               # .env template
├── chroma_db/                 # Local vector store (auto-created on ingestion)
└── *.txt                      # Source documents to be indexed
```

## Setup

### 1. Clone the repo and navigate to the project folder

```bash
cd Organization_Chat
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

```bash
cp .env.example .env
```

Open `.env` and insert your key:

```
OPENAI_API_KEY=sk-proj-...
```

### 5. Index documents (run once)

```bash
python ingest.py
```

Expected output:

```
✓ delivery_methodology.txt — 2 chunks indexed
✓ client_communication.txt — 3 chunks indexed
...
Done. 6 documents, 16 total chunks in ChromaDB.
```

### 6. Start the backend

```bash
uvicorn main:app
```

API available at `http://localhost:8000`.

### 7. Open the frontend

```bash
open index.html        # Mac
start index.html       # Windows
```

Or double-click `index.html` in your file explorer.

## Configuration

Configurable at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match the model used during ingestion |
| `CHAT_MODEL` | `gpt-4o` | Switch to `gpt-4o-mini` to reduce costs |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SYSTEM_PROMPT` | See `main.py` | Controls assistant behaviour and tone |

## Adding new documents

1. Add `.txt` files to the project folder
2. Re-run `python ingest.py`

Documents should include a header for clean source attribution:

```
DOCUMENT: Document Title
SOURCE: filename.txt
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Submit a question, receive an answer with sources |
| `/health` | GET | Server status and indexed document count |

### `/chat` request

```json
{
  "messages": [
    { "role": "user", "content": "What is the escalation procedure for a client complaint?" }
  ]
}
```

### `/chat` response

```json
{
  "answer": "If a client makes a formal complaint, do not respond yourself...",
  "sources": [
    {
      "title": "Client Communication Guidelines",
      "snippet": "Client complaints: If a client makes a formal complaint...",
      "score": 0.91
    }
  ]
}
```

## Estimated costs

| Action | Estimated cost |
|---|---|
| Full document ingestion (one-time) | ~$0.001 |
| Per chat message | ~$0.01–0.02 |
| 50-message demo session | ~$0.50–$1.00 |

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Vector database | ChromaDB (local, no external account needed) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Chat model | OpenAI GPT-4o |
| Frontend | HTML, CSS, Vanilla JavaScript |
