# Employee Assistant — Internal Knowledge Chat

A RAG (Retrieval-Augmented Generation) chat application that allows employees to ask questions about internal company documents. Built with FastAPI, ChromaDB, and the OpenAI API.

## How it works

1. Internal documents are chunked and embedded using OpenAI's `text-embedding-3-small` model
2. Embeddings are stored locally in ChromaDB (a vector database)
3. When an employee asks a question, the question is embedded and the most relevant document chunks are retrieved
4. The retrieved chunks are passed as context to GPT-4o, which generates a grounded answer
5. The answer and the source documents used are returned to the user

```
User question → Embed question → Search ChromaDB → Retrieve top 5 chunks
      → Inject chunks into GPT-4o prompt → Return answer + sources
```

## Project structure

```
Organization_Chat/
├── main.py                    # FastAPI backend — RAG pipeline and API endpoints
├── ingest.py                  # One-time script to embed and index documents
├── index.html                 # Frontend chat interface (open directly in browser)
├── requirements.txt           # Python dependencies
├── .env                       # API key (not committed to git)
├── .env.example               # Template for .env
├── chroma_db/                 # Local vector database (auto-created by ingest.py)
└── *.txt                      # Internal documents to be indexed
```

## Prerequisites

- Python 3.10+
- An OpenAI API account with credits ([platform.openai.com](https://platform.openai.com))

## Setup

### 1. Clone the repository and navigate to the project folder

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

Open `.env` and add your key:

```
OPENAI_API_KEY=sk-proj-...
```

### 5. Index the documents (run once)

```bash
python ingest.py
```

You should see each document being chunked and indexed:

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

The API will be available at `http://localhost:8000`.

### 7. Open the frontend

```bash
open index.html        # Mac
start index.html       # Windows
```

Or double-click `index.html` in your file explorer.

## Configuration

Key settings are at the top of `main.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match the model used during ingestion |
| `CHAT_MODEL` | `gpt-4o` | Switch to `gpt-4o-mini` to reduce costs |
| `TOP_K` | `5` | Number of document chunks retrieved per query |
| `SYSTEM_PROMPT` | See `main.py` | Controls the assistant's behaviour and tone |

## Adding new documents

1. Add `.txt` files to the project folder
2. Re-run `python ingest.py`

The ingestion script will index all `.txt` files it finds. Documents should include a header like:

```
DOCUMENT: Document Title
SOURCE: filename.txt
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | POST | Send a message and receive an answer with sources |
| `/health` | GET | Check server status and number of indexed documents |

### Example `/chat` request

```json
{
  "messages": [
    { "role": "user", "content": "What is the escalation procedure for a client complaint?" }
  ]
}
```

### Example `/chat` response

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

| Action | Cost |
|---|---|
| Indexing all documents (one time) | ~$0.001 |
| Per chat message | ~$0.01–0.02 |
| 50-message demo session | ~$0.50–$1.00 |

## Tech stack

- **Backend:** Python, FastAPI, Uvicorn
- **Vector database:** ChromaDB (local, no account required)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Chat model:** OpenAI GPT-4o
- **Frontend:** HTML, CSS, vanilla JavaScript
