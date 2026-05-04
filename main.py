import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
TOP_K           = 5
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "internal_docs"
SYSTEM_PROMPT   = """You are an internal employee assistant.
Your role is to help employees with questions about company methods,
processes, policies, and best practices based strictly on the provided
internal documentation.

Guidelines:
- Answer only based on the provided context. Do not make things up.
- If the context does not contain enough information to answer, say so clearly.
- Be concise, professional, and direct.
- When relevant, reference which document or section your answer comes from.
- Format answers clearly using bullet points or numbered lists when appropriate."""
# ─────────────────────────────────────────────────────────────────────────────

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

app = FastAPI(title="Employee RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    namespace: str = ""     # unused with Chroma, kept for frontend compatibility

class Source(BaseModel):
    title: str
    snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Core RAG logic ────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    """Query ChromaDB and return top-k matching chunks."""
    results = collection.query(
        query_texts=[query],
        n_results=min(TOP_K, collection.count())
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        meta  = results["metadatas"][0][i] or {}
        score = round(1 - results["distances"][0][i], 3)
        chunks.append({
            "title":   meta.get("title", meta.get("source", "Internal Document")),
            "snippet": doc,
            "score":   max(0.0, score),
        })
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {chunk['title']}\n{chunk['snippet']}")
    return "\n\n---\n\n".join(parts)


def chat_with_context(messages: list[Message], context: str) -> str:
    """Call OpenAI chat with retrieved context injected into the last user message."""
    history = [{"role": m.role, "content": m.content} for m in messages[:-1]]
    last_user_msg = messages[-1].content

    augmented = (
        f"Use the following internal documentation to answer the question.\n\n"
        f"=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        f"Question: {last_user_msg}"
    )

    openai_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": augmented},
    ]

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=openai_messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    last_user_query = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), None
    )
    if not last_user_query:
        raise HTTPException(status_code=400, detail="No user message found.")

    if collection.count() == 0:
        raise HTTPException(status_code=503, detail="No documents indexed yet. Run the ingestion script first.")

    chunks  = retrieve(last_user_query)
    context = build_context(chunks)
    answer  = chat_with_context(req.messages, context)
    sources = [Source(**c) for c in chunks if c["snippet"]]

    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {
        "status":            "ok",
        "documents_indexed": collection.count(),
        "embedding_model":   EMBEDDING_MODEL,
        "chat_model":        CHAT_MODEL,
    }
