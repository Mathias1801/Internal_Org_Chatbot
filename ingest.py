"""
ingest.py — Load documents from the /docs folder into ChromaDB.
Run this once before starting the server:  python ingest.py
"""

import os
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_FOLDER     = "."
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "internal_docs"
EMBEDDING_MODEL = "text-embedding-3-small"   # must match main.py
CHUNK_SIZE      = 400    # words per chunk
CHUNK_OVERLAP   = 50     # words of overlap between chunks
# ─────────────────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def parse_metadata(text: str, filename: str) -> dict:
    """Extract title and source from document header if present."""
    title  = filename.replace("_", " ").replace(".txt", "").title()
    source = filename
    for line in text.splitlines()[:5]:
        if line.startswith("DOCUMENT:"):
            title = line.replace("DOCUMENT:", "").strip()
        if line.startswith("SOURCE:"):
            source = line.replace("SOURCE:", "").strip()
    return {"title": title, "source": source}


def ingest():
    # Connect to Chroma
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL
    )
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # Find all .txt files in docs folder
    doc_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".txt")]
    if not doc_files:
        print(f"No .txt files found in {DOCS_FOLDER}/")
        return

    total_chunks = 0

    for filename in doc_files:
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        meta   = parse_metadata(text, filename)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        ids       = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"title": meta["title"], "source": meta["source"]} for _ in chunks]

        # Add to Chroma (embeddings generated automatically)
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"✓ {filename} — {len(chunks)} chunks indexed")
        total_chunks += len(chunks)

    print(f"\nDone. {len(doc_files)} documents, {total_chunks} total chunks in ChromaDB.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


if __name__ == "__main__":
    ingest()
