
from pathlib import Path
import hashlib
import chromadb


# Store the database inside the experiment folder
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "who_diabetes_experiment"


def store_chunks(chunks, embeddings):

    if not chunks:
        raise ValueError("No chunks provided.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = []

    for chunk in chunks:

        unique_text = (
            f"{chunk['metadata']['source']}|"
            f"{chunk['metadata'].get('page_number', 'N/A')}|"
            f"{chunk['text']}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    collection.upsert(
        ids=ids,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
        embeddings=embeddings.tolist()
    )

    return collection

