
from sentence_transformers import SentenceTransformer

# Load model from local cache only
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True
)


def generate_embeddings(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings
