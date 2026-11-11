from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts):
    """
    Generate normalized embeddings for a list of texts.
    """

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    return embeddings


def build_faiss_index(embeddings):
    """
    Build a FAISS index using cosine similarity
    (implemented as inner product on normalized vectors).
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_index(index, query, top_k):
    """
    Search the FAISS index for the most relevant chunks.
    Returns:
        scores
        indices
    """

    query_embedding = create_embeddings([query])

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]