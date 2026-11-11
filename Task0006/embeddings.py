from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from config import *

# Load embedding model once
model = SentenceTransformer(EMBEDDING_MODEL)


def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_index(index, query, top_k):

    query_embedding = create_embeddings([query])

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores[0], indices[0]