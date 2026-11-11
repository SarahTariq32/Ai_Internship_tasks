import faiss
import numpy as np


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    return index

def search_index(index, query_embedding, k=3):

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    ).reshape(1,-1)

    distances, indices = index.search(
        query_embedding,
        k
    )

    return distances, indices