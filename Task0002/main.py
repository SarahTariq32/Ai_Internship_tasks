import time
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "Artificial Intelligence enables machines to think.",
    "Machine Learning is a subset of AI.",
    "Deep Learning uses neural networks.",
    "Python is widely used in AI.",
    "SQL is used for databases.",
    "DBMS manages data efficiently.",
    "Operating systems manage computer resources.",
    "Virtual memory extends RAM.",
    "Process scheduling allocates CPU time.",
    "Linux is an open source OS.",
    "A compiler translates source code.",
    "Algorithms solve computational problems.",
    "Data structures organize information.",
    "Cloud computing provides online services.",
    "Cybersecurity protects digital systems.",
    "Networks connect computers together.",
    "RAM stores temporary data.",
    "Storage devices keep permanent data."
]

queries = [
    "What is AI?",
    "What manages data?",
    "What extends RAM?",
    "What translates code?",
    "What protects systems?"
]

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5"
]

for model_name in models:

    print("\n")
    print("="*60)
    print("MODEL:", model_name)

    model = SentenceTransformer(model_name)

    start = time.time()

    embeddings = model.encode(sentences)

    end = time.time()

    encode_time = end - start

    dimension = len(embeddings[0])

    storage = dimension * 4 * 1000 / 1024

    print("Vector Dimension:", dimension)
    print("Encoding Time:", round(encode_time,4),"seconds")
    print("Estimated Storage for 1000 chunks:",round(storage,2),"KB")

    print("\nRetrieval Test")

    for query in queries:

        query_embedding = model.encode([query])

        similarities = cosine_similarity(
            query_embedding,
            embeddings
        )[0]

        top3 = similarities.argsort()[-3:][::-1]

        print("\nQuery:",query)

        for rank,index in enumerate(top3):

            print(
                rank+1,
                sentences[index],
                "Score:",
                round(similarities[index],4)
            )