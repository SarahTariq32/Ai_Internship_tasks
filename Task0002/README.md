# Embedding Models Comparison for RAG

## Overview

This project explores the practical application of embedding models in Retrieval-Augmented Generation (RAG) systems. The goal is to compare different embedding models based on their semantic retrieval capabilities, computational efficiency, and storage requirements.

## Project Objectives

* Load and evaluate multiple embedding models.
* Generate vector embeddings for text data.
* Measure embedding dimensions and encoding speed.
* Estimate storage requirements.
* Perform semantic retrieval using cosine similarity.
* Compare retrieval performance across different models.

## Embedding Models Tested

* sentence-transformers/all-MiniLM-L6-v2
* sentence-transformers/all-mpnet-base-v2
* BAAI/bge-small-en-v1.5

## Dataset

A small dataset consisting of 18 technology-related sentences was created. Five semantic queries were used to evaluate the retrieval performance of each embedding model.

## Methodology

The project follows these steps:

1. Load an embedding model.
2. Generate embeddings for the dataset.
3. Measure encoding time.
4. Record embedding vector dimensions.
5. Estimate storage requirements.
6. Encode test queries.
7. Calculate cosine similarity.
8. Retrieve the Top-3 most relevant results.

## Technologies Used

* Python
* SentenceTransformers
* NumPy
* Scikit-learn
* Pandas
* FAISS (installed for vector search experimentation)

## Learning Outcomes

This project provided practical experience with embedding models and semantic search. It demonstrated how text is converted into vector representations and how embeddings are used for information retrieval in RAG systems. The project also highlighted the trade-offs between retrieval quality, encoding speed, and storage requirements when selecting an embedding model.

## Conclusion

The experiment shows that choosing an embedding model depends on the application's requirements. Some models prioritize speed and efficiency, while others provide stronger semantic understanding at the cost of additional computational resources. Understanding these trade-offs is essential when designing practical RAG applications.
