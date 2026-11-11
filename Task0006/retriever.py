from groq import Groq

from config import *
from embeddings import search_index

client = Groq(
    api_key=GROQ_API_KEY
)


def generate_multi_queries(query):

    prompt = f"""
Generate {NUM_MULTI_QUERIES} different versions of the question.

Question:
{query}

Return one question per line.
"""

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )

    queries = response.choices[0].message.content.split("\n")

    queries = [q.strip() for q in queries if q.strip()]

    return queries


def retrieve_context(
        query,
        index,
        child_chunks,
        parent_chunks,
        parent_mapping

):

    queries = generate_multi_queries(query)

    unique_parents = []

    seen = set()

    for q in queries:

        scores, child_ids = search_index(
            index,
            q,
            TOP_K
        )

        for child_id in child_ids:

            parent_id = parent_mapping[child_id]

            if parent_id not in seen:

                seen.add(parent_id)

                unique_parents.append(parent_id)

    context = ""

    for parent_id in unique_parents:

        context += parent_chunks[parent_id]
        context += "\n\n"

    return context