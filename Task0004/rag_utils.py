from pypdf import PdfReader # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer, AutoModelForCausalLM # pyright: ignore[reportMissingImports]
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

client = Groq(

    api_key=os.getenv(
        "GROQ_API_KEY"
    )

)
MODEL_NAME = "BAAI/bge-small-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text

def parent_child_chunking(
    text,
    parent_size=800,
    parent_overlap=50,
    child_size=200
):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False      
    )

    parent_chunks = []
    child_chunks = []
    parent_child_map = {}

    child_id = 0

    step = parent_size - parent_overlap

    for parent_id, start in enumerate(
        range(0, len(tokens), step)
    ):

        parent_tokens = tokens[
            start:start + parent_size
        ]

        parent_text = tokenizer.decode(
            parent_tokens,
            skip_special_tokens=True
        )

        parent_chunks.append(parent_text)

        parent_child_map[parent_id] = []

        for j in range(
            0,
            len(parent_tokens),
            child_size
        ):

            child_tokens = parent_tokens[
                j:j + child_size
            ]

            child_text = tokenizer.decode(
                child_tokens,
                skip_special_tokens=True
            )

            child_chunks.append({

                "child_id": child_id,

                "parent_id": parent_id,

                "text": child_text

            })

            parent_child_map[parent_id].append(
                child_id
            )

            child_id += 1

    return (
        parent_chunks,
        child_chunks,
        parent_child_map
    )

def create_embeddings(child_chunks):

    texts = [
        child["text"]
        for child in child_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    faiss.normalize_L2(
        embeddings
    )

    index.add(
        embeddings
    )

    return index

def retrieve_chunks(
    query,
    index,
    child_chunks,
    parent_chunks,
    top_k=10
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    parent_ids = []

    for idx in indices[0]:

        parent_id = child_chunks[idx]["parent_id"]

        if parent_id not in parent_ids:

            parent_ids.append(parent_id)

    results = []

    for parent_id in parent_ids:

        results.append({

            "parent_id": parent_id,

            "text": parent_chunks[parent_id]

        })

    return results

def generate_answer(

    question,

    context

):

    prompt = f"""

You are a helpful assistant.

Use ONLY the provided context.

Do not make assumptions or infer facts that are not explicitly stated.

If the answer cannot be found in the context, reply:

"The provided context does not contain enough information."

Context:

{context}

Question:

{question}

Answer:

"""

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {

                "role":"user",

                "content":prompt

            }

        ]

    )

    return response.choices[0].message.content