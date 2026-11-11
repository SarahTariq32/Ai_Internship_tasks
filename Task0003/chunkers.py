import tiktoken
import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def fixed_chunking(text, chunk_size=200):

    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    chunks = []

    for i in range(0, len(tokens), chunk_size):

        chunk_tokens = tokens[i:i + chunk_size]

        chunk_text = encoding.decode(chunk_tokens)

        chunks.append(chunk_text)

    return chunks



def overlap_chunking(text, chunk_size=200, overlap=50):

    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(tokens), step):

        chunk_tokens = tokens[i:i + chunk_size]

        chunk_text = encoding.decode(chunk_tokens)

        chunks.append(chunk_text)

    return chunks



def sentence_chunking(text):

    sentences = nltk.sent_tokenize(text)

    return sentences



# import re

# def semantic_chunking(text, target_size=500):

#     paragraphs = re.split(r'\n\s*\n+', text)

#     # If PDF formatting is poor
#     if len(paragraphs) < 10:
#         paragraphs = re.split(r'(?<=[.!?])\s+', text)

#     chunks = []
#     current = ""

#     for p in paragraphs:
#         p = p.strip()
#         if not p:
#             continue

#         if len(current) + len(p) <= target_size:
#             current += " " + p
#         else:
#             chunks.append(current.strip())
#             current = p

#     if current:
#         chunks.append(current.strip())

#     return chunks


import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_chunking(text, threshold=0.7):

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return sentences

    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = cosine_similarity(
            [embeddings[i-1]],
            [embeddings[i]]
        )[0][0]

        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))

    return chunks

def recursive_chunking(text):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=200,

        chunk_overlap=50,

        separators=[

            "\n\n",

            "\n",

            ". ",

            " ",

            ""

        ]

    )

    chunks = text_splitter.split_text(text)

    return chunks


def parent_child_chunking(
    text,
    parent_size=400,
    child_size=100
):

    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    parent_chunks = []

    child_chunks = []

    parent_child_map = {}

    child_id = 0

    # Create Parent Chunks
    for parent_id, i in enumerate(range(0, len(tokens), parent_size)):

        parent_tokens = tokens[i:i + parent_size]

        parent_text = encoding.decode(parent_tokens)

        parent_chunks.append(parent_text)

        parent_child_map[parent_id] = []

        # Create Child Chunks
        for j in range(0, len(parent_tokens), child_size):

            child_tokens = parent_tokens[j:j + child_size]

            child_text = encoding.decode(child_tokens)

            child_chunks.append(child_text)

            parent_child_map[parent_id].append(child_id)

            child_id += 1

    return parent_chunks, child_chunks, parent_child_map