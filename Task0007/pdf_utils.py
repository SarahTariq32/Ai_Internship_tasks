from pypdf import PdfReader
from config import *

import os



def load_pdfs(folder_path):
    text = ""

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            reader = PdfReader(pdf_path)

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    return text



def create_parent_chunks(text):
    parents = []

    start = 0

    while start < len(text):
        end = start + PARENT_CHUNK_SIZE

        parents.append(text[start:end])

        start += PARENT_CHUNK_SIZE - CHUNK_OVERLAP

    return parents



def create_child_chunks(parent_chunks):

    child_chunks = []

    parent_mapping = []

    for parent_id, parent in enumerate(parent_chunks):

        start = 0

        while start < len(parent):

            end = start + CHILD_CHUNK_SIZE

            child = parent[start:end]

            child_chunks.append(child)

            parent_mapping.append(parent_id)

            start += CHILD_CHUNK_SIZE - CHUNK_OVERLAP

    return child_chunks, parent_mapping



def create_fixed_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + FIXED_CHUNK_SIZE

        chunks.append(
            text[start:end]
        )

        start += FIXED_CHUNK_SIZE - FIXED_CHUNK_OVERLAP

    return chunks



