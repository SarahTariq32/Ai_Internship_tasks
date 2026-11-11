from pypdf import PdfReader
from chunkers import *
from embedder import create_embeddings
from faiss_helper import build_faiss_index, search_index
from sklearn.metrics.pairwise import cosine_similarity
from llm_helper import generate_multi_queries

def display_stats(name, chunks):

    if not chunks:
        print(f"\n{name}")
        print("No chunks available.")
        return
    
    lengths = [len(chunk) for chunk in chunks]

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print(f"Chunk Count   : {len(chunks)}")
    print(f"Average Length: {sum(lengths)/len(lengths):.2f}")
    print(f"Shortest Chunk: {min(lengths)}")
    print(f"Longest Chunk : {max(lengths)}")

    # print("\nFirst Chunk Preview:\n")
    # print(chunks[0][:500])

# -----------------------
# Load PDF
# -----------------------

pdf_path = "data/document.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# -----------------------
# PDF Statistics
# -----------------------

print("=" * 50)
print("PDF LOADED")
print("=" * 50)

print(f"Total Pages: {len(reader.pages)}")
print(f"Total Characters: {len(text)}")

# print("\nPreview:\n")
# print(text[:1000])

# -----------------------
# Save extracted text
# -----------------------

with open("data/extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("\nText saved successfully!")

# -----------------------
# Chunking Section
# -----------------------

print("\n" + "=" * 50)
print("CHUNKING")
print("=" * 50)

# Fixed Chunking
fixed_chunks = fixed_chunking(text)
display_stats("Fixed Chunking", fixed_chunks)

# Overlap Chunking
overlap_chunks = overlap_chunking(text)
display_stats("Overlap Chunking", overlap_chunks)

# Sentence Chunking
sentence_chunks = sentence_chunking(text)
display_stats("Sentence Chunking", sentence_chunks)

# Semantic Chunking
semantic_chunks = semantic_chunking(text)
display_stats("Semantic Chunking", semantic_chunks)

# Recursive Chunking
recursive_chunks = recursive_chunking(text)
display_stats("Recursive Chunking", recursive_chunks)

# Parent Child Chunking
parent_chunks, child_chunks, parent_child_map = parent_child_chunking(text)

print("\n" + "-" * 50)
display_stats("Parent Chunks", parent_chunks)

display_stats("Child Chunks", child_chunks)

print("\nParent -> Child Mapping Example:")
print(parent_child_map[0])

print("\n" + "=" * 50)
print("ALL CHUNKING STRATEGIES COMPLETED")
print("=" * 50)

print("Chunking functions ready!")

all_chunks = {

    "Fixed": fixed_chunks,

    "Overlap": overlap_chunks,

    "Sentence": sentence_chunks,

    "Semantic": semantic_chunks,

    "Recursive": recursive_chunks,

    "ParentChild": child_chunks

}

print("\n" + "=" * 50)
print("EMBEDDINGS + FAISS")
print("=" * 50)

all_embeddings = {}

all_indexes = {}

for name, chunks in all_chunks.items():

    print(f"\n{name}")

    embeddings = create_embeddings(chunks)

    all_embeddings[name] = embeddings

    print(f"Embeddings Shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)

    all_indexes[name] = index

    print("FAISS Index Built Successfully")

print("\n" + "=" * 50)
print("QUERY TESTING")
print("=" * 50)

queries = [

    # ---------- SINGLE-CHUNK QUESTIONS ----------

    "How does Cassie first meet the mysterious boy at the diner?",

    "How does Cassie figure out what kind of eggs the mysterious boy likes?",

    "What warning does Michael give Cassie about Agent Briggs and the Naturals program?",

    "Why does Cassie decide to call Agent Briggs?",

    "How does Agent Briggs explain why Cassie was chosen for the Naturals program?",

    "Why does Michael warn Cassie against joining the Naturals program?",

    "How does Briggs convince Cassie that the FBI program is real?",

    "Why does Michael leave without introducing himself properly at the diner?",

    "How does Cassie first meet Lia, and what impression does Lia make?",

    "Why does Briggs believe teenagers can become successful Naturals?",


    # ---------- MULTI-CHUNK QUESTIONS ----------

    "How did Cassie's mother teach her to understand people, and how do those lessons continue to influence Cassie?",

    "Compare Cassie's talent for reading people with Michael's talent for reading emotions.",

    "How do Michael's warnings and Briggs's promises create different views of the Naturals program?",

    "How do Cassie's Italian relatives and Agent Briggs represent different possible futures for her?",

    "How does Cassie's missing mother influence both her personality and her decision to join the Naturals?",

    "How do Cassie's first interactions with the other Naturals make her feel out of place?",

    "How do the alternating 'YOU' chapters help the reader understand the killer's personality?",

    "How does the novel use first impressions to mislead both Cassie and the reader about different characters?",

    "How does Cassie's idea of family change during the early part of the novel?",

    "How do the Naturals' unique talents help them solve problems while also creating challenges in their personal lives?"

]

results = []

for query in queries:

    print("\n")
    print("="*50)

    print("ORIGINAL QUERY:")
    print(query)

    print("="*50)

    expanded_queries = [query]

    expanded_queries.extend(
        generate_multi_queries(query)
    )   
    print("\nExpanded Queries:")

    for q in expanded_queries:
        print("-", q)

    for name in all_indexes:

        print("\n")
        print(name)

        retrieved_chunks = set()

        for expanded_query in expanded_queries:

            query_embedding = create_embeddings(
                [expanded_query]
            )

            distances, indices = search_index(

                all_indexes[name],

                query_embedding,

                k=1

            )

            retrieved_chunks.add(
                indices[0][0]
            )

        print()

        print(
            "Unique Retrieved Chunks:"
        )

        for chunk_index in retrieved_chunks:

            query_embedding = create_embeddings(
                [query]
            )

            retrieved_embedding = \
            all_embeddings[name][chunk_index]

            cos_score = cosine_similarity(

                query_embedding,

                retrieved_embedding.reshape(1,-1)

            )[0][0]

            retrieved_chunk = \
            all_chunks[name][chunk_index]

            print()

            print(
                f"Chunk Index: {chunk_index}"
            )

            print(
                f"Cosine: {cos_score:.4f}"
            )

            print()

            print(
                retrieved_chunk[:200]
            )

            print(
                "\n"+"-"*40
            )

            results.append({

                "Query":query,

                "ExpandedQueryCount":
                len(expanded_queries),

                "Strategy":name,

                "Chunk":chunk_index,

                "Cosine":cos_score,

                "Preview":
                retrieved_chunk[:100]

            })