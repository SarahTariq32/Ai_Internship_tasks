from rag_utils import (
    load_pdf,
    parent_child_chunking,
    create_embeddings,
    build_faiss_index,
    retrieve_chunks,
    generate_answer
)
PDF_PATH = "document.pdf"

# Load PDF
text = load_pdf(PDF_PATH)

# Create Parent-Child Chunks
parent_chunks, child_chunks, parent_child_map = parent_child_chunking(text)

embeddings = create_embeddings(
    child_chunks
)

index = build_faiss_index(
    embeddings
)
# Display Results
print("=" * 50)
print("PARENT-CHILD CHUNKING")
print("=" * 50)

print(f"\nTotal Characters: {len(text)}")

print(f"\nTotal Parent Chunks: {len(parent_chunks)}")

print(f"Total Child Chunks: {len(child_chunks)}")

print("\n" + "=" * 50)
print("FIRST PARENT CHUNK")
print("=" * 50)

print(parent_chunks[0][:500])

print("\n" + "=" * 50)
print("FIRST CHILD CHUNK")
print("=" * 50)

print(child_chunks[0]["text"][:300])

print("\n" + "=" * 50)
print("PARENT -> CHILD MAPPING")
print("=" * 50)

print(f"Parent 0 contains Child IDs: {parent_child_map[0]}")

print("\n" + "="*50)
print("EMBEDDINGS")
print("="*50)

print(
    f"Embedding Shape: {embeddings.shape}"
)

print("\nFAISS Index Built!")

print(
    f"Vectors Stored: {index.ntotal}"
)


print("\n" + "="*50)
print("RAG TESTING")
print("="*50)

questions = [

    "Who is Cassie?",

    "Who is Dean?",

    "Who is Agent Locke?",

    "Who is Michael?",

    "Why is Cassie important?",

    "What is Agent Locke investigating?",

    "What connects the victims?",

    "Why does Briggs worry about Cassie?",

    "What is Cassie's favorite food?",

    "What happened after the final chapter?"

]

for query in questions:

    print("\n" + "="*60)
    print(f"QUESTION: {query}")
    print("="*60)

    results = retrieve_chunks(

        query,

        index,

        child_chunks,

        parent_chunks

    )

    retrieved_context = ""

    for i, result in enumerate(results[:3]):

        print("\n" + "-"*50)

        print(f"Retrieved Parent {i+1}")

        print(f"Parent ID: {result['parent_id']}")

        print()

        print(result["text"][:400])

        retrieved_context += (
            result["text"]
            + "\n\n"
        )

    answer = generate_answer(

        query,

        retrieved_context

    )

    print("\n")

    print("-"*50)

    print("GENERATED ANSWER")

    print("-"*50)

    print(answer)

    print("\n")