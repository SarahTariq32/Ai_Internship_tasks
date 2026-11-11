from pdf_utils import load_pdfs, create_parent_chunks, create_child_chunks
from embeddings import create_embeddings, build_faiss_index
from chatbot import ChatBot

from config import HISTORY_STRATEGY, N_MESSAGES


def initialize_chatbot():

    print("=" * 40)
    print("        PDF CHATBOT")
    print("=" * 40)

    print("\nLoading PDF(s)...")

    text = load_pdfs("data/pdfs")

    print("Creating parent chunks...")
    parent_chunks = create_parent_chunks(text)

    print("Creating child chunks...")
    child_chunks, parent_mapping = create_child_chunks(parent_chunks)

    print("Creating embeddings...")
    embeddings = create_embeddings(child_chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("\nInitialization Complete!")

    print("\nConversation Mode:")

    if HISTORY_STRATEGY == 1:
        print(f"Previous {N_MESSAGES} messages")
    else:
        print("Full conversation history")

    print("\nType 'exit' to quit.")
    print("-" * 40)

    return (
        index,
        parent_chunks,
        child_chunks,
        parent_mapping
    )


def main():

    (
        index,
        parent_chunks,
        child_chunks,
        parent_mapping
    ) = initialize_chatbot()

    chatbot = ChatBot()

    while True:

        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            answer = chatbot.ask(
                question,
                index,
                child_chunks,
                parent_chunks,
                parent_mapping
            )

            print("\nBot:")
            print(answer)

        except Exception as e:

            print("\nError:")
            print(e)


if __name__ == "__main__":
    main()