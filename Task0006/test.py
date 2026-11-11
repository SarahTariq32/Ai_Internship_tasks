from pdf_utils import *
from embeddings import *
from chatbot import ChatBot

print("Loading PDFs...")

text = load_pdfs("data/pdfs")

print("Creating parent chunks...")

parents = create_parent_chunks(text)

print("Creating child chunks...")

children, mapping = create_child_chunks(parents)

print("Creating embeddings...")

embeddings = create_embeddings(children)

print("Building FAISS index...")

index = build_faiss_index(
    embeddings
)

print()
print("==========================")
print("PDF Chatbot Ready!")
print("Type 'exit' to quit.")
print("==========================")

bot = ChatBot()

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("\nGoodbye!")
        break

    answer = bot.ask(

        question,

        index,

        children,

        parents,

        mapping

    )

    print()

    print("Bot:")
    print(answer)