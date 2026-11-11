import json
import csv
import os
from groq import Groq
from config import GROQ_API_KEY

from pdf_utils import load_pdfs, create_parent_chunks, create_child_chunks
from embedder import create_embeddings, build_faiss_index
from chatbot import ChatBot

client = Groq(api_key=GROQ_API_KEY)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Scoring helpers — each asks Groq to rate 0.0–1.0
# ---------------------------------------------------------------------------

def score_faithfulness(question, context, answer):
    prompt = f"""Rate how faithfully the answer is grounded in the context on a scale from 0.0 to 1.0.
1.0 = every claim in the answer is directly supported by the context.
0.0 = the answer contradicts or ignores the context entirely.

Context: {context[:1500]}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0. No explanation."""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0


def score_answer_relevancy(question, answer):
    prompt = f"""Rate how well the answer addresses the question on a scale from 0.0 to 1.0.
1.0 = the answer directly and completely answers the question.
0.0 = the answer is completely off-topic or refuses to answer.

Question: {question}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0. No explanation."""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0


def score_context_precision(question, context):
    prompt = f"""Rate how relevant the retrieved context is for answering the question on a scale from 0.0 to 1.0.
1.0 = the context contains exactly the information needed.
0.0 = the context is completely irrelevant to the question.

Question: {question}
Context: {context[:1500]}

Reply with ONLY a number between 0.0 and 1.0. No explanation."""
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
print("Loading PDFs...")
text = load_pdfs("data/pdfs")
parent_chunks = create_parent_chunks(text)
child_chunks, parent_mapping = create_child_chunks(parent_chunks)
child_embeddings = create_embeddings(child_chunks)
index = build_faiss_index(child_embeddings)
chatbot = ChatBot()

# ---------------------------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------------------------
with open("evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
results = []
print(f"\nEvaluating {len(evaluation_data)} questions...\n")

for i, item in enumerate(evaluation_data):
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"[{i+1}/{len(evaluation_data)}] {question[:60]}...")

    answer, context = chatbot.ask(
        question, index, child_chunks, parent_chunks, parent_mapping
    )

    faith  = score_faithfulness(question, context, answer)
    relev  = score_answer_relevancy(question, answer)
    prec   = score_context_precision(question, context)

    print(f"  Faithfulness={faith:.2f}  Relevancy={relev:.2f}  Precision={prec:.2f}")

    results.append({
        "question":          question,
        "ground_truth":      ground_truth,
        "answer":            answer,
        "faithfulness":      faith,
        "answer_relevancy":  relev,
        "context_precision": prec
    })

# ---------------------------------------------------------------------------
# Save generated answers
# ---------------------------------------------------------------------------
with open("evaluation/generated_answers.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Save RAGAS-style CSV
# ---------------------------------------------------------------------------
fieldnames = ["question", "ground_truth", "answer",
              "faithfulness", "answer_relevancy", "context_precision"]

with open("results/ragas_scores.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
avg_faith = sum(r["faithfulness"]      for r in results) / len(results)
avg_relev = sum(r["answer_relevancy"]  for r in results) / len(results)
avg_prec  = sum(r["context_precision"] for r in results) / len(results)

print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)
print(f"  Faithfulness:      {avg_faith:.4f}")
print(f"  Answer Relevancy:  {avg_relev:.4f}")
print(f"  Context Precision: {avg_prec:.4f}")
print(f"\nSaved to results/ragas_scores.csv")