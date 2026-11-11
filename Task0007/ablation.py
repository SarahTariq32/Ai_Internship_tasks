import json
import csv
import os
from groq import Groq
from config import GROQ_API_KEY

from pdf_utils import (
    load_pdfs, create_parent_chunks,
    create_child_chunks, create_fixed_chunks
)
from embedder import create_embeddings, build_faiss_index
from chatbot import ChatBot

client = Groq(api_key=GROQ_API_KEY)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Scoring helpers (same as evaluator.py)
# ---------------------------------------------------------------------------

def ask_groq(prompt):
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(r.choices[0].message.content.strip())
    except:
        return 0.0


def score_faithfulness(context, answer):
    return ask_groq(f"""Rate how faithfully the answer is grounded in the context (0.0–1.0).
1.0 = fully supported. 0.0 = contradicts context.
Context: {context[:1500]}
Answer: {answer}
Reply with ONLY a number.""")


def score_answer_relevancy(question, answer):
    return ask_groq(f"""Rate how well the answer addresses the question (0.0–1.0).
1.0 = direct and complete. 0.0 = off-topic.
Question: {question}
Answer: {answer}
Reply with ONLY a number.""")


def score_context_precision(question, context):
    return ask_groq(f"""Rate how relevant the context is for answering the question (0.0–1.0).
1.0 = exactly what's needed. 0.0 = completely irrelevant.
Question: {question}
Context: {context[:1500]}
Reply with ONLY a number.""")


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("Loading PDFs...")
text = load_pdfs("data/pdfs")

with open("evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)


# ---------------------------------------------------------------------------
# Helper: build index per variant
# ---------------------------------------------------------------------------
def build_index(variant):
    if variant == "fixed":
        chunks = create_fixed_chunks(text)
        idx = build_faiss_index(create_embeddings(chunks))
        return idx, [], chunks, []
    parent_chunks = create_parent_chunks(text)
    child_chunks, parent_mapping = create_child_chunks(parent_chunks)
    idx = build_faiss_index(create_embeddings(child_chunks))
    return idx, parent_chunks, child_chunks, parent_mapping


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------
def run_variant(variant):
    print(f"\n{'='*50}")
    print(f"Variant: {variant}")
    print('='*50)

    index, parent_chunks, child_chunks, parent_mapping = build_index(variant)
    chatbot = ChatBot()

    faithfulness_scores, relevancy_scores, precision_scores = [], [], []

    for i, item in enumerate(evaluation_data):
        question = item["question"]
        print(f"  [{i+1}/{len(evaluation_data)}] {question[:55]}...")

        answer, context = chatbot.ask(
            question, index, child_chunks, parent_chunks, parent_mapping,
            variant=variant
        )

        f = score_faithfulness(context, answer)
        r = score_answer_relevancy(question, answer)
        p = score_context_precision(question, context)

        faithfulness_scores.append(f)
        relevancy_scores.append(r)
        precision_scores.append(p)
        print(f"    F={f:.2f}  R={r:.2f}  P={p:.2f}")

    return {
        "Variant":           variant,
        "Faithfulness":      round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "Answer Relevancy":  round(sum(relevancy_scores)    / len(relevancy_scores),    4),
        "Context Precision": round(sum(precision_scores)    / len(precision_scores),    4)
    }


# ---------------------------------------------------------------------------
# Run all 3 variants
# ---------------------------------------------------------------------------
variant_results = []
for v in ["fixed", "parent", "multi"]:
    scores = run_variant(v)
    variant_results.append(scores)
    print(f"\n  >> {scores}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
with open("results/ablation_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Variant", "Faithfulness", "Answer Relevancy", "Context Precision"]
    )
    writer.writeheader()
    writer.writerows(variant_results)

print("\n\nAblation table saved to results/ablation_table.csv")
print("\nFinal results:")
for row in variant_results:
    print(f"  {row['Variant']:30s}  F={row['Faithfulness']}  R={row['Answer Relevancy']}  P={row['Context Precision']}")