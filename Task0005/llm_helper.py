from groq import Groq

client = Groq(
    api_key=GROQ_API_KEY
)

def generate_multi_queries(query):

    prompt =  f"""
Rewrite the following question for document retrieval.

Rules:

- Keep names unchanged.
- Keep places unchanged.
- Preserve key events.
- Make only small wording changes.
- Do not introduce new concepts.
- Return exactly 3 rewrites.
- One query per line.

Question:

{query}
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

    text = response.choices[0].message.content

    queries = text.split("\n")

    queries = [q.strip() for q in queries if q.strip()]

    return queries