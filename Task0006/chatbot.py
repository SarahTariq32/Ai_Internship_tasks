from groq import Groq

from config import *
from retriever import retrieve_context

client = Groq(
    api_key=GROQ_API_KEY
)


class ChatBot:

    def __init__(self):

        self.history = []

    def get_history(self):

        if HISTORY_STRATEGY == 1:
            return self.history[-N_MESSAGES:]

        return self.history

    def ask(
            self,
            question,
            index,
            child_chunks,
            parent_chunks,
            parent_mapping
    ):

        context = retrieve_context(
            question,
            index,
            child_chunks,
            parent_chunks,
            parent_mapping
        )
        if not context.strip():

            return "Context not enough to answer."

        history = self.get_history()

        history_text = ""

        for role, message in history:

            history_text += f"{role}: {message}\n"

        prompt = f"""
You are a helpful PDF chatbot.

Answer using only the retrieved context.

If the context does not contain enough information,
reply exactly:

Context not enough to answer.

Do not make up information.

Conversation History:

{history_text}

Retrieved Context:

{context}

Question:

{question}

If the answer is not present in the context,
reply exactly:

Context not enough to answer.
"""

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        answer = response.choices[0].message.content

        self.history.append(
            ("User", question)
        )

        self.history.append(
            ("Bot", answer)
        )

        return answer