from groq import Groq

from config import *

from retriever import (
    retrieve_context,
    retrieve_parent_context,
    retrieve_fixed_context
)

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

            parent_mapping,

            variant=PIPELINE

    ):

        if variant == "fixed":

            context = retrieve_fixed_context(

                question,

                index,

                child_chunks

            )

        elif variant == "parent":

            context = retrieve_parent_context(

                question,

                index,

                child_chunks,

                parent_chunks,

                parent_mapping

            )

        else:

            context = retrieve_context(

                question,

                index,

                child_chunks,

                parent_chunks,

                parent_mapping

            )

        if not context.strip():

            return (

                "Context not enough to answer.",

                context

            )

        history = self.get_history()

        history_text = ""

        for role, message in history:

            history_text += f"{role}: {message}\n"

        prompt = f"""
You are a helpful PDF chatbot.

Answer using only the retrieved context.

If the retrieved context does not contain enough
information to answer the question, reply exactly:

Context not enough to answer.

Do not make up information.

Conversation History:

{history_text}

Retrieved Context:

{context}

Question:

{question}
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

        answer = response.choices[0].message.content.strip()

        self.history.append(
            ("User", question)
        )

        self.history.append(
            ("Bot", answer)
        )

        return answer, context