import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-20b"


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )


client = Groq(
    api_key=GROQ_API_KEY
)


def generate_answer(prompt):
    """
    Generate an answer using Groq LLM.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are ResearchLens, an AI research paper assistant. "
                    "Answer questions using only the provided research paper "
                    "context. Do not invent information. "
                    "If the answer is not present in the context, "
                    "say that the information is not available in the "
                    "provided research paper."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content