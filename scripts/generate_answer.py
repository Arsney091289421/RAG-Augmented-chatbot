import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer_with_gpt(user_question, context):
    """
    Generate answers based on user questions and retrieved context with OpenAI Chat Completion
    """
    system_prompt = (
        "You are a helpful AI assistant. "
        "Use the following context to answer the user's question as accurately as possible. "
        "If you don't know the answer, say you don't know."
    )

    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + context},
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content
