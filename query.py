import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_embeddings(embeddings_path, texts_path):
    embeddings = np.load(embeddings_path)
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return embeddings, texts

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def find_similar_documents(query, model, faiss_index, embeddings, texts, top_k=5):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(query_embedding, top_k)
    results = [(texts[i], D[0][idx]) for idx, i in enumerate(I[0])]
    return results

def generate_answer_with_gpt(query, context):
    prompt = f"You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # Load embeddings and indexes
    sklearn_embeddings, sklearn_texts = load_embeddings("embeddings/sklearn_embeddings.npy", "embeddings/sklearn_texts.json")
    hf_embeddings, hf_texts = load_embeddings("embeddings/hf_embeddings.npy", "embeddings/hf_texts.json")

    sklearn_index = load_faiss_index("embeddings/sklearn_index.faiss")
    hf_index = load_faiss_index("embeddings/hf_index.faiss")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    user_query = input("Enter your question: ")

    # Search both indexes
    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, top_k=3)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, top_k=3)

    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])

    answer = generate_answer_with_gpt(user_query, combined_context)
    print("\nAnswer:")
    print(answer)
