import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Load config.json
with open(os.getenv("CONFIG_PATH", "config.json"), 'r') as f:
    config = json.load(f)

TEMPERATURE = config.get("temperature", 0.3)
TOP_K = config.get("top_k", 3)
MODEL_NAME = config.get("model_name", "gpt-3.5-turbo")

#check config
print(f"Using model: {MODEL_NAME}, temperature: {TEMPERATURE}, top_k: {TOP_K}")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_embeddings(embeddings_path, texts_path):
    embeddings = np.load(embeddings_path)
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return embeddings, texts

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def find_similar_documents(query, model, faiss_index, embeddings, texts, source_label, top_k=5):
    query_embedding = model.encode([query])
    D, I = faiss_index.search(query_embedding, top_k)
    results = [(texts[i], D[0][idx], source_label) for idx, i in enumerate(I[0])]
    return results

def generate_answer_with_gpt(query, context):
    system_prompt = (
        "You are a helpful assistant. "
        "Use ONLY the following provided context to answer the user's question. "
        "If the answer is not found in the context, reply with: "
        "'I don't know based on the provided documents.' "
        "Do not make up information.\n\n"
        f"Context:\n{context}"
    )
    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=TEMPERATURE  # # temperature ensures deterministic, fact-based responses, minimizing hallucination
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
    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, source_label="sklearn", top_k=TOP_K)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, source_label="transformers", top_k=TOP_K)


    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])

    print("\nRetrieved sources:")
    for idx, (text, score, source_label) in enumerate(results_sklearn + results_hf):
        snippet_preview = text[:100].replace('\n', ' ')
        print(f"[{idx+1}] (score={score:.4f}, source={source_label}): {snippet_preview}")


    answer = generate_answer_with_gpt(user_query, combined_context)
    print("\nAnswer:")
    print(answer)
