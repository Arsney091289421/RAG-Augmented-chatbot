from flask import Flask, request, jsonify
from query import generate_answer_with_gpt, find_similar_documents, load_embeddings, load_faiss_index
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load models and indexes once at startup
sklearn_embeddings, sklearn_texts = load_embeddings("embeddings/sklearn_embeddings.npy", "embeddings/sklearn_texts.json")
hf_embeddings, hf_texts = load_embeddings("embeddings/hf_embeddings.npy", "embeddings/hf_texts.json")
sklearn_index = load_faiss_index("embeddings/sklearn_index.faiss")
hf_index = load_faiss_index("embeddings/hf_index.faiss")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("question", "")

    # Find similar docs
    results_sklearn = find_similar_documents(user_query, model, sklearn_index, sklearn_embeddings, sklearn_texts, top_k=3)
    results_hf = find_similar_documents(user_query, model, hf_index, hf_embeddings, hf_texts, top_k=3)
    combined_context = "\n\n".join([r[0] for r in results_sklearn + results_hf])

    # Generate GPT answer
    answer = generate_answer_with_gpt(user_query, combined_context)
    return jsonify({"answer": answer})

@app.route("/")
def home():
    return "RAG chatbot API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
