from flask import Flask, request, jsonify, send_from_directory
from scripts.query_pgvector import find_similar_documents_pgvector
from scripts.generate_answer import generate_answer_with_gpt
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load embedding model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("question", "")

    # Encode user query to embedding
    query_embedding = model.encode(user_query)

    # Query pgvector to get top-k similar chunks
    results_pg = find_similar_documents_pgvector(query_embedding, top_k=6)

    # Combine content for GPT context
    combined_context = "\n\n".join([r["content"] for r in results_pg])

    # Generate final answer using OpenAI
    answer = generate_answer_with_gpt(user_query, combined_context)

    # Build sources list for UI
    sources = [
        {
            "snippet": r["content"][:120].replace('\n', ' '),
            "source": r["source_file"],
            "chunk_id": r["chunk_id"],
            "relevance_score": round(float(r["distance"]), 4)
        }
        for r in results_pg
    ]

    return jsonify({
        "answer": answer,
        "sources": sources
    })

@app.route("/")
def home():
    return send_from_directory("templates", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
