import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from scripts.query_langchain_pgvector import find_and_generate_answer

load_dotenv()

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("question", "")


    result = find_and_generate_answer(user_query, top_k=5)

    return jsonify(result)

@app.route("/")
def home():
    return send_from_directory("templates", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
