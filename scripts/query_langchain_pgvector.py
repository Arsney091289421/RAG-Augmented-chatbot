import os
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from scripts.generate_answer import generate_answer_with_gpt
except ModuleNotFoundError:
    from generate_answer import generate_answer_with_gpt


load_dotenv()

PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embed_model,
    collection_name="my_collection"
)

def find_and_generate_answer(user_query, top_k=5):
    docs_and_scores = vectorstore.similarity_search_with_score(user_query, k=top_k)

    context = "\n\n".join([doc.page_content for doc, score in docs_and_scores])

    answer = generate_answer_with_gpt(user_query, context)

    sources = []
    for doc, score in docs_and_scores:
        sources.append({
            "snippet": doc.page_content[:120].replace("\n", " "),
            "source": doc.metadata.get("source_file", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "relevance_score": round(float(score), 4)
        })

    return {
        "answer": answer,
        "sources": sources
    }
