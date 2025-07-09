import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 环境变量
load_dotenv()

PG_DB = os.getenv("PG_DB", "ragdb")
PG_USER = os.getenv("PG_USER", "daniel")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")

CONNECTION_STRING = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

DST_PATH = os.getenv("DST_PATH", "./docs_clean")
EMBEDDINGS_FILES = [
    os.path.join(DST_PATH, "embeddings", "sklearn_embeddings.json"),
    os.path.join(DST_PATH, "embeddings", "transformers_embeddings.json")
]

embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

docs = []
for file in EMBEDDINGS_FILES:
    with open(file, "r") as f:
        data = json.load(f)
        for item in data:
            docs.append(
                Document(
                    page_content=item["content"],
                    metadata={
                        "source_file": item.get("source_file", ""),
                        "chunk_id": item.get("chunk_id", "")
                    }
                )
            )

print(f"Loaded {len(docs)} documents.")

db = PGVector.from_documents(
    docs,
    embed_model,
    connection_string=CONNECTION_STRING,
    collection_name="my_collection"
)

print("Ingest to pgvector done.")
