import os
import json
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load .env
load_dotenv()

DB_NAME = os.getenv("PG_DB")
DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")
DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT")

EMBED_FILES = [
    "./docs_clean/embeddings/sklearn_embeddings.json",
    "./docs_clean/embeddings/transformers_embeddings.json"
]

# connect database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

total_records = 0

for embed_file in EMBED_FILES:
    if not os.path.exists(embed_file):
        print(f"⚠️ File not found: {embed_file}")
        continue

    with open(embed_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        emb = item["embedding"]
        emb_str = "[" + ",".join(map(str, emb)) + "]"
        source_file = item.get("source_file", os.path.basename(embed_file))
        chunk_id = item.get("chunk_id", "")
        content = item["content"]
        records.append((source_file, chunk_id, content, emb_str))

    query = """
    INSERT INTO embeddings (source_file, chunk_id, content, embedding)
    VALUES %s
    """
    execute_values(cur, query, records)
    conn.commit()

    print(f"Inserted {len(records)} embeddings from {embed_file}")
    total_records += len(records)

cur.close()
conn.close()
print(f"Inserted total {total_records} embeddings into pgvector.")


