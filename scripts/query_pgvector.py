import os
import psycopg2
import numpy as np
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load .env
load_dotenv()

def find_similar_documents_pgvector(query_embedding, top_k=3):
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        dbname=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        port=os.getenv("PG_PORT")
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    emb_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql = f"""
        SELECT content, source_file, chunk_id, embedding <=> '{emb_str}' AS distance
        FROM embeddings
        ORDER BY embedding <=> '{emb_str}'
        LIMIT {top_k};
    """
    cur.execute(sql)
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows
