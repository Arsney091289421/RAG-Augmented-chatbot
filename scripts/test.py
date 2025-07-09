import psycopg2
import numpy as np
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

def find_similar_documents_pgvector(query_embedding, top_k=3):
    conn = psycopg2.connect(
        host="localhost",
        dbname="ragdb",
        user="daniel",
        password="",
        port=5432
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

if __name__ == "__main__":
    # 测试 query
    test_query = "How to fine-tune a model using transformers?"

    # 加载 embedding 模型
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 转 embedding
    query_emb = model.encode(test_query)

    # 查询 pgvector
    results = find_similar_documents_pgvector(query_emb, top_k=3)

    # 打印结果
    for i, r in enumerate(results):
        print(f"\nResult {i + 1}")
        print("Source File:", r["source_file"])
        print("Chunk ID:", r["chunk_id"])
        print("Content snippet:", r["content"][:200], "...")
        print("Distance:", r["distance"])
