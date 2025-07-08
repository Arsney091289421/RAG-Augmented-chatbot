import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

MODEL_PATH = os.getenv("MODEL_PATH", "./embedding_model")

DST_PATH = os.getenv("DST_PATH", "./docs_clean")
chunk_json = os.path.join(DST_PATH, "chunk", "sklearn_chunks.json")
output_path = os.path.join(DST_PATH, "embeddings", "sklearn_embeddings.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(chunk_json, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedder = HuggingFaceEmbeddings(model_name=MODEL_PATH)

all_embeds = []
for chunk in tqdm(chunks, desc="Embedding"):
    text = chunk["content"]
    emb = embedder.embed_query(text)
    chunk["embedding"] = emb
    all_embeds.append(chunk)

with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(all_embeds, out_f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_embeds)} embeddings to {output_path}")
