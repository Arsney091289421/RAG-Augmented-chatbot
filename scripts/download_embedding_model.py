import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
MODEL_PATH = os.getenv("MODEL_PATH", "./embedding_model")

print(f"Downloading and preparing model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

model.save(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")


