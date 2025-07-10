import os
import json
import shutil
import time
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

# Config
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "your-default-bucket-name")
DST_PATH = os.getenv("DST_PATH", "./docs_clean")

client = storage.Client()

def convert_json_to_jsonl(json_path, jsonl_path):
    """Convert a JSON array file to JSONL format"""
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(jsonl_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"Converted to JSONL: {jsonl_path}")

def backup_existing(blob_path, history_prefix):
    """Back up an existing file to a history folder with a timestamp"""
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if blob.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_blob = bucket.blob(f"{history_prefix}/{timestamp}_{os.path.basename(blob_path)}")
        bucket.copy_blob(blob, bucket, backup_blob.name)
        print(f"Backed up existing blob to: gs://{BUCKET_NAME}/{backup_blob.name}")

def upload_file(local_file, bucket_path, history_prefix):
    """Upload a file after backing up the existing version"""
    backup_existing(bucket_path, history_prefix)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded: {local_file} → gs://{BUCKET_NAME}/{bucket_path}")

def upload_folder(local_folder, bucket_folder, history_folder):
    """Upload an entire local folder"""
    bucket = client.bucket(BUCKET_NAME)
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_file_path, local_folder)
            blob_path = os.path.join(bucket_folder, rel_path)
            history_prefix = os.path.join(history_folder, os.path.dirname(rel_path))

            upload_file(local_file_path, blob_path, history_prefix)

if __name__ == "__main__":
    # Step 1: Convert embeddings JSON to JSONL
    embeddings_folder = os.path.join(DST_PATH, "embeddings")
    json_files = [f for f in os.listdir(embeddings_folder) if f.endswith(".json")]

    for json_file in json_files:
        json_path = os.path.join(embeddings_folder, json_file)
        jsonl_path = json_path.replace(".json", ".jsonl")
        convert_json_to_jsonl(json_path, jsonl_path)

    # Step 2: Upload chunk folder
    chunk_folder = os.path.join(DST_PATH, "chunk")
    upload_folder(chunk_folder, "rag/chunks", "rag/chunks/history")

    # Step 3: Upload embeddings folder (including JSON and JSONL)
    upload_folder(embeddings_folder, "rag/embeddings", "rag/embeddings/history")

    print("All files uploaded successfully!")
