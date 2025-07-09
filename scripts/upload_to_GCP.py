import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "your-default-bucket-name")
DST_PATH = os.getenv("DST_PATH", "./docs_clean")

client = storage.Client()

def upload_folder(local_folder, bucket_folder):
    bucket = client.bucket(BUCKET_NAME)
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_file_path, local_folder)
            blob_path = os.path.join(bucket_folder, rel_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded: {local_file_path} → gs://{BUCKET_NAME}/{blob_path}")

if __name__ == "__main__":
    chunk_folder = os.path.join(DST_PATH, "chunk")
    upload_folder(chunk_folder, "rag/chunks")

    embeddings_folder = os.path.join(DST_PATH, "embeddings")
    upload_folder(embeddings_folder, "rag/embeddings")

    print("All files uploaded successfully!")
