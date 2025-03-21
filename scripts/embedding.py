import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_json_path, output_embed_path, output_texts_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # Load data from JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['content'] for item in data]

    # Load the embedding model
    model = SentenceTransformer(model_name)

    # Generate embeddings in batches
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True)

    # Save the generated embeddings as a NumPy array
    np.save(output_embed_path, embeddings)
    
    # Save the corresponding text blocks for later reference
    with open(output_texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"Embeddings generated and saved to: {output_embed_path} and {output_texts_path}")


if __name__ == "__main__":
    # Generate embeddings for the sklearn documentation
    generate_embeddings(
        input_json_path="/Users/daniel/Documents/AIDI/projects/RAG/sklearn_html_clean_chunks.json",
        output_embed_path="/Users/daniel/Documents/AIDI/projects/RAG/sklearn_embeddings.npy",
        output_texts_path="/Users/daniel/Documents/AIDI/projects/RAG/sklearn_texts.json"
    )

    # Generate embeddings for the Huggingface transformers documentation
    generate_embeddings(
        input_json_path="/Users/daniel/Documents/AIDI/projects/RAG/hf_transformers_clean_chunks.json",
        output_embed_path="/Users/daniel/Documents/AIDI/projects/RAG/hf_embeddings.npy",
        output_texts_path="/Users/daniel/Documents/AIDI/projects/RAG/hf_texts.json"
    )