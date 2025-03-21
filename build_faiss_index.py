import faiss
import numpy as np
import json
import os


def build_faiss_index(embedding_file, output_index_file):
    # Load embeddings
    embeddings = np.load(embedding_file)

    # Get the dimension of embeddings
    dimension = embeddings.shape[1]

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)  # L2 distance

    # Add embeddings to index
    index.add(embeddings)

    # Save index
    faiss.write_index(index, output_index_file)
    print(f"FAISS index saved to {output_index_file}")


if __name__ == "__main__":
    # Build index for sklearn
    build_faiss_index(
        embedding_file="/Users/daniel/Documents/AIDI/projects/RAG/embeddings/sklearn_embeddings.npy",
        output_index_file="/Users/daniel/Documents/AIDI/projects/RAG/embeddings/sklearn_index.faiss"
    )

    # Build index for Huggingface transformers
    build_faiss_index(
        embedding_file="/Users/daniel/Documents/AIDI/projects/RAG/embeddings/hf_embeddings.npy",
        output_index_file="/Users/daniel/Documents/AIDI/projects/RAG/embeddings/hf_index.faiss"
    )
