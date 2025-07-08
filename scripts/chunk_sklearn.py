import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

DST_PATH = os.getenv("DST_PATH", "./docs_clean")
chunk_dir = os.path.join(DST_PATH, "chunk")
os.makedirs(chunk_dir, exist_ok=True)

def chunk_all_docs(src_root, file_ext, output_json, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    for dirpath, _, files in os.walk(src_root):
        for fname in files:
            if fname.endswith(file_ext):
                fpath = os.path.join(dirpath, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = splitter.split_text(text)
                for idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "source_file": os.path.relpath(fpath, src_root),
                        "chunk_id": f"{os.path.relpath(fpath, src_root)}_{idx}",
                        "content": chunk.strip()
                    })
    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(all_chunks, out_f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_chunks)} chunks to {output_json}")

if __name__ == "__main__":
    # sklearn
    chunk_all_docs(
        src_root=os.path.join(DST_PATH, "sklearn"),
        file_ext=".rst",
        output_json=os.path.join(chunk_dir, "sklearn_chunks.json")
    )
    # transformers
    #chunk_all_docs(
        #src_root=os.path.join(DST_PATH, "transformers"),
        #file_ext=".md",
        #output_json=os.path.join(chunk_dir, "transformers_chunks.json")
    #)
