import os
import json

def parse_markdown_files(folder_path, output_json, chunk_size=1200):
    all_chunks = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                if not os.path.exists(file_path):
                    print(f"文件不存在，跳过: {file_path}")
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # split by paragraph
                paragraphs = content.split('\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) < chunk_size:
                        current_chunk += para.strip() + "\n"
                    else:
                        all_chunks.append({
                            "source_file": file,
                            "content": current_chunk.strip()
                        })
                        current_chunk = para.strip() + "\n"

                if current_chunk:
                    all_chunks.append({
                        "source_file": file,
                        "content": current_chunk.strip()
                    })

    with open(output_json, 'w', encoding='utf-8') as out_f:
        json.dump(all_chunks, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    md_folder = "/Users/daniel/Documents/AIDI/projects/RAG/transformers/docs/source/en"  # markdown 文件目录
    output_file = "/Users/daniel/Documents/AIDI/projects/RAG/hf_transformers_clean_chunks.json"

    parse_markdown_files(md_folder, output_file)
    print(f"Output saved at: {output_file}")
