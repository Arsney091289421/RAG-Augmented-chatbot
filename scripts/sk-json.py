import os
import json
from bs4 import BeautifulSoup

def parse_html_files(folder_path, output_json, chunk_size=1200):
    all_chunks = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')

                
                body = soup.find('body')
                if body:
                    text = body.get_text(separator='\n').strip()

                    
                    paragraphs = text.split('\n')
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
    html_folder = "/Users/daniel/Documents/AIDI/projects/RAG/scikit-learn-docs"  # 你的 HTML 文件目录
    output_file = "/Users/daniel/Documents/AIDI/projects/RAG/sklearn_html_clean_chunks.json"

    parse_html_files(html_folder, output_file)
    print(f"Output saved at: {output_file}")
