import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DOCS_PATH = os.getenv("DOCS_PATH", "docs/source/en")
DST_PATH = os.path.join(os.getenv("DST_PATH", "./docs_clean"), "transformers")

REPO_OWNER = "huggingface"
REPO_NAME = "transformers"
EXCLUDE_DIRS = {"reference", "internal"}

headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def github_api(path):
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    return requests.get(url, headers=headers)

def collect_md_files(path):
    """Recursively collect all .md file download URLs and their relative paths."""
    files = []
    resp = github_api(path)
    if resp.status_code != 200:
        print(f"Failed to list {path}: HTTP {resp.status_code}")
        return files
    for item in resp.json():
        name = item["name"]
        if item["type"] == "dir":
            if name in EXCLUDE_DIRS:
                continue
            files += collect_md_files(item["path"])
        elif item["type"] == "file" and name.endswith(".md"):
            files.append((item["download_url"], item["path"]))
    return files

def save_md_file(url, rel_path, dst_root):
    dst_file = os.path.join(dst_root, rel_path)
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        with open(dst_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return True
    else:
        print(f"Download failed: {url}")
        return False

if __name__ == "__main__":
    print(f"Collecting Markdown files from {REPO_OWNER}/{REPO_NAME}:{DOCS_PATH} ...")
    files = collect_md_files(DOCS_PATH)
    print(f"Found {len(files)} .md files to download (excluding reference/internal directories).")

    for url, rel_path in tqdm(files, desc="Downloading", unit="file"):
        save_md_file(url, rel_path, DST_PATH)
    print("Download completed.")
