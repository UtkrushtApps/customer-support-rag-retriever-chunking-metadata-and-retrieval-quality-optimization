import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma.db"))
coll = client.get_collection("support_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')

DOC_PATH = 'data/documents/support_article_set.txt'
CHUNK_SIZE = 400
OVERLAP = 200


def tokenize(text):
    return text.split()

def detokenize(tokens):
    return ' '.join(tokens)

def parse_metadata(header):
    meta = {}
    parts = header.split('|')
    for part in parts:
        if ':' in part:
            k, v = part.strip().split(':', 1)
            meta[k.strip().lower()] = v.strip()
    return meta

def chunk_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    curr_meta = {}
    curr_content = ""
    doc_idx = 0
    docs = []
    for line in lines:
        if line.startswith('Category:'):
            if curr_content.strip():
                docs.append((curr_meta, curr_content.strip()))
            curr_meta = parse_metadata(line)
            curr_content = ""
        else:
            curr_content += line + '\n'
    if curr_content.strip():
        docs.append((curr_meta, curr_content.strip()))
    return docs

def chunk_text(text, chunk_size, overlap):
    pats = re.split(r'\n+', text)
    text = ' '.join(pats)
    tokens = tokenize(text)
    chunks = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start+chunk_size, len(tokens))
        chunk = detokenize(tokens[start:end])
        chunks.append((chunk, idx, start))
        if end == len(tokens): break
        start += chunk_size - overlap
        idx += 1
    return chunks

if __name__ == "__main__":
    docs = chunk_articles(DOC_PATH)
    upserts = []
    print(f"[INFO] Documents to process: {len(docs)}")
    for i, (meta, content) in enumerate(tqdm(docs)):
        doc_id = f"doc_{i}"
        chunks = chunk_text(content, CHUNK_SIZE, OVERLAP)
        for chunk, chunk_idx, pos in chunks:
            data = {
                "id": f"{doc_id}_c{chunk_idx}",
                "documents": chunk,
                "metadatas": {
                    **meta,
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "position": pos,
                    "tokens": len(tokenize(chunk))
                }
            }
            upserts.append(data)
    print(f"[INFO] Chunks for embedding: {len(upserts)}")
    batched = [upserts[i:i+64] for i in range(0, len(upserts), 64)]
    for batch in tqdm(batched):
        texts = [item["documents"] for item in batch]
        embeds = model.encode(texts, batch_size=32, normalize_embeddings=True)
        ids = [item["id"] for item in batch]
        metas = [item["metadatas"] for item in batch]
        coll.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeds,
            metadatas=metas
        )
    print("[INFO] All chunks upserted to Chroma DB")
