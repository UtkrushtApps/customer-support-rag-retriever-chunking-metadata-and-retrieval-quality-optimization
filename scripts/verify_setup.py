import chromadb
from chromadb.config import Settings
client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='data/chroma.db'))
coll = client.get_collection('support_docs')
count = coll.count()
print(f"[VERIFY] Chroma collection contains {count} chunks.")
samp = coll.get(limit=2)
print("[VERIFY] Sample chunk metadata:")
for meta in samp['metadatas']:
    print(meta)
