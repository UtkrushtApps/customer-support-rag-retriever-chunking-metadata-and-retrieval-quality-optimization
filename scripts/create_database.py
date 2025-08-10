import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma.db"))
collections = client.list_collections()
if not any([c.name == "support_docs" for c in collections]):
    client.create_collection(name="support_docs", metadata={"hnsw:space": "cosine"})
    print("[INFO] Chroma collection 'support_docs' created.")
else:
    print("[INFO] Chroma collection 'support_docs' already exists.")
