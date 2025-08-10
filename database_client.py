import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="data/chroma.db"))
collection = client.get_collection("support_docs")

def top_k_query(query_embedding, k=5, category=None):
    filter = {"category": category} if category else None
    results = collection.query(query_embeddings=[query_embedding], n_results=k, where=filter)
    return results
