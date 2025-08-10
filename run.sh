#!/bin/bash
set -e

echo "[INFO] Launching Chroma vector DB via docker-compose..."
docker-compose up -d
sleep 4
echo "[INFO] Creating collection and base DB setup..."
python scripts/create_database.py
echo "[INFO] Chunking documents, generating embeddings, and attaching metadata..."
python scripts/process_documents.py
echo "[INFO] Verifying document ingestion and chunk distribution..."
python scripts/verify_setup.py
echo "[SUCCESS] Chroma DB ready for semantic RAG retrieval."
