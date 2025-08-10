# Customer Support Knowledge RAG: Retriever Optimization

## Task Overview
You will optimize a retrieval-augmented generation (RAG) system for a customer support QA platform. The Chroma vector DB is fully set up and pre-populated with embeddings, but retrieval results are weak due to improper chunking and lack of attached metadata on support articles. Your goal is to implement text chunking with overlap, batch embedding, metadata attachment, and efficient top-k retrieval using cosine similarity. Complete the retrieval pipeline and verify improvements for a set of real-world queries.

## Your Objectives
- Implement chunking for each document (max 400 tokens per chunk with 200-token overlap)
- Attach metadata to every chunk: category, priority, date
- Batch embed these chunks and idempotently upsert them to the Chroma DB
- Configure Chroma collection for cosine similarity and reasonable ANN search settings
- Complete `rag_retrieval.py`: encode query, retrieve top-5 chunks, assemble answer context
- Spot-check retrieval and calculate recall@5 using `sample_queries.txt`

## RAG System Gaps
- Current retrieval uses single massive chunk per document, leading to irrelevant results
- Missing per-chunk metadata makes filtering and debugging harder
- Retrieval and assembly pipeline need accurate query embedding and cosine top-k logic
- Manual/automated evaluation missing (recall@5)

## Environment Status
- All infrastructure (database, base embeddings, configuration) is automated through `run.sh`
- Embedding and chunking improvements, retrieval pipeline, and context assembly logic remain to be implemented

## Database Access
- **Chroma Collection:** `support_docs`
- **Dimensions:** 384
- **Metadata Schema:** `category` (str), `priority` (str), `date` (str, ISO), `doc_id` (str), `chunk_index` (int), `tokens` (int), `position` (int)
- **Host:** `<DROPLET_IP>`
- Use Chroma Python SDK for searching and upserts

## How to Verify
- Run retrievals for the sample test queries in `sample_queries.txt` and check if context includes the correct support answer (spot-check)
- For a labeled test set, output recall@5 (was correct answer in top-5 chunks?)
- Validate chunk and metadata structure in DB using admin or Python tools

## Definitions
- **Chunking:** Splitting documents into overlapping text segments to improve retrieval granularity
- **Embedding:** Transforming text chunks into numeric vectors via Sentence-Transformers
- **Top-k retrieval:** Returning the k most similar chunks for a query using cosine similarity
- **Recall@k:** Fraction of queries where a relevant chunk is among top-k results

---
