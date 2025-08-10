from sentence_transformers import SentenceTransformer
from database_client import top_k_query

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# ---- Fill these in ----
def process_query(query_text):
    """
    Encode a user query to an embedding using the sentence transformer.
    Input: str (query)
    Output: normalized embedding (1D float list or numpy array)
    """
    # TODO: Implement query embedding logic
    pass

def retrieve_top_chunks(query_embedding, k=5):
    """
    Perform cosine-similarity top-k search in Chroma DB.
    Input: embedding
    Output: retrieved chunk dicts (with 'documents' and 'metadatas')
    """
    # TODO: Implement retrieval with database_client.top_k_query
    pass

def assemble_context(chunks):
    """
    Combine retrieved chunk texts into semantically coherent answer context.
    Input: list of chunk dicts
    Output: str (context for LLM generation)
    """
    # TODO: Implement context assembly (order and concatenate texts)
    pass

# Example end-to-end usage for spot check:
# q = 'How do I reset my password?'
# emb = process_query(q)
# res = retrieve_top_chunks(emb)
# ctx = assemble_context(res)
# print(ctx)
