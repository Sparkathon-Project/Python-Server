def find_similar_image(faiss_index, query_embedding, top_k, faiss_ids):
    """
    Finds the most similar embeddings in a FAISS index to a given query embedding.

    Args:
        faiss_index: The pre-loaded FAISS index containing the embeddings.
        query_embedding: The embedding of the query item,
        top_k (int): The number of top similar embeddings to retrieve.
        faiss_ids (list or numpy.ndarray): A list or array mapping the internal FAISS index IDs to your actual original IDs (e.g., product IDs).

    Returns:
        list: A list of original IDs corresponding to the top_k most similar embeddings found in the FAISS index.
    """
    _, indices = faiss_index.search(query_embedding, top_k)
    similar_ids = [faiss_ids[i] for i in indices[0]]
    return similar_ids

def find_using_text(faiss_index, query_embeddings, top_k, faiss_ids):
    D, I = faiss_index.search(query_embeddings, top_k)
    ids = []
    seen = set()
    for row in I:
        for idx in row:
            pid = faiss_ids[idx]
            if pid not in seen:
                ids.append(pid)
                seen.add(pid)
    return ids