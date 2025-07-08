import json
import faiss


def load_FAISS():
    """Loads the FAISS Index and map.json for similarity search."""
    try:
        faiss_index = faiss.read_index("./embeddings.index")
        with open("./clip_id_map.json", "r") as f:
            faiss_ids = json.load(f)
        return faiss_index, faiss_ids
    except Exception as e:
        raise RuntimeError("Could not load FAISS store. Error: {e}")

def find_similar_embeddings(faiss_index, query_embedding, top_k, faiss_ids):
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