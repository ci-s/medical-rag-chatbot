import faiss
from typing import Any

from core.embedding import embed_chunks


class FaissService:
    def __init__(self):
        self.index = None
        self.chunks = None

    def create_index(self, chunks: list[str]):
        if chunks is None:
            raise ValueError("Image representations cannot be None")

        embeddings = embed_chunks([chunk.text for chunk in chunks], task_type="search_document")

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # TODO: parametrize
        index.add(embeddings)
        print("Index created with {} chunks".format(index.ntotal))
        self.index = index
        self.chunks = chunks

    def search_index(self, query, k):
        if isinstance(query, str):
            query_embedding = self._encode_chunks([query])
        else:
            query_embedding = query

        D, I = self.index.search(query_embedding, k)

        return D[0], [self.chunks[idx] for idx in I[0]]


def retrieve(query: str, faiss_service: FaissService, top_k: int = 3):
    query_embedding = embed_chunks(query, task_type="search_query")

    similarity, retrieved_documents = faiss_service.search_index(query_embedding, top_k)

    print("Retrieved documents:")
    for i, retrieved_document in enumerate(retrieved_documents):
        print("*" * 20, f"Retrieval {i+1} with similarity score {similarity[i]}", "*" * 20)
        print(retrieved_document)

    return retrieved_documents
