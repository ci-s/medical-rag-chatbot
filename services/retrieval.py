import faiss
from typing import Any

from core.embedding import embed_chunks


class FaissService:
    def __init__(self, encoder_model=None):
        if encoder_model is not None:
            self.encoder_model = encoder_model
        else:
            self.encoder_model = None
        self.index = None

    def _encode_sentences(self, sentences):
        return self.encoder_model.encode(sentences, normalize_embeddings=True)

    def create_index(self, text: list[Any]):
        if text is None:
            raise ValueError("Image representations cannot be None")
        if isinstance(text, list) and isinstance(text[0], str):
            embeddings = self._encode_sentences(text)
        else:
            embeddings = text
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # TODO: parametrize
        index.add(embeddings)
        print("Index created with {} chunks".format(index.ntotal))
        self.index = index

    def search_index(self, query, k):
        if isinstance(query, str):
            query_embedding = self._encode_sentences([query])
        else:
            query_embedding = query

        print("Returning top k results...")
        D, I = self.index.search(query_embedding, k)
        return D[0], I[0]


def retrieve(chunks: list[str], query: str, faiss_service: FaissService, top_k: int = 3):
    query_embedding = embed_chunks(query, task_type="search_query")

    similarity, i = faiss_service.search_index(query_embedding, top_k)

    retrieved_documents = [chunks[idx] for idx in i]

    print("Retrieved documents:")
    for i, retrieved_document in enumerate(retrieved_documents):
        print("*" * 20, f"Retrieval {i+1} with similarity score {similarity[i]}", "*" * 20)
        print(retrieved_document)

    return retrieved_documents
