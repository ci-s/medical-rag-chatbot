import faiss
from typing import Any


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
        print("Index created with {} sentences".format(index.ntotal))
        self.index = index

    def search_index(self, query, k):
        if isinstance(query, str):
            query_embedding = self._encode_sentences([query])
        else:
            query_embedding = query

        print("Returning top k results...")
        D, I = self.index.search(query_embedding, k)
        return D[0], I[0]
