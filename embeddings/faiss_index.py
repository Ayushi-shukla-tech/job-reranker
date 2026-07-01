import pickle
import numpy as np
import faiss

class FaissIndex:
    def __init__(self):
        self._index: faiss.Index | None = None
        self._ids: list[str] = []

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._ids = list(ids)

    def query(self, query_embedding: np.ndarray, top_k: int) -> list[str]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q = query_embedding.reshape(1, -1).astype(np.float32)
        _, indices = self._index.search(q, top_k)
        return [self._ids[i] for i in indices[0] if i < len(self._ids)]

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".ids", "wb") as f:
            pickle.dump(self._ids, f)

    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        obj = cls()
        obj._index = faiss.read_index(path + ".faiss")
        with open(path + ".ids", "rb") as f:
            obj._ids = pickle.load(f)
        return obj
