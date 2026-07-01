import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings

class MiniLMEncoder:
    def __init__(self):
        self._model = SentenceTransformer(settings.minilm_model)

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
