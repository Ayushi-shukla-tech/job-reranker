import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import settings

class GemmaEncoder:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(settings.gemma_model)
        # Gemma 3 ships as a causal-LM checkpoint (backbone weights live under `model.`).
        # Loading via AutoModel would silently random-init the backbone, so load the
        # CausalLM and take its `.model` (the Gemma3TextModel that returns last_hidden_state).
        self._model = AutoModelForCausalLM.from_pretrained(settings.gemma_model).model
        self._model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (token_embeddings * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True).clip(min=1e-9)
            all_embeddings.append(mean_pooled / norms)
        return np.vstack(all_embeddings)
