import pytest
from unittest.mock import patch, PropertyMock
import numpy as np
from pipeline.stage1_prefilter import run_stage1

JD_TEXT = "We are hiring a Senior ML Engineer with PyTorch, NLP, and production ML system experience."

def test_run_stage1_returns_list_of_candidate_ids(tmp_path, monkeypatch):
    from embeddings.minilm_encoder import MiniLMEncoder
    from embeddings.faiss_index import FaissIndex
    from pathlib import Path
    from config import settings

    cache_dir = tmp_path / "cache" / "embeddings_cache"
    cache_dir.mkdir(parents=True)

    enc = MiniLMEncoder()
    texts = ["ML engineer PyTorch NLP", "accountant finance", "data scientist python"]
    ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
    embs = enc.encode(texts)

    idx = FaissIndex()
    idx.build(embs, ids)
    idx.save(str(cache_dir / "minilm.index"))

    # Patch the property on the Settings class to return cache_dir
    monkeypatch.setattr(type(settings), "embeddings_cache_dir", PropertyMock(return_value=cache_dir))
    monkeypatch.setattr(settings, "prefilter_top_n", 2)

    results = run_stage1(JD_TEXT)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(cid.startswith("CAND_") for cid in results)
    assert "CAND_0000001" in results
