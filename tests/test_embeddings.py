import numpy as np
import pytest, tempfile, os
from embeddings.minilm_encoder import MiniLMEncoder
from embeddings.gemma_encoder import GemmaEncoder
from embeddings.faiss_index import FaissIndex

def test_minilm_encode_returns_correct_shape():
    enc = MiniLMEncoder()
    result = enc.encode(["hello world", "machine learning engineer"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32

def test_minilm_encode_single_string():
    enc = MiniLMEncoder()
    result = enc.encode(["test"])
    assert result.shape == (1, 384)

def test_gemma_encode_returns_ndarray():
    enc = GemmaEncoder()
    result = enc.encode(["Python expert 60 months; PyTorch advanced"])
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1
    assert result.dtype == np.float32

def test_gemma_encode_is_normalized():
    enc = GemmaEncoder()
    result = enc.encode(["test text"])
    norm = np.linalg.norm(result[0])
    assert abs(norm - 1.0) < 1e-5

def test_faiss_index_build_query_save_load():
    enc = MiniLMEncoder()
    texts = ["machine learning engineer", "accountant finance", "data scientist python"]
    ids = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
    embs = enc.encode(texts)

    idx = FaissIndex()
    idx.build(embs, ids)

    query = enc.encode(["deep learning researcher"])
    results = idx.query(query[0], top_k=2)
    assert len(results) == 2
    assert "CAND_0000001" in results

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.index")
        idx.save(path)
        loaded = FaissIndex.load(path)
        results2 = loaded.query(query[0], top_k=2)
        assert results == results2
