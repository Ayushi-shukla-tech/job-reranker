from config import settings
from embeddings.minilm_encoder import MiniLMEncoder
from embeddings.faiss_index import FaissIndex

def run_stage1(jd_text: str) -> list[str]:
    """Embed JD with MiniLM, query FAISS index, return top-N candidate_ids."""
    encoder = MiniLMEncoder()
    index_path = str(settings.embeddings_cache_dir / "minilm.index")
    index = FaissIndex.load(index_path)
    jd_embedding = encoder.encode([jd_text])[0]
    top_ids = index.query(jd_embedding, top_k=settings.prefilter_top_n)
    return top_ids
