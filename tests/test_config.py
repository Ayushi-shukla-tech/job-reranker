import pytest
from config import settings

def test_settings_have_required_fields():
    assert settings.gemini_flash_model
    assert settings.gemini_pro_model
    assert settings.minilm_model
    assert settings.gemma_model
    assert settings.prefilter_top_n == 2500
    assert settings.final_candidates_n == 150
    assert settings.semantic_weight == 0.65
    assert settings.redrob_weight == 0.35
    assert abs(settings.semantic_weight + settings.redrob_weight - 1.0) < 1e-6

def test_cache_dirs_exist():
    assert settings.embeddings_cache_dir.exists()
    assert settings.gemma_cache_dir.exists()
