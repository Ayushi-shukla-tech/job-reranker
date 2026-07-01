import json, pytest
from unittest.mock import patch, MagicMock
from models.ideal_profile import IdealProfile
from pipeline.stage2_ideal_profile import run_stage2

SAMPLE_JD = "Hiring a Senior ML Engineer. Must have: Python, PyTorch, 5+ years ML."

MOCK_LLM_RESPONSE = json.dumps({
    "required_skills": ["Python", "PyTorch"],
    "preferred_skills": ["MLflow"],
    "experience_requirements": {
        "min_years": 5,
        "seniority": "senior",
        "domains": ["Machine Learning"],
        "trajectory_signals": ["transitioned from SWE to ML"]
    },
    "red_flags": ["only tutorial-level projects"],
    "soft_signals": {
        "summary_traits": ["self-directed learner"],
        "activities_and_awards": ["Kaggle competitions"],
        "certifications": ["Google ML Professional"]
    },
    "dimension_weights": {"skills": 0.45, "experience": 0.40, "soft_signals": 0.15}
})

def test_run_stage2_returns_ideal_profile(tmp_path, monkeypatch):
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = MOCK_LLM_RESPONSE
    mock_model.generate_content.return_value = mock_response

    monkeypatch.setattr("pipeline.stage2_ideal_profile._get_model", lambda: mock_model)
    monkeypatch.setattr("pipeline.stage2_ideal_profile.CACHE_PATH",
                        str(tmp_path / "ideal_profile_cache.json"))

    profile = run_stage2(SAMPLE_JD)
    assert isinstance(profile, IdealProfile)
    assert "Python" in profile.required_skills
    assert profile.dimension_weights.skills == 0.45

def test_run_stage2_uses_cache(tmp_path, monkeypatch):
    cache_path = str(tmp_path / "ideal_profile_cache.json")
    import hashlib
    jd_hash = hashlib.md5(SAMPLE_JD.encode()).hexdigest()
    cached = {"jd_hash": jd_hash, "profile": json.loads(MOCK_LLM_RESPONSE)}
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    call_count = {"n": 0}
    def mock_get_model():
        m = MagicMock()
        def side_effect(*a, **kw):
            call_count["n"] += 1
            r = MagicMock(); r.text = MOCK_LLM_RESPONSE; return r
        m.generate_content.side_effect = side_effect
        return m

    monkeypatch.setattr("pipeline.stage2_ideal_profile._get_model", mock_get_model)
    monkeypatch.setattr("pipeline.stage2_ideal_profile.CACHE_PATH", cache_path)

    run_stage2(SAMPLE_JD)
    assert call_count["n"] == 0
