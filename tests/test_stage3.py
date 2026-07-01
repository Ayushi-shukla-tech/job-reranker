import json, pytest
from models.ideal_profile import IdealProfile, DimensionWeights, ExperienceRequirements, SoftSignals
from models.candidate import Candidate
from pipeline.stage3_reranker import run_stage3

SAMPLE_PATH = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"

def _load_sample_candidates() -> dict:
    candidates = {}
    with open(SAMPLE_PATH) as f:
        import json as _json
        try:
            data = _json.load(f)
            for item in data:
                c = Candidate.model_validate(item)
                candidates[c.candidate_id] = c
        except Exception:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    c = Candidate.model_validate(_json.loads(line))
                    candidates[c.candidate_id] = c
    return candidates

def _make_ideal_profile() -> IdealProfile:
    return IdealProfile(
        required_skills=["Python", "Machine Learning", "PyTorch"],
        preferred_skills=["MLflow"],
        experience_requirements=ExperienceRequirements(
            min_years=3, seniority="senior",
            domains=["ML", "Data Science"],
            trajectory_signals=["grew from SWE to ML lead"]
        ),
        red_flags=["no production experience"],
        soft_signals=SoftSignals(
            summary_traits=["self-starter"],
            activities_and_awards=["Kaggle"],
            certifications=["Google ML"]
        ),
        dimension_weights=DimensionWeights(skills=0.4, experience=0.4, soft_signals=0.2)
    )

def test_run_stage3_returns_sorted_scores():
    candidates = _load_sample_candidates()
    ideal = _make_ideal_profile()
    top_ids = list(candidates.keys())[:3]

    results = run_stage3(ideal, top_ids, candidates)

    assert len(results) == 3
    for cid, sk, ex, soft, comp in results:
        assert cid.startswith("CAND_")
        assert 0.0 <= sk <= 1.0
        assert 0.0 <= ex <= 1.0
        assert 0.0 <= soft <= 1.0
        assert 0.0 <= comp <= 1.0

    composites = [comp for _, _, _, _, comp in results]
    assert composites == sorted(composites, reverse=True)
