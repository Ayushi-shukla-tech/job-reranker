import json, pytest
from unittest.mock import MagicMock
from models.candidate import Candidate
from models.scoring import CandidateScore, FinalRankedCandidate
from models.ideal_profile import IdealProfile, DimensionWeights, ExperienceRequirements, SoftSignals
from pipeline.stage5_llm_ranker import run_stage5

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

def _make_ideal_profile():
    return IdealProfile(
        required_skills=["Python"], preferred_skills=["PyTorch"],
        experience_requirements=ExperienceRequirements(
            min_years=3, seniority="senior", domains=["ML"],
            trajectory_signals=["grew into ML lead"]
        ),
        red_flags=["no production exp"],
        soft_signals=SoftSignals(summary_traits=["self-starter"], activities_and_awards=[], certifications=[]),
        dimension_weights=DimensionWeights(skills=0.4, experience=0.4, soft_signals=0.2)
    )

def _make_shortlist(candidates: dict) -> list[CandidateScore]:
    return [
        CandidateScore(
            candidate_id=cid,
            skills_score=0.8, experience_score=0.7, soft_signals_score=0.6,
            semantic_composite=0.74, redrob_score=0.65, blended_score=0.71,
            is_low_quality=False
        )
        for cid in list(candidates.keys())[:3]
    ]

def test_run_stage5_returns_final_ranked(monkeypatch):
    candidates = _load_sample_candidates()
    shortlist = _make_shortlist(candidates)
    ideal = _make_ideal_profile()

    mock_response_data = [
        {"candidate_id": cs.candidate_id, "rank": i+1,
         "score": round(0.9 - i * 0.05, 2), "reasoning": "Strong ML fit."}
        for i, cs in enumerate(shortlist)
    ]
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(mock_response_data)
    mock_model.generate_content.return_value = mock_resp
    monkeypatch.setattr("pipeline.stage5_llm_ranker._get_model", lambda: mock_model)

    results = run_stage5(shortlist, candidates, ideal)
    assert len(results) == len(shortlist)
    for frc in results:
        assert isinstance(frc, FinalRankedCandidate)
        assert frc.reasoning
    ranks = sorted(frc.rank for frc in results)
    assert ranks == list(range(1, len(shortlist) + 1))

def test_run_stage5_scores_non_increasing(monkeypatch):
    candidates = _load_sample_candidates()
    shortlist = _make_shortlist(candidates)
    ideal = _make_ideal_profile()

    mock_response_data = [
        {"candidate_id": cs.candidate_id, "rank": i+1,
         "score": round(0.9 - i * 0.05, 2), "reasoning": "Good fit."}
        for i, cs in enumerate(shortlist)
    ]
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(mock_response_data)
    mock_model.generate_content.return_value = mock_resp
    monkeypatch.setattr("pipeline.stage5_llm_ranker._get_model", lambda: mock_model)

    results = run_stage5(shortlist, candidates, ideal)
    results_sorted = sorted(results, key=lambda x: x.rank)
    scores = [r.score for r in results_sorted]
    assert scores == sorted(scores, reverse=True)
