import json, pytest
from models.candidate import Candidate
from models.scoring import CandidateScore
from pipeline.stage4_signal_blender import run_stage4

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

def test_run_stage4_returns_candidate_scores():
    candidates = _load_sample_candidates()
    ids = list(candidates.keys())[:3]
    stage3_results = [
        (ids[0], 0.85, 0.80, 0.70, 0.81),
        (ids[1], 0.70, 0.65, 0.60, 0.67),
        (ids[2], 0.50, 0.45, 0.40, 0.47),
    ]
    results = run_stage4(stage3_results, candidates)
    assert len(results) <= len(stage3_results)
    for cs in results:
        assert isinstance(cs, CandidateScore)
        assert 0.0 <= cs.blended_score <= 1.0
        assert 0.0 <= cs.redrob_score <= 1.0

def test_run_stage4_sorted_descending():
    candidates = _load_sample_candidates()
    ids = list(candidates.keys())[:3]
    stage3_results = [
        (ids[0], 0.85, 0.80, 0.70, 0.81),
        (ids[1], 0.70, 0.65, 0.60, 0.67),
        (ids[2], 0.50, 0.45, 0.40, 0.47),
    ]
    results = run_stage4(stage3_results, candidates)
    scores = [cs.blended_score for cs in results]
    assert scores == sorted(scores, reverse=True)

def test_low_quality_candidates_flagged():
    candidates = _load_sample_candidates()
    cid = list(candidates.keys())[0]
    candidates[cid].redrob_signals.profile_completeness_score = 20.0
    stage3_results = [(cid, 0.6, 0.6, 0.6, 0.6)]
    results = run_stage4(stage3_results, candidates)
    low_q = [cs for cs in results if cs.candidate_id == cid]
    assert len(low_q) == 1
    assert low_q[0].is_low_quality is True
