import json, pytest
from models.candidate import Candidate
from data.loader import stream_candidates
from data.candidate_texts import (
    build_minilm_text, build_skills_text,
    build_experience_text, build_soft_signals_text
)

SAMPLE_PATH = "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json"

def test_stream_candidates_yields_candidates():
    count = 0
    for c in stream_candidates(SAMPLE_PATH):
        assert isinstance(c, Candidate)
        assert c.candidate_id.startswith("CAND_")
        count += 1
    assert count > 0

def _sample_candidate():
    with open(SAMPLE_PATH) as f:
        content = f.read().strip()
        if content.startswith('['):
            # JSON array format
            data = json.loads(content)
            return Candidate.model_validate(data[0])
        else:
            # NDJSON format
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return Candidate.model_validate(json.loads(lines[0]))

def test_build_minilm_text_contains_headline():
    c = _sample_candidate()
    text = build_minilm_text(c)
    assert c.profile.headline in text

def test_build_skills_text_contains_skill_names():
    c = _sample_candidate()
    text = build_skills_text(c)
    for skill in c.skills[:3]:
        assert skill.name in text

def test_build_experience_text_contains_titles():
    c = _sample_candidate()
    text = build_experience_text(c)
    assert c.career_history[0].title in text

def test_build_soft_signals_text_contains_summary():
    c = _sample_candidate()
    text = build_soft_signals_text(c)
    assert c.profile.summary[:50] in text
