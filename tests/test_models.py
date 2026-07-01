import pytest
from models.candidate import Candidate, RedrobSignals
from models.ideal_profile import IdealProfile, DimensionWeights
from models.scoring import CandidateScore, FinalRankedCandidate

def test_candidate_parses_valid_data():
    raw = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "5 years ML",
            "location": "Mumbai",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "Acme",
            "current_company_size": "51-200",
            "current_industry": "Tech"
        },
        "career_history": [{
            "company": "Acme", "title": "ML Engineer",
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": 60, "is_current": True,
            "industry": "Tech", "company_size": "51-200",
            "description": "Built models"
        }],
        "education": [],
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 60}],
        "redrob_signals": {
            "profile_completeness_score": 85.0,
            "signup_date": "2025-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {"Python": 90.0},
            "connection_count": 200,
            "endorsements_received": 50,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 35.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 75.0,
            "search_appearance_30d": 15,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 1.0,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True
        }
    }
    c = Candidate.model_validate(raw)
    assert c.candidate_id == "CAND_0000001"
    assert c.redrob_signals.github_activity_score == 75.0

def test_ideal_profile_weights_sum_to_one():
    w = DimensionWeights(skills=0.4, experience=0.4, soft_signals=0.2)
    assert abs(w.skills + w.experience + w.soft_signals - 1.0) < 1e-6

def test_ideal_profile_weights_invalid_raises():
    with pytest.raises(Exception):
        DimensionWeights(skills=0.5, experience=0.5, soft_signals=0.5)

def test_candidate_score_fields():
    cs = CandidateScore(
        candidate_id="CAND_0000001",
        skills_score=0.8, experience_score=0.7, soft_signals_score=0.6,
        semantic_composite=0.73, redrob_score=0.65, blended_score=0.70,
        is_low_quality=False
    )
    assert cs.blended_score == 0.70

def test_final_ranked_candidate_fields():
    frc = FinalRankedCandidate(
        candidate_id="CAND_0000001", rank=1, score=0.95,
        reasoning="Strong ML background with active GitHub."
    )
    assert frc.rank == 1
