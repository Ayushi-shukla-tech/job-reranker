import pytest
from datetime import date
from models.candidate import RedrobSignals, SalaryRange
from models.ideal_profile import DimensionWeights
from scoring.redrob_signals import score_redrob
from scoring.composite import compute_semantic_composite


def _make_signals(**overrides) -> RedrobSignals:
    base = {
        "profile_completeness_score": 85.0,
        "signup_date": date(2025, 1, 1),
        "last_active_date": date(2026, 6, 1),
        "open_to_work_flag": True,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 4.0,
        "skill_assessment_scores": {"Python": 90.0},
        "connection_count": 200,
        "endorsements_received": 50,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": SalaryRange(min=20.0, max=35.0),
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 75.0,
        "search_appearance_30d": 15,
        "saved_by_recruiters_30d": 3,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 0.8,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return RedrobSignals(**base)


def test_score_redrob_returns_all_keys():
    signals = _make_signals()
    result = score_redrob(signals)
    for key in ["engagement", "responsiveness", "credibility", "market_signal", "technical_signal", "logistics", "redrob_composite"]:
        assert key in result
        assert 0.0 <= result[key] <= 1.0


def test_github_minus_one_treated_as_zero():
    signals_no_github = _make_signals(github_activity_score=-1.0)
    signals_with_github = _make_signals(github_activity_score=0.0)
    r1 = score_redrob(signals_no_github)
    r2 = score_redrob(signals_with_github)
    assert r1["technical_signal"] == r2["technical_signal"]


def test_offer_acceptance_minus_one_treated_as_neutral():
    signals = _make_signals(offer_acceptance_rate=-1.0, interview_completion_rate=1.0, recruiter_response_rate=0.9)
    result = score_redrob(signals)
    assert result["responsiveness"] > 0.5


def test_compute_semantic_composite_weighted():
    weights = DimensionWeights(skills=0.4, experience=0.4, soft_signals=0.2)
    score = compute_semantic_composite(0.9, 0.8, 0.6, weights)
    expected = 0.4 * 0.9 + 0.4 * 0.8 + 0.2 * 0.6
    assert abs(score - expected) < 1e-6


def test_compute_semantic_composite_clamped():
    weights = DimensionWeights(skills=0.4, experience=0.4, soft_signals=0.2)
    score = compute_semantic_composite(1.1, 1.1, 1.1, weights)
    assert score <= 1.0
