import math
from datetime import date
from models.candidate import RedrobSignals


def _recency_score(last_active: date) -> float:
    days_ago = (date.today() - last_active).days
    if days_ago <= 7:
        return 1.0
    if days_ago <= 30:
        return 0.8
    if days_ago <= 90:
        return 0.5
    if days_ago <= 180:
        return 0.2
    return 0.0


def _log_normalize(value: float, scale: float = 10.0) -> float:
    return min(1.0, math.log1p(value) / math.log1p(scale))


def score_redrob(signals: RedrobSignals) -> dict[str, float]:
    # Engagement
    open_score = 1.0 if signals.open_to_work_flag else 0.3
    recency = _recency_score(signals.last_active_date)
    activity = min(1.0, signals.applications_submitted_30d / 5.0)
    engagement = 0.4 * open_score + 0.4 * recency + 0.2 * activity

    # Responsiveness — offer_acceptance_rate=-1 → 0.5 neutral
    response_time_score = max(0.0, 1.0 - signals.avg_response_time_hours / 168.0)
    responsiveness = (
        0.4 * signals.recruiter_response_rate
        + 0.3 * response_time_score
        + 0.3 * signals.interview_completion_rate
    )

    # Credibility
    verification_score = (
        (1.0 if signals.verified_email else 0.0)
        + (1.0 if signals.verified_phone else 0.0)
        + (1.0 if signals.linkedin_connected else 0.0)
    ) / 3.0
    completeness_norm = signals.profile_completeness_score / 100.0
    credibility = 0.5 * verification_score + 0.5 * completeness_norm

    # Market signal
    saved_norm = _log_normalize(signals.saved_by_recruiters_30d, scale=10.0)
    search_norm = _log_normalize(signals.search_appearance_30d, scale=50.0)
    market_signal = 0.6 * saved_norm + 0.4 * search_norm

    # Technical signal — github_activity_score=-1 → 0
    github = max(0.0, signals.github_activity_score) / 100.0
    assessment_avg = (
        sum(signals.skill_assessment_scores.values()) / len(signals.skill_assessment_scores)
        if signals.skill_assessment_scores else 0.0
    ) / 100.0
    technical_signal = 0.5 * github + 0.5 * assessment_avg

    # Logistics
    notice_score = 1.0 if signals.notice_period_days <= 30 else (
        0.7 if signals.notice_period_days <= 60 else (
        0.4 if signals.notice_period_days <= 90 else 0.1
    ))
    relocate_bonus = 0.2 if signals.willing_to_relocate else 0.0
    logistics = min(1.0, notice_score * 0.8 + relocate_bonus)

    redrob_composite = (
        0.25 * engagement
        + 0.25 * responsiveness
        + 0.20 * credibility
        + 0.10 * market_signal
        + 0.10 * technical_signal
        + 0.10 * logistics
    )

    return {
        "engagement": round(engagement, 4),
        "responsiveness": round(responsiveness, 4),
        "credibility": round(credibility, 4),
        "market_signal": round(market_signal, 4),
        "technical_signal": round(technical_signal, 4),
        "logistics": round(logistics, 4),
        "redrob_composite": round(redrob_composite, 4),
    }
