from datetime import date
from models.candidate import Candidate
from models.scoring import CandidateScore
from scoring.redrob_signals import score_redrob
from config import settings

LOW_QUALITY_COMPLETENESS = 30.0
LOW_QUALITY_INACTIVE_DAYS = 180
LOW_QUALITY_SCORE_EXCEPTION = 0.90

def _is_low_quality(candidate: Candidate, semantic_composite: float) -> bool:
    if semantic_composite >= LOW_QUALITY_SCORE_EXCEPTION:
        return False
    signals = candidate.redrob_signals
    if signals.profile_completeness_score < LOW_QUALITY_COMPLETENESS:
        return True
    days_inactive = (date.today() - signals.last_active_date).days
    if days_inactive > LOW_QUALITY_INACTIVE_DAYS:
        return True
    return False

def run_stage4(
    stage3_results: list[tuple[str, float, float, float, float]],
    all_candidates: dict[str, Candidate],
) -> list[CandidateScore]:
    """
    Blend semantic composite with Redrob signals.
    Returns top-N CandidateScore objects sorted descending by blended_score.
    """
    scores = []
    for cid, sk, ex, soft, semantic_composite in stage3_results:
        candidate = all_candidates.get(cid)
        if candidate is None:
            continue

        redrob_breakdown = score_redrob(candidate.redrob_signals)
        redrob_score = redrob_breakdown["redrob_composite"]

        blended = (
            settings.semantic_weight * semantic_composite
            + settings.redrob_weight * redrob_score
        )
        blended = round(min(1.0, max(0.0, blended)), 6)

        low_quality = _is_low_quality(candidate, semantic_composite)

        scores.append(CandidateScore(
            candidate_id=cid,
            skills_score=round(sk, 6),
            experience_score=round(ex, 6),
            soft_signals_score=round(soft, 6),
            semantic_composite=round(semantic_composite, 6),
            redrob_score=round(redrob_score, 6),
            blended_score=blended,
            is_low_quality=low_quality,
        ))

    scores.sort(key=lambda cs: cs.blended_score, reverse=True)
    return scores[:settings.final_candidates_n]
