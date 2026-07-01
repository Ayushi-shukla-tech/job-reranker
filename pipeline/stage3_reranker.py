import numpy as np
from models.ideal_profile import IdealProfile
from models.candidate import Candidate
from embeddings.gemma_encoder import GemmaEncoder
from scoring.composite import compute_semantic_composite
from data.candidate_texts import build_skills_text, build_experience_text, build_soft_signals_text

def _build_query_texts(ideal: IdealProfile) -> tuple[str, str, str]:
    skills_query = ", ".join(ideal.required_skills + ideal.preferred_skills)
    exp_req = ideal.experience_requirements
    experience_query = (
        f"{exp_req.seniority} level, {exp_req.min_years}+ years, "
        f"domains: {', '.join(exp_req.domains)}. "
        f"Trajectory: {'; '.join(exp_req.trajectory_signals)}"
    )
    soft = ideal.soft_signals
    soft_query = (
        f"Traits: {', '.join(soft.summary_traits)}. "
        f"Activities: {', '.join(soft.activities_and_awards)}. "
        f"Certs: {', '.join(soft.certifications)}"
    )
    return skills_query, experience_query, soft_query

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def run_stage3(
    ideal_profile: IdealProfile,
    top_candidate_ids: list[str],
    all_candidates: dict[str, Candidate],
) -> list[tuple[str, float, float, float, float]]:
    """
    Returns list of (candidate_id, skills_score, experience_score, soft_score, semantic_composite)
    sorted descending by semantic_composite.
    """
    encoder = GemmaEncoder()
    skills_q, exp_q, soft_q = _build_query_texts(ideal_profile)

    query_embs = encoder.encode([skills_q, exp_q, soft_q])
    q_skills, q_exp, q_soft = query_embs[0], query_embs[1], query_embs[2]

    candidates = [all_candidates[cid] for cid in top_candidate_ids if cid in all_candidates]

    skills_texts = [build_skills_text(c) for c in candidates]
    exp_texts = [build_experience_text(c) for c in candidates]
    soft_texts = [build_soft_signals_text(c) for c in candidates]

    skills_embs = encoder.encode(skills_texts)
    exp_embs = encoder.encode(exp_texts)
    soft_embs = encoder.encode(soft_texts)

    results = []
    for i, candidate in enumerate(candidates):
        sk = max(0.0, _cosine(q_skills, skills_embs[i]))
        ex = max(0.0, _cosine(q_exp, exp_embs[i]))
        soft = max(0.0, _cosine(q_soft, soft_embs[i]))
        composite = compute_semantic_composite(sk, ex, soft, ideal_profile.dimension_weights)
        results.append((candidate.candidate_id, sk, ex, soft, composite))

    results.sort(key=lambda x: x[4], reverse=True)
    return results
