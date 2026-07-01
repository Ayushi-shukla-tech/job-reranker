from models.ideal_profile import DimensionWeights


def compute_semantic_composite(
    skills_score: float,
    experience_score: float,
    soft_score: float,
    weights: DimensionWeights,
) -> float:
    raw = (
        weights.skills * skills_score
        + weights.experience * experience_score
        + weights.soft_signals * soft_score
    )
    return round(min(1.0, max(0.0, raw)), 6)
