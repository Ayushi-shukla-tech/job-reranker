from pydantic import BaseModel

class CandidateScore(BaseModel):
    candidate_id: str
    skills_score: float
    experience_score: float
    soft_signals_score: float
    semantic_composite: float
    redrob_score: float
    blended_score: float
    is_low_quality: bool = False

class FinalRankedCandidate(BaseModel):
    candidate_id: str
    rank: int
    score: float
    reasoning: str
