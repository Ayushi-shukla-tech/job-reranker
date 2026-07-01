import json
import time
from pathlib import Path
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from config import settings
from models.candidate import Candidate
from models.scoring import CandidateScore, FinalRankedCandidate
from models.ideal_profile import IdealProfile

_PROMPT_PATH = Path("prompts/final_ranking.txt")
BATCH_SIZE = 50
_MAX_RETRIES = 6
_INTER_BATCH_SLEEP = 30  # pace requests to respect per-minute free-tier limits

def _get_model():
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(settings.gemini_pro_model)

def _generate_with_retry(model, prompt):
    """Call Gemini with exponential backoff on 429/503 (free-tier rate limits)."""
    delay = 20.0
    for attempt in range(_MAX_RETRIES):
        try:
            return model.generate_content(prompt)
        except (ResourceExhausted, ServiceUnavailable) as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = getattr(getattr(e, "retry", None), "_deadline", None) or delay
            print(f"    rate-limited ({type(e).__name__}); retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120)

def _candidate_summary(cs: CandidateScore, candidate: Candidate) -> dict:
    top_skills = [
        {"name": s.name, "proficiency": s.proficiency}
        for s in candidate.skills[:5]
    ]
    trajectory = [
        {"title": e.title, "company": e.company, "industry": e.industry}
        for e in candidate.career_history[:2]
    ]
    return {
        "candidate_id": cs.candidate_id,
        "current_title": candidate.profile.current_title,
        "years_of_experience": candidate.profile.years_of_experience,
        "headline": candidate.profile.headline,
        "top_skills": top_skills,
        "career_trajectory": trajectory,
        "scores": {
            "skills": cs.skills_score,
            "experience": cs.experience_score,
            "soft_signals": cs.soft_signals_score,
            "semantic_composite": cs.semantic_composite,
            "redrob_engagement": cs.redrob_score,
        },
        "low_quality_flag": cs.is_low_quality,
    }

def _call_llm_batch(
    batch: list[CandidateScore],
    all_candidates: dict[str, Candidate],
    ideal_profile: IdealProfile,
) -> list[dict]:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    summaries = [_candidate_summary(cs, all_candidates[cs.candidate_id]) for cs in batch]
    prompt = (
        template
        .replace("{{total}}", str(len(batch)))
        .replace("{{ideal_profile_json}}", json.dumps(ideal_profile.model_dump(), indent=2))
        .replace("{{candidates_json}}", json.dumps(summaries, indent=2))
    )
    model = _get_model()
    response = _generate_with_retry(model, prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)

def _enforce_non_increasing_scores(results: list[FinalRankedCandidate]) -> list[FinalRankedCandidate]:
    sorted_results = sorted(results, key=lambda x: x.rank)
    for i in range(1, len(sorted_results)):
        if sorted_results[i].score > sorted_results[i - 1].score:
            sorted_results[i] = sorted_results[i].model_copy(
                update={"score": sorted_results[i - 1].score}
            )
    return sorted_results

def run_stage5(
    shortlist: list[CandidateScore],
    all_candidates: dict[str, Candidate],
    ideal_profile: IdealProfile,
) -> list[FinalRankedCandidate]:
    """Batch LLM ranking → merge → enforce constraints → return top-100."""
    batches = [shortlist[i:i + BATCH_SIZE] for i in range(0, len(shortlist), BATCH_SIZE)]

    all_items = []
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(_INTER_BATCH_SLEEP)
        items = _call_llm_batch(batch, all_candidates, ideal_profile)
        all_items.extend(items)

    all_items.sort(key=lambda x: x["score"], reverse=True)

    final = []
    for i, item in enumerate(all_items[:100]):
        final.append(FinalRankedCandidate(
            candidate_id=item["candidate_id"],
            rank=i + 1,
            score=float(item["score"]),
            reasoning=item["reasoning"],
        ))

    return _enforce_non_increasing_scores(final)
