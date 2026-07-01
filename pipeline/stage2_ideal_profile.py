import hashlib, json
from pathlib import Path
import google.generativeai as genai
from config import settings
from models.ideal_profile import IdealProfile

CACHE_PATH = "cache/ideal_profile_cache.json"
_PROMPT_PATH = Path("prompts/ideal_candidate.txt")

def _jd_hash(jd_text: str) -> str:
    return hashlib.md5(jd_text.encode()).hexdigest()

def _get_model():
    genai.configure(api_key=settings.google_api_key)
    return genai.GenerativeModel(settings.gemini_flash_model)

def _load_prompt(jd_text: str) -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{{jd_text}}", jd_text)

def _call_llm(jd_text: str) -> IdealProfile:
    model = _get_model()
    prompt = _load_prompt(jd_text)
    response = model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return IdealProfile.model_validate(json.loads(raw))

def run_stage2(jd_text: str) -> IdealProfile:
    """Return IdealProfile from cache if JD unchanged, otherwise call Gemini Flash."""
    jd_key = _jd_hash(jd_text)
    cache_path = Path(CACHE_PATH)

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("jd_hash") == jd_key:
            return IdealProfile.model_validate(cached["profile"])

    profile = _call_llm(jd_text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"jd_hash": jd_key, "profile": profile.model_dump()}, indent=2),
        encoding="utf-8",
    )
    return profile
