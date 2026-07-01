from models.candidate import Candidate

def build_minilm_text(candidate: Candidate) -> str:
    """Broad text for Stage 1 pre-filter — headline + summary + top 3 roles + skill names."""
    parts = [candidate.profile.headline, candidate.profile.summary]
    for entry in candidate.career_history[:3]:
        parts.append(entry.description)
    skill_names = " ".join(s.name for s in candidate.skills)
    parts.append(skill_names)
    return " | ".join(p for p in parts if p)

def build_skills_text(candidate: Candidate) -> str:
    """Skills dimension text for Stage 3 Gemma embedding."""
    skill_parts = []
    for s in candidate.skills:
        months = f"{s.duration_months}mo" if s.duration_months else ""
        skill_parts.append(f"{s.name} ({s.proficiency}) {months}".strip())
    assessment_keys = list(candidate.redrob_signals.skill_assessment_scores.keys())
    if assessment_keys:
        skill_parts.append("Assessed: " + ", ".join(assessment_keys))
    return "; ".join(skill_parts)

def build_experience_text(candidate: Candidate) -> str:
    """Experience dimension text for Stage 3 Gemma embedding."""
    parts = [
        f"{candidate.profile.years_of_experience} years experience",
        f"Current: {candidate.profile.current_title} at {candidate.profile.current_company}",
    ]
    for entry in candidate.career_history:
        parts.append(f"{entry.title} at {entry.company} ({entry.industry}): {entry.description}")
    return " | ".join(parts)

def build_soft_signals_text(candidate: Candidate) -> str:
    """Soft signals dimension text for Stage 3 Gemma embedding."""
    parts = [candidate.profile.summary]
    if candidate.certifications:
        certs = ", ".join(c.name for c in candidate.certifications)
        parts.append(f"Certifications: {certs}")
    if candidate.languages:
        langs = ", ".join(f"{l.language} ({l.proficiency})" for l in candidate.languages)
        parts.append(f"Languages: {langs}")
    if candidate.education:
        edu_parts = []
        for e in candidate.education:
            tier = f" [{e.tier}]" if e.tier else ""
            edu_parts.append(f"{e.degree} in {e.field_of_study} from {e.institution}{tier}")
        parts.append("Education: " + "; ".join(edu_parts))
    return " | ".join(p for p in parts if p)
