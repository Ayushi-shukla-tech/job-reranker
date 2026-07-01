from pydantic import BaseModel, model_validator

class ExperienceRequirements(BaseModel):
    min_years: int
    seniority: str
    domains: list[str]
    trajectory_signals: list[str]

class SoftSignals(BaseModel):
    summary_traits: list[str]
    activities_and_awards: list[str]
    certifications: list[str]

class DimensionWeights(BaseModel):
    skills: float
    experience: float
    soft_signals: float

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "DimensionWeights":
        total = self.skills + self.experience + self.soft_signals
        if abs(total - 1.0) > 0.05:
            raise ValueError(f"dimension_weights must sum to 1.0, got {total:.3f}")
        return self

class IdealProfile(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str]
    experience_requirements: ExperienceRequirements
    red_flags: list[str]
    soft_signals: SoftSignals
    dimension_weights: DimensionWeights
