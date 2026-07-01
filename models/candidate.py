from datetime import date
from typing import Optional
from pydantic import BaseModel, Field

class CareerEntry(BaseModel):
    company: str
    title: str
    start_date: date
    end_date: Optional[date] = None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: Optional[str] = None

class Skill(BaseModel):
    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int] = None

class Certification(BaseModel):
    name: str
    issuer: str
    year: int

class Language(BaseModel):
    language: str
    proficiency: str

class SalaryRange(BaseModel):
    min: float
    max: float

class RedrobSignals(BaseModel):
    profile_completeness_score: float
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

class Profile(BaseModel):
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

class Candidate(BaseModel):
    candidate_id: str
    profile: Profile
    career_history: list[CareerEntry]
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals
