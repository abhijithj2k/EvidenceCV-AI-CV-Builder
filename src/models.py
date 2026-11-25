from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContactDetails(BaseModel):
    email: str = ""
    phone: str = ""
    location: str = ""
    github: str = ""
    linkedin: str = ""


class EducationEntry(BaseModel):
    qualification: str
    institution: str
    status: str = ""
    details: list[str] = Field(default_factory=list)


class ExperienceEntry(BaseModel):
    role: str
    organisation: str
    description: str
    employment_type: str = ""
    source_fact_id: str = ""


class ProjectEntry(BaseModel):
    title: str
    context: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    source_fact_id: str = ""


class SkillEvidence(BaseModel):
    skill: str
    source: str


class ConfirmedProfile(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    user_id: str
    name: str = ""
    contact: ContactDetails = Field(default_factory=ContactDetails)
    education: list[EducationEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    skill_evidence: list[SkillEvidence] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    achievements_reviewed: bool = False
    job_goal: str = ""
    summary: str = ""
    confirmed: bool = False
    source_label: str = "user introduction and follow-up answers"

    def retrieval_query(self) -> str:
        """Create a job-search query only from confirmed profile fields."""
        project_text = " ".join(
            f"{p.title} {p.description} {' '.join(p.technologies)}" for p in self.projects
        )
        experience_text = " ".join(
            f"{e.role} {e.description}" for e in self.experience
        )
        return " ".join(
            part
            for part in [
                self.job_goal,
                self.job_goal,
                " ".join(self.skills),
                " ".join(self.skills),
                self.summary,
                project_text,
                experience_text,
            ]
            if part
        )


class JobRecord(BaseModel):
    document_id: str
    title: str
    location: str
    job_type: str | None = None
    salary: str | None = None
    benefits: list[str] = Field(default_factory=list)
    description: str


class RetrievalHit(BaseModel):
    document_id: str
    title: str
    score: float
    supporting_excerpt: str


class MatchItem(BaseModel):
    requirement: str
    profile_evidence: str
    job_evidence: str


class MissingItem(BaseModel):
    requirement: str
    job_evidence: str


class CVDocument(BaseModel):
    user_id: str
    profile_name: str
    target_document_id: str
    target_title: str
    contact_line: str
    professional_summary: str
    section_order: list[str]
    matched_skills: list[MatchItem] = Field(default_factory=list)
    missing_requirements: list[MissingItem] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    source_fact_ids: list[str] = Field(default_factory=list)
    edited_and_reconfirmed: bool = False


class AgentEvent(BaseModel):
    step: int
    event_type: str
    name: str
    content: dict[str, Any] | str
    state: str
