from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .matching import analyse_match
from .models import CVDocument, ConfirmedProfile, JobRecord, ProjectEntry


CV_PLANNER_SYSTEM_PROMPT = """Role: You are the planning component of an evidence-grounded CV Builder.
Task: choose and order confirmed profile facts for the selected job.
Allowed evidence: only the supplied confirmed profile fact IDs and selected job record.
Rules: return IDs and exact skill labels only; never write new biographical claims; never add facts; keep missing job requirements out of the CV.
Output: strict JSON containing section_order, project_fact_ids, experience_fact_ids, skill_names and summary_focus."""


ALLOWED_SECTIONS = ["Matched Skills", "Projects", "Experience", "Education", "Achievements"]


class CVPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_order: list[str]
    project_fact_ids: list[str] = Field(default_factory=list)
    experience_fact_ids: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    summary_focus: list[str] = Field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9+#/-]{3,}", text.lower()))


def _rank_projects(projects: list[ProjectEntry], job: JobRecord) -> list[ProjectEntry]:
    job_tokens = _tokens(job.description)
    return sorted(
        projects,
        key=lambda project: len(
            job_tokens & _tokens(f"{project.title} {project.description} {' '.join(project.technologies)}")
        ),
        reverse=True,
    )


def deterministic_plan(profile: ConfirmedProfile, job: JobRecord) -> CVPlan:
    matched, _ = analyse_match(profile, job)
    ranked_projects = _rank_projects(profile.projects, job)
    if "graduate" in job.title.lower():
        order = ["Matched Skills", "Education", "Projects", "Experience", "Achievements"]
    elif any(term in job.description.lower() for term in ("three or more years", "four years", "production")):
        order = ["Matched Skills", "Experience", "Projects", "Education", "Achievements"]
    else:
        order = ["Matched Skills", "Projects", "Experience", "Education", "Achievements"]
    return CVPlan(
        section_order=order,
        project_fact_ids=[project.source_fact_id for project in ranked_projects],
        experience_fact_ids=[item.source_fact_id for item in profile.experience],
        skill_names=[item.requirement for item in matched],
        summary_focus=[item.requirement for item in matched[:4]],
    )


def _plan_schema() -> dict:
    schema = CVPlan.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}))
    return schema


def model_plan(client: Any, model: str, profile: ConfirmedProfile, job: JobRecord) -> CVPlan:
    evidence = {
        "profile": {
            "job_goal": profile.job_goal,
            "skills": profile.skills,
            "skill_evidence": [item.model_dump() for item in profile.skill_evidence],
            "projects": [
                {
                    "source_fact_id": item.source_fact_id,
                    "title": item.title,
                    "description": item.description,
                    "technologies": item.technologies,
                }
                for item in profile.projects
            ],
            "experience": [
                {
                    "source_fact_id": item.source_fact_id,
                    "role": item.role,
                    "description": item.description,
                }
                for item in profile.experience
            ],
        },
        "selected_job": job.model_dump(),
        "allowed_sections": ALLOWED_SECTIONS,
    }
    response = client.responses.create(
        model=model,
        instructions=CV_PLANNER_SYSTEM_PROMPT,
        input=json.dumps(evidence),
        text={
            "format": {
                "type": "json_schema",
                "name": "cv_plan",
                "strict": True,
                "schema": _plan_schema(),
            }
        },
    )
    return CVPlan.model_validate_json(response.output_text)


def _validated_plan(plan: CVPlan, profile: ConfirmedProfile) -> CVPlan:
    project_ids = {item.source_fact_id for item in profile.projects}
    experience_ids = {item.source_fact_id for item in profile.experience}
    allowed_skills = {item.lower(): item for item in profile.skills}
    if not set(plan.project_fact_ids).issubset(project_ids):
        raise ValueError("CV plan contained an unsupported project fact ID")
    if not set(plan.experience_fact_ids).issubset(experience_ids):
        raise ValueError("CV plan contained an unsupported experience fact ID")
    if not set(plan.section_order).issubset(ALLOWED_SECTIONS):
        raise ValueError("CV plan contained an unknown section")
    plan.skill_names = [allowed_skills[item.lower()] for item in plan.skill_names if item.lower() in allowed_skills]
    return plan


def generate_cv(
    profile: ConfirmedProfile,
    job: JobRecord,
    client: Any | None = None,
    model: str = "gpt-5.6-luna",
) -> CVDocument:
    if not profile.confirmed:
        raise ValueError("The profile must be confirmed before CV generation")
    matched, missing = analyse_match(profile, job)
    fallback_plan = deterministic_plan(profile, job)
    plan = fallback_plan
    if client is not None:
        try:
            plan = _validated_plan(model_plan(client, model, profile, job), profile)
        except Exception:
            plan = fallback_plan

    project_by_id = {item.source_fact_id: item for item in profile.projects}
    experience_by_id = {item.source_fact_id: item for item in profile.experience}
    projects = [project_by_id[item] for item in plan.project_fact_ids if item in project_by_id]
    experience = [experience_by_id[item] for item in plan.experience_fact_ids if item in experience_by_id]
    matched_labels = [item.requirement for item in matched]
    selected_skills = [skill for skill in profile.skills if skill.lower() in {x.lower() for x in matched_labels}]
    selected_skills += [skill for skill in profile.skills if skill not in selected_skills]
    focus = plan.summary_focus or matched_labels[:4]
    focus_text = ", ".join(focus) if focus else "confirmed study, project and work evidence"
    summary = (
        f"{profile.summary.rstrip('.')} Targeting {job.title}, with confirmed evidence relevant to "
        f"{focus_text}."
    ).strip()
    contact_values = [
        profile.contact.email,
        profile.contact.phone if profile.contact.phone != "Not provided" else "",
        profile.contact.location,
        profile.contact.github,
        profile.contact.linkedin,
    ]
    fact_ids = [*plan.project_fact_ids, *plan.experience_fact_ids]
    return CVDocument(
        user_id=profile.user_id,
        profile_name=profile.name,
        target_document_id=job.document_id,
        target_title=job.title,
        contact_line=" | ".join(item for item in contact_values if item),
        professional_summary=summary,
        section_order=plan.section_order,
        matched_skills=matched,
        missing_requirements=missing,
        skills=selected_skills,
        education=profile.education,
        experience=experience,
        projects=projects,
        achievements=profile.achievements,
        source_fact_ids=fact_ids,
    )


def validate_cv_grounding(cv: CVDocument, profile: ConfirmedProfile) -> list[str]:
    errors: list[str] = []
    if cv.user_id != profile.user_id:
        errors.append("CV user_id does not match the active profile")
    allowed_project_ids = {item.source_fact_id for item in profile.projects}
    allowed_experience_ids = {item.source_fact_id for item in profile.experience}
    for item in cv.projects:
        if item.source_fact_id not in allowed_project_ids:
            errors.append(f"Unsupported project fact: {item.source_fact_id}")
    for item in cv.experience:
        if item.source_fact_id not in allowed_experience_ids:
            errors.append(f"Unsupported experience fact: {item.source_fact_id}")
    unsupported_skills = set(cv.skills) - set(profile.skills)
    if unsupported_skills:
        errors.append(f"Unsupported skills: {sorted(unsupported_skills)}")
    return errors
