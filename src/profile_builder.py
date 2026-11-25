from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .models import (
    ConfirmedProfile,
    ContactDetails,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
)


PROFILE_SYSTEM_PROMPT = """You are the profile-building component of a CV system.
Task: extract only facts explicitly stated by the user.
Allowed evidence: the current user's introduction and their follow-up answers only.
Rules: do not infer employers, dates, qualifications, seniority, achievements, skills, or contact details.
Output: valid JSON matching the provided schema. Use empty strings or arrays for missing fields.
Never mix information from different user_id values."""


KNOWN_SKILLS = [
    "Python", "Java", "SQL", "NumPy", "pandas", "scikit-learn", "PyTorch",
    "TensorFlow", "OpenCV", "Docker", "Git", "GitHub", "Linux", "Flask",
    "REST APIs", "Power BI", "Excel", "Matplotlib", "RAG", "LLMs",
    "embeddings", "FAISS", "Wireshark", "unit testing", "CI/CD",
]


@dataclass(frozen=True)
class FollowUp:
    field: str
    question: str


def missing_profile_fields(profile: ConfirmedProfile) -> list[str]:
    missing: list[str] = []
    if not profile.name.strip():
        missing.append("name")
    if not (profile.contact.email or profile.contact.phone):
        missing.append("contact")
    if not profile.education:
        missing.append("education")
    if not profile.skills:
        missing.append("skills")
    if not (profile.experience or profile.projects):
        missing.append("evidence")
    if not profile.job_goal.strip():
        missing.append("job_goal")
    if not profile.achievements_reviewed:
        missing.append("achievements")
    return missing


FOLLOW_UP_QUESTIONS = {
    "name": "What full name should appear on the CV?",
    "contact": "What email address or phone number should the CV use?",
    "education": "What qualification are you studying or have completed, and at which institution?",
    "skills": "Which technical or professional skills can you support with study, project or work evidence?",
    "evidence": "Describe one project or work experience that demonstrates your skills. What did you personally do?",
    "job_goal": "What specific role or job area are you targeting?",
    "achievements": "List any confirmed achievements, or answer ‘none’ if you have none to add.",
}


def next_follow_up(profile: ConfirmedProfile) -> FollowUp | None:
    missing = missing_profile_fields(profile)
    if not missing:
        return None
    field = missing[0]
    return FollowUp(field, FOLLOW_UP_QUESTIONS[field])


def _find_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""


def _find_phone(text: str) -> str:
    match = re.search(r"(?:\+?\d[\d ()-]{7,}\d)", text)
    return match.group(0).strip() if match else ""


def parse_introduction_offline(user_id: str, text: str) -> ConfirmedProfile:
    """Conservative fallback parser used when no API key is configured."""
    lowered = text.lower()
    name = ""
    name_match = re.search(
        r"(?i:my name is|i'm|i am)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,2})",
        text,
    )
    if name_match:
        name = name_match.group(1).strip()

    education: list[EducationEntry] = []
    if "student" in lowered or "degree" in lowered or "bachelor" in lowered or "master" in lowered:
        institution = "University of Technology Sydney" if "uts" in lowered else ""
        qualification = "Computing qualification"
        degree_match = re.search(r"((?:bachelor|master)[^.,;]+)", text, re.IGNORECASE)
        if degree_match:
            qualification = degree_match.group(1).strip()
        elif "computing science" in lowered:
            qualification = "Computing Science degree"
        education.append(EducationEntry(qualification=qualification, institution=institution))

    skills = [skill for skill in KNOWN_SKILLS if skill.lower() in lowered]
    project_sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\b(built|created|developed|implemented|project)\b", sentence, re.IGNORECASE)
    ]
    projects = [
        ProjectEntry(
            title=f"Confirmed project {index}",
            context="User introduction",
            description=sentence,
            technologies=[s for s in skills if s.lower() in sentence.lower()],
            source_fact_id=f"{user_id}:intro-project:{index}",
        )
        for index, sentence in enumerate(project_sentences, start=1)
    ]
    experience_sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\b(work|worked|intern|cadet|assistant|volunteer)\b", sentence, re.IGNORECASE)
        and sentence not in project_sentences
    ]
    experience = [
        ExperienceEntry(
            role="Role described by user",
            organisation="Organisation not specified",
            description=sentence,
            source_fact_id=f"{user_id}:intro-experience:{index}",
        )
        for index, sentence in enumerate(experience_sentences, start=1)
    ]
    goal = ""
    goal_match = re.search(
        r"(?:targeting|looking for|job goal is|interested in)\s+([^.!?]+)", text, re.IGNORECASE
    )
    if goal_match:
        goal = goal_match.group(1).strip()

    return ConfirmedProfile(
        user_id=user_id,
        name=name,
        contact=ContactDetails(email=_find_email(text), phone=_find_phone(text)),
        education=education,
        skills=skills,
        experience=experience,
        projects=projects,
        achievements_reviewed="no achievement" in lowered or "none" in lowered,
        job_goal=goal,
        summary=text.strip(),
        confirmed=False,
    )


def _strictify_schema(node: Any) -> Any:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _strictify_schema(value)
    elif isinstance(node, list):
        for value in node:
            _strictify_schema(value)
    return node


def _profile_json_schema() -> dict[str, Any]:
    schema = ConfirmedProfile.model_json_schema()
    return _strictify_schema(schema)


def parse_introduction_with_model(client: Any, model: str, user_id: str, text: str) -> ConfirmedProfile:
    response = client.responses.create(
        model=model,
        instructions=PROFILE_SYSTEM_PROMPT,
        input=f"user_id: {user_id}\nintroduction: {text}",
        text={
            "format": {
                "type": "json_schema",
                "name": "confirmed_profile_draft",
                "strict": True,
                "schema": _profile_json_schema(),
            }
        },
    )
    payload = json.loads(response.output_text)
    payload["user_id"] = user_id
    payload["confirmed"] = False
    return ConfirmedProfile.model_validate(payload)


def apply_follow_up(profile: ConfirmedProfile, field: str, answer: str) -> ConfirmedProfile:
    """Apply a user's answer to only the requested missing field."""
    answer = answer.strip()
    if not answer:
        return profile
    if field == "name":
        profile.name = answer
    elif field == "contact":
        email, phone = _find_email(answer), _find_phone(answer)
        profile.contact.email = email or (answer if "@" in answer else profile.contact.email)
        profile.contact.phone = phone or profile.contact.phone
    elif field == "education":
        parts = [part.strip() for part in answer.split(" at ", 1)]
        profile.education = [
            EducationEntry(
                qualification=parts[0],
                institution=parts[1] if len(parts) == 2 else "Institution not specified",
            )
        ]
    elif field == "skills":
        profile.skills = [item.strip() for item in re.split(r"[,;]", answer) if item.strip()]
    elif field == "evidence":
        profile.projects.append(
            ProjectEntry(
                title="Confirmed project or experience",
                context="Follow-up answer",
                description=answer,
                technologies=[s for s in profile.skills if s.lower() in answer.lower()],
                source_fact_id=f"{profile.user_id}:follow-up-evidence:1",
            )
        )
    elif field == "job_goal":
        profile.job_goal = answer
    elif field == "achievements":
        profile.achievements_reviewed = True
        if answer.lower() not in {"none", "no", "no achievements", "nothing"}:
            profile.achievements = [item.strip() for item in re.split(r"[;\n]", answer) if item.strip()]
    return profile


def profile_review_summary(profile: ConfirmedProfile) -> str:
    education = "; ".join(
        f"{item.qualification} — {item.institution}" for item in profile.education
    ) or "Not provided"
    return (
        f"Name: {profile.name or 'Not provided'}\n"
        f"Contact: {profile.contact.email or profile.contact.phone or 'Not provided'}\n"
        f"Education: {education}\n"
        f"Skills: {', '.join(profile.skills) or 'Not provided'}\n"
        f"Experience records: {len(profile.experience)}\n"
        f"Projects: {len(profile.projects)}\n"
        f"Achievements: {', '.join(profile.achievements) if profile.achievements else 'None confirmed'}\n"
        f"Job goal: {profile.job_goal or 'Not provided'}"
    )
