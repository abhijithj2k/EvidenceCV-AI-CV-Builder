from __future__ import annotations

import json
from pathlib import Path

from .models import (
    ConfirmedProfile,
    ContactDetails,
    EducationEntry,
    ExperienceEntry,
    JobRecord,
    ProjectEntry,
    SkillEvidence,
)


TEST_JOB_GOALS = {
    "student_profile_01": "Graduate AI Engineer working on applied machine-learning systems",
    "student_profile_02": "Machine Learning Engineer specialising in NLP and multimodal AI",
    "student_profile_03": "Machine Learning Engineer for recommendation systems and data products",
    "student_profile_04": "Machine Learning Engineer for reliable healthcare and operational platforms",
    "student_profile_05": "AI Engineer building agentic AI workflows and production applications",
}


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jobs(path: str | Path) -> list[JobRecord]:
    jobs = [JobRecord.model_validate(row) for row in load_jsonl(path)]
    ids = [job.document_id for job in jobs]
    if len(ids) != 9 or len(ids) != len(set(ids)):
        raise ValueError("jobs.jsonl must contain nine records with unique document_id values")
    return jobs


def _safe_slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "." for c in name).strip(".")


def supplied_record_to_confirmed_profile(record: dict) -> ConfirmedProfile:
    """Adapt a supplied fictional profile without changing its evidence statements."""
    user_id = record["document_id"]
    name = record["name"]
    links = record.get("links", {})
    projects = [
        ProjectEntry(
            title=item["title"],
            context=item.get("context", ""),
            description=item["description"],
            technologies=item.get("technologies", []),
            source_fact_id=f"{user_id}:project:{index}",
        )
        for index, item in enumerate(record.get("projects", []), start=1)
    ]
    experience = [
        ExperienceEntry(
            role=item["role"],
            organisation=item["organisation"],
            description=item["description"],
            employment_type=item.get("type", ""),
            source_fact_id=f"{user_id}:experience:{index}",
        )
        for index, item in enumerate(record.get("experience", []), start=1)
    ]
    return ConfirmedProfile(
        user_id=user_id,
        name=name,
        contact=ContactDetails(
            email=f"{_safe_slug(name)}@example.com",
            phone="Not provided",
            location="Sydney NSW",
            github=links.get("github", ""),
            linkedin=links.get("linkedin", ""),
        ),
        education=[
            EducationEntry(
                qualification=record["degree"],
                institution=record["institution"],
                status=f"{record.get('year', '')}; {record.get('expected_graduation', '')}",
                details=record.get("coursework_highlights", []),
            )
        ],
        skills=record.get("technical_skills", []),
        skill_evidence=[SkillEvidence.model_validate(x) for x in record.get("skill_evidence", [])],
        experience=experience,
        projects=projects,
        achievements=record.get("activities", []),
        achievements_reviewed=True,
        job_goal=TEST_JOB_GOALS[user_id],
        summary=record.get("summary", ""),
        confirmed=True,
        source_label="supplied synthetic profile plus confirmed test job goal and contact placeholder",
    )


def load_test_profiles(path: str | Path) -> list[ConfirmedProfile]:
    return [supplied_record_to_confirmed_profile(row) for row in load_jsonl(path)]


def load_introductions(path: str | Path) -> dict[str, str]:
    return {row["document_id"]: row["initial_introduction"] for row in load_jsonl(path)}
