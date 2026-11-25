from __future__ import annotations

import re

from .models import ConfirmedProfile, JobRecord, MatchItem, MissingItem


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "SQL": ("sql",),
    "NumPy": ("numpy",),
    "pandas": ("pandas",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
    "OpenCV": ("opencv",),
    "computer vision": ("computer vision",),
    "object detection": ("object detection",),
    "video analytics": ("video analytics",),
    "NLP": ("nlp", "natural language processing"),
    "transformers": ("transformer architectures", "transformers"),
    "LLMs": ("llm", "large language model"),
    "RAG": ("rag", "retrieval-augmented generation"),
    "AI agents": ("ai agents", "ai-agent", "agentic ai", "agentic workflows"),
    "embeddings": ("embeddings",),
    "vector databases": ("vector databases", "vector database"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes",),
    "cloud platforms": ("cloud platform", "cloud infrastructure", "cloud ai"),
    "CI/CD": ("ci/cd",),
    "Git": ("git",),
    "REST APIs": ("rest", "apis", "api"),
    "Linux": ("linux",),
    "data visualisation": ("data visualisation", "data visualization"),
    "statistics": ("statistics",),
    "data analysis": ("data analysis", "data analytics"),
    "testing": ("testing", "unit testing"),
    "monitoring": ("monitoring", "logging", "alerting"),
    "MLOps": ("mlops",),
    "Spark": ("spark",),
    "Kafka": ("kafka",),
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _sentence_for(job: JobRecord, aliases: tuple[str, ...]) -> str:
    for sentence in _sentences(job.description):
        lowered = sentence.lower()
        if any(alias in lowered for alias in aliases):
            return sentence
    return job.description[:260]


def _profile_text(profile: ConfirmedProfile) -> str:
    parts = [
        *profile.skills,
        *(e.description for e in profile.experience),
        *(p.description for p in profile.projects),
        *(technology for p in profile.projects for technology in p.technologies),
        *(item.source for item in profile.skill_evidence),
        profile.summary,
    ]
    return " ".join(parts).lower()


def analyse_match(
    profile: ConfirmedProfile, job: JobRecord, maximum_missing: int = 8
) -> tuple[list[MatchItem], list[MissingItem]]:
    profile_text = _profile_text(profile)
    job_text = job.description.lower()
    evidence_by_skill = {item.skill.lower(): item.source for item in profile.skill_evidence}
    matched: list[MatchItem] = []
    missing: list[MissingItem] = []
    for label, aliases in SKILL_ALIASES.items():
        if not any(alias in job_text for alias in aliases):
            continue
        present = any(alias in profile_text for alias in aliases)
        if present:
            evidence = evidence_by_skill.get(label.lower())
            if not evidence:
                evidence = next(
                    (
                        item.source
                        for item in profile.skill_evidence
                        if item.skill.lower() in profile_text and item.skill.lower() in job_text
                    ),
                    "Confirmed profile skills, projects or experience",
                )
            matched.append(
                MatchItem(
                    requirement=label,
                    profile_evidence=evidence,
                    job_evidence=_sentence_for(job, aliases),
                )
            )
        else:
            missing.append(MissingItem(requirement=label, job_evidence=_sentence_for(job, aliases)))

    for sentence in _sentences(job.description):
        lower = sentence.lower()
        years = re.search(r"(?:at least\s+)?(?:two|three|four|five|\d+)(?:\s+to\s+(?:two|three|four|five|\d+))?\s+or more years", lower)
        if years and "year" not in profile_text:
            missing.insert(0, MissingItem(requirement="specified professional experience", job_evidence=sentence))
            break
    return matched, missing[:maximum_missing]
