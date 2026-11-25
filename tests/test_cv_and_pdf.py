from pypdf import PdfReader

from src.cv_generator import generate_cv, validate_cv_grounding
from src.pdf_export import cv_pdf_bytes


def test_cv_uses_only_confirmed_profile_facts(jobs, profiles):
    profile = profiles[1]
    job = next(item for item in jobs if item.document_id == "rag_agent_engineer")
    cv = generate_cv(profile, job)
    assert validate_cv_grounding(cv, profile) == []
    assert set(cv.skills).issubset(set(profile.skills))
    assert {item.source_fact_id for item in cv.projects}.issubset(
        {item.source_fact_id for item in profile.projects}
    )


def test_prompt_improvement_rejects_unsupported_skill(jobs, profiles):
    profile = profiles[1]
    job = next(item for item in jobs if item.document_id == "nlp_multimodal_ml_engineer")
    unconstrained_draft = generate_cv(profile, job)
    unconstrained_draft.skills.append("Kubernetes")
    initial_errors = validate_cv_grounding(unconstrained_draft, profile)
    improved_draft = generate_cv(profile, job)
    improved_errors = validate_cv_grounding(improved_draft, profile)
    assert any("Kubernetes" in error for error in initial_errors)
    assert improved_errors == []


def test_two_jobs_change_cv_emphasis_but_not_profile(jobs, profiles):
    profile = profiles[1]
    snapshot = profile.model_dump_json()
    nlp = next(item for item in jobs if item.document_id == "nlp_multimodal_ml_engineer")
    rag = next(item for item in jobs if item.document_id == "rag_agent_engineer")
    cv_nlp = generate_cv(profile, nlp)
    cv_rag = generate_cv(profile, rag)
    assert cv_nlp.target_document_id != cv_rag.target_document_id
    assert cv_nlp.professional_summary != cv_rag.professional_summary
    assert [item.title for item in cv_nlp.projects] != [item.title for item in cv_rag.projects]
    assert profile.model_dump_json() == snapshot


def test_pdf_is_readable_and_records_target(jobs, profiles):
    profile = profiles[4]
    job = next(item for item in jobs if item.document_id == "agentic_ai_engineer")
    cv = generate_cv(profile, job)
    data = cv_pdf_bytes(cv)
    assert data.startswith(b"%PDF")
    reader = PdfReader(__import__("io").BytesIO(data))
    assert len(reader.pages) >= 1
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert profile.name in extracted
    assert job.document_id in extracted
