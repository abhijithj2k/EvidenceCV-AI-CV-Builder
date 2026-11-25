from src.retrieval import LocalSemanticRetriever


EXPECTED = {
    "student_profile_01": "graduate_ai_engineer",
    "student_profile_02": "nlp_multimodal_ml_engineer",
    "student_profile_03": "recommendation_systems_ml_engineer",
    "student_profile_04": "healthcare_ml_engineer",
    "student_profile_05": "agentic_ai_engineer",
}


def test_nine_unique_job_records(jobs):
    assert len(jobs) == 9
    assert len({job.document_id for job in jobs}) == 9


def test_expected_job_is_in_top_three_for_all_profiles(jobs, profiles):
    retriever = LocalSemanticRetriever(jobs)
    for profile in profiles:
        hits = retriever.search(profile.retrieval_query(), top_k=3)
        ids = [hit.document_id for hit in hits]
        assert EXPECTED[profile.user_id] in ids
        assert len(ids) == len(set(ids)) == 3
        assert all(hit.supporting_excerpt for hit in hits)
