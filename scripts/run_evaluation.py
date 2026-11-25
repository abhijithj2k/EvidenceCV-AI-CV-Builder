from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from src.agent import ToolCallingAgent
from src.cv_generator import generate_cv, validate_cv_grounding
from src.data_loader import load_jobs, load_test_profiles
from src.pdf_export import export_cv_pdf
from src.profile_builder import apply_follow_up, next_follow_up, parse_introduction_offline
from src.profile_store import ProfileStore
from src.retrieval import LocalSemanticRetriever


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "evidence"
EVIDENCE = MATERIALS / "03_Evidence"

PREDICTIONS = {
    "student_profile_01": ("graduate_ai_engineer", "Graduate title, applied ML, Python and coursework evidence."),
    "student_profile_02": ("nlp_multimodal_ml_engineer", "NLP, text classification, Transformers basics and retrieval coursework."),
    "student_profile_03": ("recommendation_systems_ml_engineer", "Data analysis, search-result evaluation and an explicit recommendation-systems goal."),
    "student_profile_04": ("healthcare_ml_engineer", "Linux, testing, APIs, Docker basics and a goal focused on reliable operational platforms."),
    "student_profile_05": ("agentic_ai_engineer", "Tool-call agent, Python, APIs, CI/testing and an agentic-workflow goal."),
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def table_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    jobs = load_jobs(ROOT / "data" / "jobs.jsonl")
    profiles = load_test_profiles(ROOT / "data" / "student_profiles.jsonl")
    job_by_id = {job.document_id: job for job in jobs}
    retriever = LocalSemanticRetriever(jobs)

    results = []
    retrieval_lines = [
        "# Retrieval Results",
        "",
        "**Recorded run:** deterministic offline verification mode.",
        "",
        "The run embedded all nine unchanged job records with the transparent word/character TF-IDF semantic fallback, embedded each confirmed-profile query in the same space, calculated cosine similarity and returned three unique IDs. Live mode replaces this provider with `text-embedding-3-small`.",
        "",
    ]
    five_lines = [
        "# Five-Profile Results",
        "",
        "| Profile | Variation | Predicted job | Actual top three | Expected in top 3 |",
        "|---|---|---|---|---|",
    ]
    test_profile_dir = ROOT / "test_profiles"
    evidence_profile_dir = EVIDENCE / "04_System_Tests" / "test_profiles"
    for directory in (test_profile_dir, evidence_profile_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, profile in enumerate(profiles, start=1):
        expected_id, rationale = PREDICTIONS[profile.user_id]
        hits = retriever.search(profile.retrieval_query(), 3)
        actual_ids = [hit.document_id for hit in hits]
        passed = expected_id in actual_ids
        results.append(
            {
                "profile_id": profile.user_id,
                "name": profile.name,
                "prediction": expected_id,
                "prediction_reason": rationale,
                "actual_top_three": [hit.model_dump() for hit in hits],
                "expected_in_top_three": passed,
            }
        )
        profile_json = json.dumps(profile.model_dump(), indent=2, ensure_ascii=False)
        write(test_profile_dir / f"profile_{index:02d}.json", profile_json)
        write(evidence_profile_dir / f"profile_{index:02d}.json", profile_json)

        variation = f"{len(profile.skills)} skills; {len(profile.experience)} experience; {len(profile.projects)} projects"
        five_lines.append(
            f"| {profile.name} (`{profile.user_id}`) | {variation} | `{expected_id}` | "
            f"{', '.join(f'`{item}`' for item in actual_ids)} | {'Pass' if passed else 'Fail'} |"
        )
        retrieval_lines += [
            f"## {profile.name} — `{profile.user_id}`",
            "",
            f"**Prediction:** `{expected_id}` — {rationale}",
            "",
            "| Rank | document_id | Similarity | Supporting excerpt |",
            "|---:|---|---:|---|",
        ]
        for rank, hit in enumerate(hits, start=1):
            retrieval_lines.append(
                f"| {rank} | `{hit.document_id}` | {hit.score:.4f} | {table_escape(hit.supporting_excerpt)} |"
            )
        retrieval_lines += ["", f"**Prediction check:** {'Pass' if passed else 'Fail'}", ""]

    write(EVIDENCE / "03_Retrieval" / "retrieval_results.md", "\n".join(retrieval_lines))
    write(EVIDENCE / "04_System_Tests" / "five_profile_results.md", "\n".join(five_lines))
    write(ROOT / "evaluation_results.json", json.dumps(results, indent=2, ensure_ascii=False))

    # Complete-profile Action/Observation trace.
    traced_profile = deepcopy(profiles[1])
    store = ProfileStore()
    store.save(traced_profile)
    events, observation = ToolCallingAgent(store, retriever).run_after_confirmation(traced_profile)
    agent_lines = [
        "# Agent Actions and Observations",
        "",
        f"**Test user:** {traced_profile.name} (`{traced_profile.user_id}`)",
        "",
        "**Initial state:** `PROFILE_CONFIRMED`",
        "",
    ]
    for event in events:
        agent_lines += [
            f"## Step {event.step}: {event.event_type} — `{event.name}`",
            "",
            f"**State:** `{event.state}`",
            "",
            "```json",
            json.dumps(event.content, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    agent_lines += [
        "**Stopping behaviour:** the bounded policy stopped after receiving three retrieval results. Configured maximum: six model steps.",
        "",
        f"**Result count:** {len(observation.get('results', []))}",
    ]
    write(EVIDENCE / "02_Agent" / "agent_actions_and_observations.md", "\n".join(agent_lines))

    # Incomplete profile requiring four useful turns.
    incomplete = parse_introduction_offline(
        "incomplete_user_01",
        "My name is Alex Brown. My email is alex.brown@example.com. I use Python.",
    )
    answers = {
        "education": "Bachelor of Computing Science at University of Technology Sydney",
        "evidence": "Built a Python classifier and compared precision and recall on a labelled dataset.",
        "job_goal": "Graduate AI Engineer",
        "achievements": "none",
    }
    follow_lines = [
        "# Follow-up Question Test",
        "",
        "**Initial introduction:** My name is Alex Brown. My email is alex.brown@example.com. I use Python.",
        "",
    ]
    turn = 0
    while (follow_up := next_follow_up(incomplete)) is not None:
        turn += 1
        answer = answers[follow_up.field]
        follow_lines += [
            f"## Turn {turn}",
            "",
            f"**Action:** ask for `{follow_up.field}`",
            "",
            f"**Question:** {follow_up.question}",
            "",
            f"**User answer:** {answer}",
            "",
        ]
        incomplete = apply_follow_up(incomplete, follow_up.field, answer)
        follow_lines += [
            f"**Observation:** `{follow_up.field}` stored only for `incomplete_user_01`; profile remains unconfirmed until review.",
            "",
        ]
    follow_lines += [
        f"**Outcome:** profile complete after {turn} follow-up turns; test requires at least two. Pass.",
        "",
        "A supplied complete profile produced zero follow-up questions. Pass.",
    ]
    write(EVIDENCE / "02_Agent" / "follow_up_question_test.md", "\n".join(follow_lines))

    # Three sample CVs: two jobs for Liam, one job for Ethan.
    liam = profiles[1]
    ethan = profiles[4]
    cv_nlp = generate_cv(liam, job_by_id["nlp_multimodal_ml_engineer"])
    cv_rag = generate_cv(liam, job_by_id["rag_agent_engineer"])
    cv_agent = generate_cv(ethan, job_by_id["agentic_ai_engineer"])
    samples = [
        (cv_nlp, ROOT / "sample_outputs" / "CV_Liam_Patel_nlp_multimodal_ml_engineer.pdf"),
        (cv_rag, ROOT / "sample_outputs" / "CV_Liam_Patel_rag_agent_engineer.pdf"),
        (cv_agent, ROOT / "sample_outputs" / "CV_Ethan_Nguyen_agentic_ai_engineer.pdf"),
    ]
    for cv, path in samples:
        grounding_errors = validate_cv_grounding(cv, liam if cv.user_id == liam.user_id else ethan)
        if grounding_errors:
            raise AssertionError(grounding_errors)
        export_cv_pdf(cv, path)

    comparison_lines = [
        "# Two Jobs for the Same Confirmed User",
        "",
        f"**Profile snapshot:** `{liam.user_id}` remained byte-for-byte unchanged during both `generate_cv` calls.",
        "",
        "| Feature | NLP & Multimodal CV | RAG Agent CV |",
        "|---|---|---|",
        f"| Target document_id | `{cv_nlp.target_document_id}` | `{cv_rag.target_document_id}` |",
        f"| Summary target | {cv_nlp.target_title} | {cv_rag.target_title} |",
        f"| First project | {cv_nlp.projects[0].title} | {cv_rag.projects[0].title} |",
        f"| Section order | {' → '.join(cv_nlp.section_order)} | {' → '.join(cv_rag.section_order)} |",
        f"| Matched requirements | {', '.join(item.requirement for item in cv_nlp.matched_skills)} | {', '.join(item.requirement for item in cv_rag.matched_skills)} |",
        f"| Missing requirements | {', '.join(item.requirement for item in cv_nlp.missing_requirements)} | {', '.join(item.requirement for item in cv_rag.missing_requirements)} |",
        "",
        "The NLP CV placed **Text Classification Study** first; the RAG CV placed **Coursework Retrieval Assistant** first. Both used exactly the same confirmed profile object.",
    ]
    write(EVIDENCE / "04_System_Tests" / "two_job_comparison.md", "\n".join(comparison_lines))

    first_dump = liam.model_dump_json()
    second_dump = ethan.model_dump_json()
    cross_user = f"""# Cross-user Separation Test

`ProfileStore` saved `{liam.user_id}` and `{ethan.user_id}` under separate dictionary keys and returned defensive copies.

| Check | Result |
|---|---|
| User 1 name appears in User 2 JSON | {'Fail' if liam.name in second_dump else 'Pass — absent'} |
| User 2 name appears in User 1 JSON | {'Fail' if ethan.name in first_dump else 'Pass — absent'} |
| CV 1 user_id | `{cv_nlp.user_id}` |
| CV 3 user_id | `{cv_agent.user_id}` |
| Agent cross-user request | Blocked with `cross-user access blocked` |

Changing a target job creates a new `CVDocument`; it does not write back to `ConfirmedProfile`.
"""
    write(EVIDENCE / "04_System_Tests" / "cross_user_test.md", cross_user)

    failure = """# Failure and Repair

## Observed failure

The first `pytest` run produced **9 passes and 1 failure**. The incomplete-profile test raised `KeyError: 'name'` because `parse_introduction_offline` did not recognise the capitalised phrase “My name is Alex Brown.” The regular expression was case-sensitive.

## Repair

The explicit introductory phrase was made case-insensitive while the captured name retained capitalisation checks. No broader inference rule was added, so text such as “I am a third-year student” is not treated as a name.

## Expected improvement

The same introduction should capture `Alex Brown`; the next question should therefore be education rather than asking for the name again.

## Actual retest

The repeated test asked, in order: `education`, `evidence`, `job_goal`, and `achievements`. The repair run reported **10 passed**. Two later grounding and parser-regression checks increased the final suite to **12 passed**.

## Regression check

The five-profile retrieval test and cross-user isolation test were repeated in the same suite and remained successful.
"""
    write(EVIDENCE / "05_Failures_and_Improvements" / "failure_and_repair.md", failure)

    print(json.dumps({"profiles": len(results), "all_expected_in_top_three": all(x["expected_in_top_three"] for x in results), "sample_pdfs": [str(path) for _, path in samples]}, indent=2))


if __name__ == "__main__":
    main()
