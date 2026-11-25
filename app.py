from __future__ import annotations

import os
import json
from copy import deepcopy
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agent import ToolCallingAgent
from src.cv_generator import generate_cv, validate_cv_grounding
from src.data_loader import load_introductions, load_jobs, load_test_profiles
from src.models import ConfirmedProfile
from src.pdf_export import cv_pdf_bytes
from src.profile_builder import (
    apply_follow_up,
    missing_profile_fields,
    next_follow_up,
    parse_introduction_offline,
    parse_introduction_with_model,
    profile_review_summary,
)
from src.profile_store import ProfileStore
from src.retrieval import LocalSemanticRetriever, build_retriever


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="EvidenceCV", page_icon="◎", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; color: #17212b; }
      .stApp p, .stApp label, .stApp li,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-baseweb="tab"] { color: #17212b; }
      .stApp input, .stApp textarea {
        background: #ffffff !important;
        color: #17212b !important;
        -webkit-text-fill-color: #17212b !important;
      }
      .block-container { max-width: 1180px; padding-top: 1.8rem; }
      .hero { padding: 1.4rem 1.6rem; background: linear-gradient(120deg,#17324d,#176d73);
              border-radius: 18px; color: white; margin-bottom: 1rem; }
      .hero h1 { margin: 0; font-size: 2.05rem; color: #ffffff !important; }
      .hero p { margin: .35rem 0 0; color: #d8edf0 !important; }
      .evidence { background:white; border:1px solid #dce5eb; border-radius:12px;
                  padding:.8rem 1rem; margin:.4rem 0; }
      div[data-testid="stMetric"] { background:white; border:1px solid #dce5eb; padding:.6rem;
                                    border-radius:12px; }
    </style>
    <div class="hero"><h1>EvidenceCV</h1>
    <p>Confirmed facts → evidence-ranked jobs → editable, job-tailored PDF</p></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def supplied_data():
    return (
        load_jobs(ROOT / "data" / "jobs.jsonl"),
        load_test_profiles(ROOT / "data" / "student_profiles.jsonl"),
        load_introductions(ROOT / "data" / "student_introductions.jsonl"),
    )


def api_client():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    return OpenAI()


def get_store() -> ProfileStore:
    if "profile_store" not in st.session_state:
        st.session_state.profile_store = ProfileStore()
    return st.session_state.profile_store


def get_retriever(jobs, client):
    cache_key = "openai" if client is not None else "offline"
    if st.session_state.get("retriever_mode") != cache_key:
        try:
            st.session_state.retriever = build_retriever(
                jobs,
                client=client,
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            st.session_state.retriever_mode = cache_key
        except Exception as exc:
            st.warning(f"Online embedding setup failed; using deterministic fallback. Details: {exc}")
            st.session_state.retriever = LocalSemanticRetriever(jobs)
            st.session_state.retriever_mode = "offline"
    return st.session_state.retriever


def save_active(profile: ConfirmedProfile) -> None:
    get_store().save(profile)
    st.session_state.active_user_id = profile.user_id
    st.session_state.pop(f"agent:{profile.user_id}", None)


jobs, test_profiles, introductions = supplied_data()
job_by_id = {job.document_id: job for job in jobs}
profile_by_id = {profile.user_id: profile for profile in test_profiles}
client = api_client()

with st.sidebar:
    st.subheader("Application status")
    if client:
        st.success("Enhanced generation mode is ready")
    else:
        st.info("Local demonstration mode is ready")
    st.caption("Job records: 9 unique document IDs")
    stored = get_store().list_user_ids()
    st.caption("Session users: " + (", ".join(stored) if stored else "none"))
    st.divider()
    st.markdown("**Privacy**")
    st.caption("Profiles remain in this browser session and are keyed by user ID. Credentials stay in .env and are never written to output files.")

profile_tab, jobs_tab, cv_tab = st.tabs(["1 · Confirm profile", "2 · Retrieve jobs", "3 · Tailor and export"])

with profile_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Start from an introduction")
        source_mode = st.radio(
            "Profile source",
            ["Supplied test profile", "Custom user introduction"],
            horizontal=True,
        )
        if source_mode == "Supplied test profile":
            selected_demo = st.selectbox(
                "Test user",
                list(profile_by_id),
                format_func=lambda item: f"{profile_by_id[item].name} · {item}",
            )
            st.text_area("Natural-language introduction", introductions[selected_demo], height=120, disabled=True)
            if st.button("Build profile from supplied evidence", type="primary"):
                profile = deepcopy(profile_by_id[selected_demo])
                profile.confirmed = False
                save_active(profile)
                st.rerun()
        else:
            custom_user_id = st.text_input("Unique user ID", placeholder="e.g. user_alex_01")
            introduction = st.text_area(
                "Introduction",
                placeholder=(
                    "My name is Alex Smith. I study ... My email is ... I built ... "
                    "My skills include ... I am targeting ..."
                ),
                height=150,
            )
            if st.button("Build draft profile", type="primary", disabled=not (custom_user_id and introduction)):
                try:
                    if client:
                        profile = parse_introduction_with_model(
                            client,
                            os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
                            custom_user_id,
                            introduction,
                        )
                    else:
                        profile = parse_introduction_offline(custom_user_id, introduction)
                    save_active(profile)
                    st.rerun()
                except Exception as exc:
                    st.error(f"The profile draft could not be created: {exc}")

    active_user_id = st.session_state.get("active_user_id", "")
    profile = get_store().get(active_user_id) if active_user_id else None
    with right:
        st.subheader("Review and confirm")
        if profile is None:
            st.info("Build a profile to continue.")
        else:
            missing = missing_profile_fields(profile)
            c1, c2, c3 = st.columns(3)
            c1.metric("Skills", len(profile.skills))
            c2.metric("Projects", len(profile.projects))
            c3.metric("Missing fields", len(missing))
            st.code(profile_review_summary(profile), language=None)
            follow_up = next_follow_up(profile)
            if follow_up:
                st.warning("The draft is incomplete. Answer one useful question at a time.")
                answer = st.text_input(follow_up.question, key=f"follow:{profile.user_id}:{follow_up.field}")
                if st.button("Save answer and continue", disabled=not answer):
                    save_active(apply_follow_up(profile, follow_up.field, answer))
                    st.rerun()
            elif not profile.confirmed:
                accepted = st.checkbox("I reviewed this summary and confirm these facts are accurate.")
                if st.button("Confirm reusable profile", type="primary", disabled=not accepted):
                    profile.confirmed = True
                    get_store().save(profile)
                    st.success("Profile confirmed. It will remain stable when the target job changes.")
                    st.rerun()
            else:
                st.success("Confirmed profile ready for retrieval.")

with jobs_tab:
    active_user_id = st.session_state.get("active_user_id", "")
    profile = get_store().get(active_user_id) if active_user_id else None
    if profile is None or not profile.confirmed:
        st.info("Complete and confirm a profile first.")
    else:
        retriever = get_retriever(jobs, client)
        agent_key = f"agent:{profile.user_id}"
        if st.button("Run tool-calling agent and retrieve top 3", type="primary"):
            agent = ToolCallingAgent(
                get_store(),
                retriever,
                client=client,
                model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            )
            try:
                events, observation = agent.run_after_confirmation(profile)
                st.session_state[agent_key] = {
                    "events": [event.model_dump() for event in events],
                    "observation": observation,
                }
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")
        run = st.session_state.get(agent_key)
        if run:
            observation = run["observation"]
            results = observation.get("results", [])
            st.caption(f"Retrieval provider: {observation.get('provider', 'not available')}")
            for rank, result in enumerate(results, start=1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {result['title']}** · `{result['document_id']}`")
                    st.progress(min(1.0, max(0.0, float(result["score"]))))
                    st.caption(f"Similarity: {result['score']:.4f}")
                    st.markdown(f"> {result['supporting_excerpt']}")
            with st.expander("Action → Observation trace"):
                for event in run["events"]:
                    st.markdown(f"**Step {event['step']} · {event['event_type']} · `{event['name']}`** — state `{event['state']}`")
                    st.json(event["content"])
            st.download_button(
                "Download agent trace JSON",
                data=json.dumps(run, indent=2),
                file_name=f"agent_trace_{profile.user_id}.json",
                mime="application/json",
            )

with cv_tab:
    active_user_id = st.session_state.get("active_user_id", "")
    profile = get_store().get(active_user_id) if active_user_id else None
    run = st.session_state.get(f"agent:{active_user_id}") if active_user_id else None
    results = run["observation"].get("results", []) if run else []
    if profile is None or not profile.confirmed or not results:
        st.info("Confirm a profile and retrieve jobs before generating a CV.")
    else:
        target_id = st.radio(
            "Selected target job",
            [item["document_id"] for item in results],
            format_func=lambda item: f"{job_by_id[item].title} · {item}",
        )
        cv_key = f"cv:{profile.user_id}:{target_id}"
        if st.button("Generate tailored CV", type="primary"):
            st.session_state[cv_key] = generate_cv(
                profile,
                job_by_id[target_id],
                client=client,
                model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            ).model_dump()
        if cv_key in st.session_state:
            from src.models import CVDocument

            cv = CVDocument.model_validate(st.session_state[cv_key])
            matched_tab, missing_tab = st.tabs(["Matched evidence", "Missing requirements"])
            with matched_tab:
                if cv.matched_skills:
                    st.dataframe(
                        [item.model_dump() for item in cv.matched_skills],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No explicit skill matches were detected.")
            with missing_tab:
                st.caption("These gaps are shown honestly and are not added to the CV.")
                st.dataframe(
                    [item.model_dump() for item in cv.missing_requirements],
                    width="stretch",
                    hide_index=True,
                )

            st.subheader("Editable CV")
            edited = deepcopy(cv)
            edited.professional_summary = st.text_area(
                "Professional summary",
                value=cv.professional_summary,
                height=110,
                key=f"summary:{cv_key}",
            )
            edited.skills = st.multiselect(
                "Skills to display (confirmed-profile skills only)",
                options=profile.skills,
                default=[skill for skill in cv.skills if skill in profile.skills],
                key=f"skills:{cv_key}",
            )
            edited.section_order = st.multiselect(
                "Section order",
                options=["Matched Skills", "Projects", "Experience", "Education", "Achievements"],
                default=cv.section_order,
                key=f"sections:{cv_key}",
            )
            st.caption("Project and experience statements below are locked to confirmed source facts. The summary and selection remain editable.")
            for project in edited.projects:
                st.markdown(f"**{project.title}** — {project.description}")
            for item in edited.experience:
                st.markdown(f"**{item.role}** — {item.description}")

            grounding_errors = validate_cv_grounding(edited, profile)
            if grounding_errors:
                st.error("Grounding check failed: " + "; ".join(grounding_errors))
            reconfirmed = st.checkbox(
                "I checked the editable content and confirm it is factual.", key=f"confirm-cv:{cv_key}"
            )
            edited.edited_and_reconfirmed = reconfirmed
            pdf_data = cv_pdf_bytes(edited) if reconfirmed and not grounding_errors else b""
            st.download_button(
                "Download polished CV PDF",
                data=pdf_data,
                file_name=f"CV_{profile.user_id}_{target_id}.pdf",
                mime="application/pdf",
                disabled=not pdf_data,
            )
            st.caption(f"Target document_id recorded in the PDF: `{target_id}`")

st.divider()
st.caption("EvidenceCV · Assessment 1 · Confirmed information is stored separately from job-specific CV content")
