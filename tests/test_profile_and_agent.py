from copy import deepcopy

from src.agent import ToolCallingAgent
from src.profile_builder import (
    apply_follow_up,
    missing_profile_fields,
    next_follow_up,
    parse_introduction_offline,
)
from src.profile_store import ProfileStore
from src.retrieval import LocalSemanticRetriever


def test_complete_profile_needs_no_follow_up(profiles):
    assert missing_profile_fields(profiles[0]) == []
    assert next_follow_up(profiles[0]) is None


def test_incomplete_profile_uses_multiple_useful_turns():
    profile = parse_introduction_offline(
        "incomplete_user",
        "My name is Alex Brown. My email is alex.brown@example.com. I use Python.",
    )
    answers = {
        "education": "Bachelor of Computing Science at University of Technology Sydney",
        "evidence": "Built a Python classifier and compared precision and recall on a labelled dataset.",
        "job_goal": "Graduate AI Engineer",
        "achievements": "none",
    }
    asked = []
    while (follow_up := next_follow_up(profile)) is not None:
        asked.append(follow_up.field)
        profile = apply_follow_up(profile, follow_up.field, answers[follow_up.field])
    assert len(asked) >= 2
    assert asked == ["education", "evidence", "job_goal", "achievements"]
    assert missing_profile_fields(profile) == []


def test_student_description_is_not_mistaken_for_a_name():
    profile = parse_introduction_offline(
        "anonymous_student",
        "I am a third-year UTS student. I use Python and SQL.",
    )
    assert profile.name == ""
    assert next_follow_up(profile).field == "name"


def test_profile_store_blocks_cross_user_leakage(profiles):
    store = ProfileStore()
    first, second = deepcopy(profiles[0]), deepcopy(profiles[1])
    store.save(first)
    store.save(second)
    obtained = store.get(first.user_id)
    assert obtained is not None
    assert obtained.name == first.name
    assert second.name not in obtained.model_dump_json()


def test_agent_returns_actions_observations_and_three_jobs(jobs, profiles):
    store = ProfileStore()
    profile = deepcopy(profiles[1])
    store.save(profile)
    agent = ToolCallingAgent(store, LocalSemanticRetriever(jobs))
    events, observation = agent.run_after_confirmation(profile)
    assert [event.event_type for event in events] == ["Action", "Observation", "Action", "Observation", "Final"]
    assert events[0].name == "profile_tool"
    assert events[2].name == "job_retrieval_tool"
    assert observation["status"] == "ok"
    assert len(observation["results"]) == 3


def test_agent_rejects_unconfirmed_profile(jobs, profiles):
    store = ProfileStore()
    profile = deepcopy(profiles[0])
    profile.confirmed = False
    store.save(profile)
    agent = ToolCallingAgent(store, LocalSemanticRetriever(jobs))
    _, observation = agent.run_after_confirmation(profile)
    assert observation["status"] == "error"
