from __future__ import annotations

import json
from typing import Any

from .models import AgentEvent, ConfirmedProfile
from .profile_builder import missing_profile_fields
from .profile_store import ProfileStore
from .retrieval import JobRetriever


AGENT_SYSTEM_PROMPT = """Role: You are a bounded workflow agent for an evidence-based CV Builder.
Task: after the user has confirmed a profile, inspect that profile and retrieve exactly three relevant jobs.
Allowed evidence: the current state, the named user's confirmed profile returned by profile_tool, and job records returned by job_retrieval_tool.
Required process: call profile_tool with operation=get first. If and only if that Observation says the profile is confirmed and complete, call job_retrieval_tool for the same user_id with top_k=3. Never request another user's profile. Do not invent profile facts or job evidence.
Output: after successful retrieval, give a short final answer telling the application to show the three returned jobs. Stop after the result or an error."""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "profile_tool",
        "description": "Read the current user's session-scoped profile and confirmation state.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["get"]},
                "user_id": {"type": "string"},
            },
            "required": ["operation", "user_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "job_retrieval_tool",
        "description": "Retrieve the three most relevant supplied job records for a confirmed user profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "top_k": {"type": "integer", "enum": [3]},
            },
            "required": ["user_id", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class ToolCallingAgent:
    def __init__(
        self,
        store: ProfileStore,
        retriever: JobRetriever,
        client: Any | None = None,
        model: str = "gpt-5.6-luna",
        max_steps: int = 6,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.client = client
        self.model = model
        self.max_steps = max_steps

    def _profile_observation(self, requested_user_id: str, active_user_id: str) -> dict:
        if requested_user_id != active_user_id:
            return {"status": "error", "reason": "cross-user access blocked"}
        profile = self.store.get(requested_user_id)
        if profile is None:
            return {"status": "error", "reason": "profile not found"}
        return {
            "status": "ok",
            "user_id": profile.user_id,
            "name": profile.name,
            "confirmed": profile.confirmed,
            "missing_fields": missing_profile_fields(profile),
            "job_goal": profile.job_goal,
        }

    def _retrieval_observation(self, requested_user_id: str, active_user_id: str) -> dict:
        if requested_user_id != active_user_id:
            return {"status": "error", "reason": "cross-user access blocked"}
        profile = self.store.get(requested_user_id)
        if profile is None or not profile.confirmed:
            return {"status": "error", "reason": "a confirmed profile is required"}
        missing = missing_profile_fields(profile)
        if missing:
            return {"status": "error", "reason": "profile incomplete", "missing_fields": missing}
        hits = self.retriever.search(profile.retrieval_query(), top_k=3)
        return {
            "status": "ok",
            "provider": self.retriever.provider_name,
            "results": [hit.model_dump() for hit in hits],
        }

    def _execute(self, name: str, arguments: dict, active_user_id: str) -> dict:
        if name == "profile_tool":
            return self._profile_observation(arguments.get("user_id", ""), active_user_id)
        if name == "job_retrieval_tool":
            return self._retrieval_observation(arguments.get("user_id", ""), active_user_id)
        return {"status": "error", "reason": f"unknown tool: {name}"}

    def run_after_confirmation(self, profile: ConfirmedProfile) -> tuple[list[AgentEvent], dict]:
        if self.client is None:
            return self._run_deterministic(profile)
        return self._run_model(profile)

    def _run_deterministic(self, profile: ConfirmedProfile) -> tuple[list[AgentEvent], dict]:
        events: list[AgentEvent] = []
        state = "PROFILE_CONFIRMED"
        actions = [
            ("profile_tool", {"operation": "get", "user_id": profile.user_id}),
            ("job_retrieval_tool", {"user_id": profile.user_id, "top_k": 3}),
        ]
        final_observation: dict = {}
        for step, (name, arguments) in enumerate(actions, start=1):
            events.append(AgentEvent(step=step, event_type="Action", name=name, content=arguments, state=state))
            observation = self._execute(name, arguments, profile.user_id)
            state = "PROFILE_CHECKED" if name == "profile_tool" and observation.get("status") == "ok" else state
            if name == "job_retrieval_tool" and observation.get("status") == "ok":
                state = "JOBS_RETRIEVED"
            events.append(
                AgentEvent(step=step, event_type="Observation", name=name, content=observation, state=state)
            )
            final_observation = observation
            if observation.get("status") == "error":
                break
        events.append(
            AgentEvent(
                step=len(actions) + 1,
                event_type="Final",
                name="stop",
                content="Show retrieved jobs" if final_observation.get("status") == "ok" else "Stop with error",
                state=state,
            )
        )
        return events, final_observation

    def _run_model(self, profile: ConfirmedProfile) -> tuple[list[AgentEvent], dict]:
        input_items: list[Any] = [
            {
                "role": "user",
                "content": json.dumps(
                    {"state": "PROFILE_CONFIRMED", "active_user_id": profile.user_id}
                ),
            }
        ]
        events: list[AgentEvent] = []
        state = "PROFILE_CONFIRMED"
        last_observation: dict = {}
        for step in range(1, self.max_steps + 1):
            response = self.client.responses.create(
                model=self.model,
                instructions=AGENT_SYSTEM_PROMPT,
                input=input_items,
                tools=TOOL_DEFINITIONS,
                parallel_tool_calls=False,
            )
            input_items += response.output
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                events.append(
                    AgentEvent(
                        step=step,
                        event_type="Final",
                        name="stop",
                        content=response.output_text or "Workflow completed",
                        state=state,
                    )
                )
                return events, last_observation
            for call in calls:
                arguments = json.loads(call.arguments)
                events.append(
                    AgentEvent(step=step, event_type="Action", name=call.name, content=arguments, state=state)
                )
                observation = self._execute(call.name, arguments, profile.user_id)
                if call.name == "profile_tool" and observation.get("status") == "ok":
                    state = "PROFILE_CHECKED"
                if call.name == "job_retrieval_tool" and observation.get("status") == "ok":
                    state = "JOBS_RETRIEVED"
                events.append(
                    AgentEvent(
                        step=step,
                        event_type="Observation",
                        name=call.name,
                        content=observation,
                        state=state,
                    )
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(observation),
                    }
                )
                last_observation = observation
        events.append(
            AgentEvent(
                step=self.max_steps,
                event_type="Final",
                name="step_limit",
                content="Stopped at configured step limit",
                state=state,
            )
        )
        return events, last_observation
