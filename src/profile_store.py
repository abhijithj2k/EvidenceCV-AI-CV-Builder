from __future__ import annotations

from copy import deepcopy

from .models import ConfirmedProfile


class ProfileStore:
    """Session-scoped profile storage keyed by a non-secret user identifier."""

    def __init__(self) -> None:
        self._profiles: dict[str, ConfirmedProfile] = {}

    def save(self, profile: ConfirmedProfile) -> ConfirmedProfile:
        if not profile.user_id.strip():
            raise ValueError("user_id is required")
        self._profiles[profile.user_id] = deepcopy(profile)
        return deepcopy(profile)

    def get(self, user_id: str) -> ConfirmedProfile | None:
        profile = self._profiles.get(user_id)
        return deepcopy(profile) if profile else None

    def confirm(self, user_id: str) -> ConfirmedProfile:
        profile = self.get(user_id)
        if profile is None:
            raise KeyError(f"Unknown user_id: {user_id}")
        profile.confirmed = True
        self.save(profile)
        return profile

    def list_user_ids(self) -> list[str]:
        return sorted(self._profiles)

    def clear(self) -> None:
        self._profiles.clear()
