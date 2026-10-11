"""Structured records exchanged and produced by vasync agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SyncMessage:
    sender_id: str
    receiver_id: str
    source_step: int
    summary: str
    content: str
    sender_role_description: str = ""
    sender_role_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncMessage":
        return cls(
            sender_id=str(data["sender_id"]),
            receiver_id=str(data["receiver_id"]),
            source_step=int(data["source_step"]),
            summary=str(data["summary"]),
            content=str(data["content"]),
            sender_role_description=str(data.get("sender_role_description", "")),
            sender_role_prompt=str(data.get("sender_role_prompt", "")),
        )


@dataclass(slots=True)
class AgentNode:
    agent_id: str
    step_index: int
    reasoning_summary: str
    candidate_answer: str | None
    open_issue: str
    next_action: str
    sync_summary: str
    sync_content: str
    is_final: bool
    confidence: float = 0.0
    context_entries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_llm_payload(
        cls,
        payload: dict[str, Any],
        *,
        agent_id: str,
        step_index: int,
    ) -> "AgentNode":
        return cls(
            agent_id=agent_id,
            step_index=step_index,
            reasoning_summary=_coerce_text(payload.get("reasoning_summary")),
            candidate_answer=_coerce_optional_text(payload.get("candidate_answer")),
            confidence=_coerce_confidence(payload.get("confidence")),
            open_issue=_coerce_text(payload.get("open_issue")),
            next_action=_coerce_text(payload.get("next_action")),
            sync_summary=_coerce_text(payload.get("sync_summary")),
            sync_content=_coerce_text(payload.get("sync_content")),
            is_final=bool(payload.get("is_final", False)),
            context_entries=_coerce_context_entries(payload.get("context_entries")),
        )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(str(item) for item in value)
    return str(value)


def _coerce_optional_text(value: Any) -> str | None:
    text = _coerce_text(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(confidence, 0.0), 1.0)


def _coerce_context_entries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
