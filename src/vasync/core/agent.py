"""Base agent state and minimal interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import AgentConfig
from .llm import LLMClient
from .message import AgentNode, SyncMessage


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig, llm_client: LLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.agent_id = config.agent_id
        self.role_prompt = config.role_prompt
        self.question_text = ""
        self.dataset_name = ""
        self.choices: list[Any] = []
        self.metadata: dict[str, Any] = {}
        self.template_path = ""
        self.received_messages: list[SyncMessage] = []
        self.intent = ""
        self.open_questions: list[str] = []
        self.local_nodes: list[AgentNode] = []
        self.context_init_node: AgentNode | None = None
        self.unsynced_window: list[AgentNode] = []
        self.final_answer: str | None = None
        self.is_finished = False
        self.last_reviewed_sync_count = 0
        self.raw_responses: list[str] = []
        self.invalid_node_events: list[dict[str, Any]] = []
        self.context_store: Any = None

    def set_question(self, question_text: str) -> None:
        self.question_text = question_text

    def set_task_context(
        self,
        *,
        dataset_name: str,
        choices: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        template_path: str = "",
    ) -> None:
        self.dataset_name = dataset_name
        self.choices = list(choices or [])
        self.metadata = dict(metadata or {})
        self.template_path = template_path

    def receive_sync_message(self, message: SyncMessage) -> None:
        self.received_messages.append(message)

    def get_unsynced_window(self) -> list[AgentNode]:
        return list(self.unsynced_window)

    def clear_unsynced_window(self) -> None:
        self.unsynced_window.clear()

    def export_state(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role_prompt": self.role_prompt,
            "question_text": self.question_text,
            "dataset_name": self.dataset_name,
            "choices": list(self.choices),
            "metadata": dict(self.metadata),
            "template_path": self.template_path,
            "received_messages": [message.to_dict() for message in self.received_messages],
            "intent": self.intent,
            "open_questions": list(self.open_questions),
            "local_nodes": [node.to_dict() for node in self.local_nodes],
            "context_init_node": (
                self.context_init_node.to_dict()
                if self.context_init_node is not None
                else None
            ),
            "unsynced_window": [node.to_dict() for node in self.unsynced_window],
            "final_answer": self.final_answer,
            "is_finished": self.is_finished,
            "last_reviewed_sync_count": self.last_reviewed_sync_count,
            "invalid_node_events": list(self.invalid_node_events),
            "context_store": (
                self.context_store.to_dict()
                if self.context_store is not None and hasattr(self.context_store, "to_dict")
                else None
            ),
        }

    @abstractmethod
    def step(self) -> AgentNode:
        """Advance the agent by one reasoning node."""

    @abstractmethod
    def run(self, question_text: str | None = None, max_steps: int | None = None) -> list[AgentNode]:
        """Run the agent until final or max_steps."""
