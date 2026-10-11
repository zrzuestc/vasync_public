"""Cos-Gate value prediction based on agent-window divergence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vasync.core.message import AgentNode

from .pairwise import Encoder, PairwiseScores, compute_mean_pairwise_cosine


@dataclass(slots=True)
class AgentWindowInput:
    agent_id: str
    role_prompt: str
    unsynced_window: Sequence[AgentNode]


@dataclass(slots=True)
class CosGateResult:
    mean_cosine: float
    divergence_score: float
    math_divergence: float
    cosine_matrix: np.ndarray
    pairwise_scores: PairwiseScores
    rationale_count: int
    agent_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_cosine": self.mean_cosine,
            "divergence_score": self.divergence_score,
            "math_divergence": self.math_divergence,
            "cosine_matrix": self.cosine_matrix.tolist(),
            "pairwise_scores": {
                f"{i},{j}": score for (i, j), score in self.pairwise_scores.items()
            },
            "rationale_count": self.rationale_count,
            "agent_ids": list(self.agent_ids) if self.agent_ids is not None else None,
        }


class CosGateValuePredictor:
    """Lightweight predictor that scores divergence but never syncs messages."""

    def __init__(self, encoder: Encoder, *, eps: float = 1e-12) -> None:
        self.encoder = encoder
        self.eps = eps

    def predict(
        self,
        rationales: Sequence[str],
        *,
        agent_ids: Sequence[str] | None = None,
    ) -> CosGateResult:
        return compute_cos_gate_score(
            rationales,
            self.encoder,
            eps=self.eps,
            agent_ids=agent_ids,
        )

    def predict_from_windows(self, windows: Sequence[AgentWindowInput]) -> CosGateResult:
        rationales = [
            build_window_rationale(window.role_prompt, window.unsynced_window)
            for window in windows
        ]
        agent_ids = [window.agent_id for window in windows]
        return self.predict(rationales, agent_ids=agent_ids)


def build_window_rationale(role_prompt: str, unsynced_window: Sequence[AgentNode]) -> str:
    """Build the text encoded by Cos-Gate from role prompt and sync summaries."""

    lines = ["Role:", role_prompt.strip(), "", "Rationale:"]
    added_summary = False
    for node in unsynced_window:
        summary = node.sync_summary.strip()
        if not summary:
            continue
        lines.append(f"Step {node.step_index}: {summary}")
        added_summary = True

    if not added_summary:
        lines.append("No sync summary.")
    return "\n".join(lines)


def compute_cos_gate_score(
    rationales: Sequence[str],
    encoder: Encoder,
    *,
    eps: float = 1e-12,
    agent_ids: Sequence[str] | None = None,
) -> CosGateResult:
    mean_cosine, cosine_matrix, pairwise_scores = compute_mean_pairwise_cosine(
        rationales,
        encoder,
        eps=eps,
    )
    divergence_score = _clamp_01(1.0 - mean_cosine)
    math_divergence = _clamp_01((1.0 - mean_cosine) / 2.0)
    ids = list(agent_ids) if agent_ids is not None else None
    if ids is not None and len(ids) != len(rationales):
        raise ValueError("agent_ids length must match rationales length.")

    return CosGateResult(
        mean_cosine=float(mean_cosine),
        divergence_score=divergence_score,
        math_divergence=math_divergence,
        cosine_matrix=cosine_matrix,
        pairwise_scores=pairwise_scores,
        rationale_count=len(rationales),
        agent_ids=ids,
    )


def _clamp_01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))
