"""Plain dataclass models — no pydantic dependency."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpeechSegment:
    """A speech segment detected by VAD (no speaker label)."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SubSegment:
    """An embedding window within a speech segment."""
    start: float
    end: float
    parent_idx: int


@dataclass
class Segment:
    """A diarization segment with speaker label."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SpeakerEstimationDetails:
    method: str = "gmm_bic"
    best_k: int = 1
    pca_dim: Optional[int] = None
    k_bics: dict = field(default_factory=dict)
    reason: Optional[str] = None
    cosine_sim_p10: Optional[float] = None
