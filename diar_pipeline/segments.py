"""Segment assembly + RTTM/TXT output."""

from __future__ import annotations
from pathlib import Path
import numpy as np

from .models import Segment, SpeechSegment, SubSegment


def build_segments(
    speech_segments: list[SpeechSegment],
    subsegments: list[SubSegment],
    labels: np.ndarray,
    *,
    merge_gap: float = 0.7,
) -> list[Segment]:
    """Assemble final diarization segments from sub-segment labels.

    Short VAD segments without embeddings are assigned the nearest (by time)
    labeled sub-segment's speaker, then adjacent same-speaker segments are
    merged if the gap between them is < merge_gap seconds.
    """
    raw: list[tuple[float, float, str]] = []
    for sub, label in zip(subsegments, labels):
        raw.append((sub.start, sub.end, f"SPEAKER_{int(label):02d}"))

    covered = {sub.parent_idx for sub in subsegments}
    for idx, seg in enumerate(speech_segments):
        if idx in covered:
            continue
        mid = (seg.start + seg.end) / 2
        best_spk, best_dist = "SPEAKER_00", float("inf")
        for sub, label in zip(subsegments, labels):
            d = abs(mid - (sub.start + sub.end) / 2)
            if d < best_dist:
                best_dist = d
                best_spk = f"SPEAKER_{int(label):02d}"
        raw.append((seg.start, seg.end, best_spk))

    raw.sort(key=lambda x: x[0])
    if not raw:
        return []

    merged = [[raw[0][0], raw[0][1], raw[0][2]]]
    for start, end, spk in raw[1:]:
        prev = merged[-1]
        if spk == prev[2] and (start - prev[1]) < merge_gap:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end, spk])

    return [Segment(start=m[0], end=m[1], speaker=m[2]) for m in merged]


def write_txt(path: Path, segments: list[Segment]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:7.2f} - {seg.end:7.2f}]  {seg.speaker}\n")


def write_rttm(path: Path, segments: list[Segment], file_id: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            dur = seg.end - seg.start
            f.write(
                f"SPEAKER {file_id} 1 {seg.start:.6f} {dur:.6f} "
                f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
            )
