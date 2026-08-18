"""Voice Activity Detection — Silero VAD or pyannote segmentation-3.0."""

from __future__ import annotations
import os
from pathlib import Path
from .models import SpeechSegment


def run_vad(
    audio_path: str | Path,
    *,
    model: str = "silero",
    threshold: float = 0.45,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 50,
    speech_pad_ms: int = 20,
) -> list[SpeechSegment]:
    """Detect speech segments.

    Args:
        model: "silero" or "pyannote".
        threshold: speech probability threshold (Silero) or onset threshold (pyannote).
        Other args apply to Silero only; pyannote uses its own internal logic.
    """
    if model == "pyannote":
        return _vad_pyannote(
            audio_path,
            onset=threshold,
            min_duration_on=min_speech_duration_ms / 1000.0,
            min_duration_off=min_silence_duration_ms / 1000.0,
        )
    return _vad_silero(
        audio_path,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )


def _vad_silero(
    audio_path: str | Path,
    *,
    threshold: float = 0.45,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 50,
    speech_pad_ms: int = 20,
) -> list[SpeechSegment]:
    """Silero VAD."""
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    vad_model = load_silero_vad()
    wav = read_audio(str(audio_path))

    timestamps = get_speech_timestamps(
        wav,
        vad_model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    return [SpeechSegment(start=ts["start"], end=ts["end"]) for ts in timestamps]


def _vad_pyannote(
    audio_path: str | Path,
    *,
    onset: float = 0.5,
    min_duration_on: float = 0.2,
    min_duration_off: float = 0.05,
) -> list[SpeechSegment]:
    """pyannote segmentation-3.0 used as VAD.

    Requires:
      pip install pyannote.audio
      export HF_TOKEN=<your huggingface token>
      Accept license at https://huggingface.co/pyannote/segmentation-3.0
    """
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    # Try loading from .env at repo root if not already in environment.
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
            hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "pyannote VAD requires a HuggingFace token. "
            "Set HF_TOKEN env var (get one free at https://huggingface.co/settings/tokens) "
            "and accept the model license at https://huggingface.co/pyannote/segmentation-3.0"
        )

    seg_model = Model.from_pretrained(
        "pyannote/segmentation-3.0", use_auth_token=hf_token
    )
    pipeline = VoiceActivityDetection(segmentation=seg_model)
    pipeline.instantiate({
        "onset": onset,
        "offset": 0.2,
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off,
    })

    vad_result = pipeline(str(audio_path))

    segments = []
    for speech in vad_result.get_timeline().support():
        segments.append(SpeechSegment(start=speech.start, end=speech.end))
    return segments
