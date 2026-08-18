"""Transcription step — sherpa-onnx (kroko streaming Zipformer).

Produces word-level timestamps, then aligns words to speaker segments
to create a speaker-attributed transcript.

Depends on: sherpa-onnx, imageio-ffmpeg, numpy (already in venv).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Default model path (kroko French streaming Zipformer)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "sherpa-onnx-streaming-zipformer-fr-kroko"


# ── Audio loading ────────────────────────────────────────────────────────────

def load_audio_pcm(audio_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Decode any audio to float32 mono PCM via ffmpeg."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-i", str(audio_path),
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ── Sherpa-onnx streaming transcription ──────────────────────────────────────

def transcribe(
    audio: np.ndarray,
    model_dir: Path = DEFAULT_MODEL_DIR,
    sr: int = 16000,
    chunk_seconds: float = 0.5,
    verbose: bool = True,
) -> list[dict]:
    """Transcribe audio -> list of {word, start, end}.

    Uses BPE token reassembly with SentencePiece U+2581 handling.
    """
    import sherpa_onnx

    md = model_dir
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(md / "encoder.onnx"),
        decoder=str(md / "decoder.onnx"),
        joiner=str(md / "joiner.onnx"),
        tokens=str(md / "tokens.txt"),
        num_threads=4,
        sample_rate=sr,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )

    stream = rec.create_stream()
    chunk = int(chunk_seconds * sr)
    duration = len(audio) / sr
    t_start = time.perf_counter()
    next_progress = 10.0

    for i in range(0, len(audio), chunk):
        stream.accept_waveform(sr, audio[i:i + chunk])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        pct = min(i / len(audio) * 100, 100)
        if verbose and pct >= next_progress:
            elapsed = time.perf_counter() - t_start
            audio_done = i / sr
            rtf = elapsed / audio_done if audio_done > 0 else 0
            print(f"        {pct:5.1f}% | {audio_done:.0f}s/{duration:.0f}s | "
                  f"RTF {rtf:.2f}x", flush=True)
            next_progress += 10.0

    # Flush
    tail = np.zeros(int(0.5 * sr), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    while rec.is_ready(stream):
        rec.decode_stream(stream)

    tokens = rec.tokens(stream)
    timestamps = rec.timestamps(stream)
    return _tokens_to_words(tokens, timestamps)


def _tokens_to_words(tokens: list, timestamps: list) -> list[dict]:
    """Reassemble BPE tokens into words with start/end timestamps."""
    words = []
    current_word = ""
    current_start = None

    for tok, ts in zip(tokens, timestamps):
        if tok in (".", ",", "!", "?", ":", ";", "..."):
            if words:
                words[-1]["word"] += tok
                words[-1]["end"] = ts + 0.08
            continue
        if tok in (" ", "\u2581"):
            continue
        if tok.startswith(" ") or tok.startswith("\u2581") or current_start is None:
            if current_word and current_start is not None:
                words.append({"word": current_word, "start": current_start, "end": ts})
            current_word = tok.lstrip(" \u2581")
            current_start = ts
        else:
            current_word += tok

    if current_word and current_start is not None:
        last_ts = timestamps[-1] if timestamps else current_start
        words.append({"word": current_word, "start": current_start,
                      "end": last_ts + 0.08})
    return words


# ── Speaker attribution ──────────────────────────────────────────────────────

def _find_speaker(t: float, rttm: list[tuple], last_idx: int = 0) -> tuple[str, int]:
    n = len(rttm)
    i = last_idx
    while i < n:
        s, e, spk = rttm[i]
        if t < s:
            if i == 0:
                return rttm[0][2], 0
            prev_end = rttm[i - 1][1]
            if t - prev_end < s - t:
                return rttm[i - 1][2], i - 1
            return spk, i
        if s <= t < e:
            return spk, i
        i += 1
    return rttm[-1][2], n - 1


def align_words_to_speakers(
    words: list[dict],
    segments,
) -> list[dict]:
    """Add 'speaker' key to each word based on diarization segments.

    segments: list of Segment (from diar_pipeline.models) with .start, .end, .speaker
    """
    if not segments or not words:
        for w in words:
            w["speaker"] = "SPEAKER_00"
        return words

    rttm = [(s.start, s.end, s.speaker) for s in segments]
    rttm.sort(key=lambda x: x[0])

    last_idx = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk, last_idx = _find_speaker(mid, rttm, last_idx)
        w["speaker"] = spk
    return words


def align_words_by_boundaries(
    words: list[dict],
    segments,
    hop_seconds: float = 5.0,
) -> list[dict]:
    """Alternative alignment: transcribe once, then insert speaker changes at
    natural punctuation boundaries near the diarization speaker-change times.

    Logic adapted from the "apply_speaker_tags_to_transcript" approach:
    for every moment the diarization says the speaker changes, we search the
    closest '.', '!', '?' within ±hop_seconds of that time and place the
    boundary there. Words between two boundaries share one speaker.

    Returns: a *copy* of `words`, each with a 'speaker' key, where speaker
    transitions are snapped to punctuation instead of per-word midpoints.
    """
    if not segments or not words:
        out = [dict(w) for w in words]
        for w in out:
            w["speaker"] = "SPEAKER_00"
        return out

    rttm = sorted([(s.start, s.end, s.speaker) for s in segments],
                  key=lambda x: x[0])

    # Build speaker-change list: (time, new_speaker) each time speaker differs
    changes: list[tuple[float, str]] = []
    prev_spk = None
    for s, _e, spk in rttm:
        if spk != prev_spk:
            changes.append((s, spk))
            prev_spk = spk

    out = [dict(w) for w in words]
    out.sort(key=lambda x: x["start"])

    # Map each change time -> word index where the change should be applied
    # (snapped to nearest punctuation within ±hop_seconds)
    change_points: list[tuple[int, str]] = []  # (word_idx, new_speaker)
    word_idx = 0
    n = len(out)
    PUNCT = set(".!?।")

    for ch_time, new_spk in changes:
        # Advance word_idx to the first word starting after ch_time
        while word_idx < n and out[word_idx]["start"] <= ch_time:
            word_idx += 1
        if word_idx == 0:
            change_points.append((0, new_spk))
            continue
        if word_idx >= n:
            break

        ref_time = out[word_idx]["start"]
        forward_idx = -1
        backward_idx = -1

        # Forward search for punctuation
        for j in range(word_idx, n):
            if out[j]["start"] > ref_time + hop_seconds:
                break
            if any(p in out[j]["word"] for p in PUNCT):
                forward_idx = j
                break

        # Backward search for punctuation
        for j in range(word_idx - 1, -1, -1):
            if out[j]["start"] < ref_time - hop_seconds:
                break
            if any(p in out[j]["word"] for p in PUNCT):
                backward_idx = j
                break

        if forward_idx != -1 and backward_idx != -1:
            chosen = (forward_idx if (forward_idx - word_idx)
                      <= (word_idx - backward_idx) else backward_idx)
        elif forward_idx != -1:
            chosen = forward_idx
        elif backward_idx != -1:
            chosen = backward_idx
        else:
            chosen = word_idx

        # Boundary applies *after* the punctuation word -> speaker of next word
        change_points.append((chosen + 1, new_spk))

    # Propagate speakers from boundaries forward
    first_spk = rttm[0][2]
    current = first_spk
    boundary_map = dict(change_points)
    for i, w in enumerate(out):
        if i in boundary_map:
            current = boundary_map[i]
        w["speaker"] = current
    return out


def words_to_turns(words: list[dict]) -> list[dict]:
    """Group consecutive words with the same speaker into turns.

    Returns: [{start, end, speaker, text}]
    """
    if not words:
        return []
    turns = []
    cur = {"start": words[0]["start"], "end": words[0]["end"],
           "speaker": words[0].get("speaker", "SPEAKER_00"),
           "words": [words[0]["word"]]}
    for w in words[1:]:
        spk = w.get("speaker", "SPEAKER_00")
        if spk == cur["speaker"]:
            cur["end"] = w["end"]
            cur["words"].append(w["word"])
        else:
            cur["text"] = " ".join(cur.pop("words"))
            turns.append(cur)
            cur = {"start": w["start"], "end": w["end"],
                   "speaker": spk, "words": [w["word"]]}
    cur["text"] = " ".join(cur.pop("words"))
    turns.append(cur)
    return turns


# ── Output formatting ────────────────────────────────────────────────────────

def format_transcript_txt(turns: list[dict], with_timestamps: bool = False) -> str:
    lines = []
    for t in turns:
        if with_timestamps:
            s = t["start"]
            e = t["end"]
            sm, ss = divmod(s, 60)
            em, es = divmod(e, 60)
            lines.append(f"[{int(sm):02d}:{ss:05.2f} - {int(em):02d}:{es:05.2f}]  "
                          f"{t['speaker']} : {t['text']}")
        else:
            lines.append(f"{t['speaker']} : {t['text']}")
    return "\n".join(lines)
