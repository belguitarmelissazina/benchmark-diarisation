#!/usr/bin/env python3
"""
Transcription sherpa-onnx (kroko streaming zipformer) + fusion RTTM.

Pipeline :
  1. Décode l'audio via imageio-ffmpeg
  2. Streame dans sherpa-onnx OnlineRecognizer en chunks de 0.5 s
  3. Récupère tokens + timestamps via rec.tokens() / rec.timestamps()
  4. Recolle les tokens BPE en mots avec timestamps
  5. Fusionne avec le RTTM de diarisation
  6. Écrit la transcription finale par locuteur

Usage :
  python transcribe_sherpa.py -i dicte_audio_3.m4a
  python transcribe_sherpa.py -i dicte_audio_3.m4a --diar diarize_neural
  python transcribe_sherpa.py -i dicte_audio_3.m4a --rttm path/to/file.rttm

Dépendances :
  pip install sherpa-onnx imageio-ffmpeg numpy
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx


# ═══════════════════════════════════════════════════════════════════════════════
#   Audio loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ═══════════════════════════════════════════════════════════════════════════════
#   Transcription sherpa-onnx streaming
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_sherpa(audio: np.ndarray, model_dir: Path, sr: int = 16000,
                      chunk_seconds: float = 0.5) -> list:
    """
    Retourne une liste de mots :
        [{"word": "...", "start": float, "end": float}, ...]
    """
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
    next_progress = 5.0

    for i in range(0, len(audio), chunk):
        stream.accept_waveform(sr, audio[i:i + chunk])
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        # Progression
        pct = min(i / len(audio) * 100, 100)
        if pct >= next_progress:
            elapsed = time.perf_counter() - t_start
            audio_done = i / sr
            rtf = elapsed / audio_done if audio_done > 0 else 0
            eta = (duration - audio_done) * rtf
            print(f"        {pct:5.1f}%  |  {audio_done:6.0f}s/{duration:.0f}s  "
                  f"|  RTF {rtf:.2f}x  |  ETA {eta:.0f}s", flush=True)
            next_progress += 5.0

    # Flush
    tail = np.zeros(int(0.5 * sr), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    while rec.is_ready(stream):
        rec.decode_stream(stream)

    # Récupérer tokens + timestamps
    tokens = rec.tokens(stream)
    timestamps = rec.timestamps(stream)

    # Recoller les tokens BPE en mots
    words = tokens_to_words(tokens, timestamps)
    return words


def tokens_to_words(tokens: list, timestamps: list) -> list:
    """
    Recolle les tokens BPE en mots.
    Convention SentencePiece : un token qui commence par ' ' (espace)
    marque le début d'un nouveau mot.
    """
    words = []
    current_word = ""
    current_start = None

    for tok, ts in zip(tokens, timestamps):
        if tok in (".", ",", "!", "?", ":", ";", "..."):
            # Ponctuation → rattacher au mot précédent
            if words:
                words[-1]["word"] += tok
                words[-1]["end"] = ts + 0.08
            continue

        if tok in (" ", "\u2581"):
            # Espace isolé (normal ou SentencePiece ▁) → ignorer
            continue

        if tok.startswith(" ") or tok.startswith("\u2581") or current_start is None:
            # Début d'un nouveau mot
            if current_word and current_start is not None:
                words.append({
                    "word": current_word,
                    "start": current_start,
                    "end": ts,
                })
            current_word = tok.lstrip(" \u2581")
            current_start = ts
        else:
            # Suite du mot
            current_word += tok

    # Dernier mot
    if current_word and current_start is not None:
        last_ts = timestamps[-1] if timestamps else current_start
        words.append({
            "word": current_word,
            "start": current_start,
            "end": last_ts + 0.08,
        })

    return words


# ═══════════════════════════════════════════════════════════════════════════════
#   RTTM
# ═══════════════════════════════════════════════════════════════════════════════

def load_rttm(path: Path) -> list:
    segments = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            segments.append((start, start + dur, spk))
    segments.sort(key=lambda x: x[0])
    return segments


def smooth_rttm(rttm: list, merge_gap: float = 2.0, min_dur: float = 1.0) -> list:
    """
    Post-traitement du RTTM pour réduire la fragmentation :
    1. Fusionne les segments consécutifs du même locuteur séparés de < merge_gap s
    2. Absorbe les micro-segments (< min_dur s) dans le locuteur voisin dominant
    """
    if not rttm:
        return rttm

    # Étape 1 : fusionner segments consécutifs du même locuteur
    merged = [list(rttm[0])]
    for s, e, spk in rttm[1:]:
        prev = merged[-1]
        if spk == prev[2] and s - prev[1] < merge_gap:
            prev[1] = max(prev[1], e)
        else:
            merged.append([s, e, spk])

    # Étape 2 : absorber les micro-segments dans le locuteur voisin
    cleaned = []
    for i, (s, e, spk) in enumerate(merged):
        dur = e - s
        if dur < min_dur and len(merged) > 1:
            # Trouver le voisin le plus proche (avant ou après) et prendre son locuteur
            prev_spk = merged[i - 1][2] if i > 0 else None
            next_spk = merged[i + 1][2] if i < len(merged) - 1 else None
            # Si les deux voisins sont le même locuteur → absorber
            if prev_spk and prev_spk == next_spk:
                continue  # on drop ce micro-segment
            # Si un seul voisin, absorber dans celui-là
            if prev_spk and not next_spk:
                continue
            if next_spk and not prev_spk:
                continue
        cleaned.append((s, e, spk))

    # Étape 3 : re-fusionner après absorption
    if not cleaned:
        return [(rttm[0][0], rttm[-1][1], rttm[0][2])]
    final = [list(cleaned[0])]
    for s, e, spk in cleaned[1:]:
        prev = final[-1]
        if spk == prev[2] and s - prev[1] < merge_gap:
            prev[1] = max(prev[1], e)
        else:
            final.append([s, e, spk])

    return [(s, e, spk) for s, e, spk in final]


def find_speaker(t: float, rttm: list, last_idx: int = 0) -> tuple:
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


# ═══════════════════════════════════════════════════════════════════════════════
#   Fusion mots ↔ locuteurs (avec lissage)
# ═══════════════════════════════════════════════════════════════════════════════

def merge_words_to_turns(words: list, rttm: list) -> list:
    if not rttm:
        if not words:
            return []
        text = " ".join(w["word"] for w in words)
        return [(words[0]["start"], words[-1]["end"], "SPK00", text)]

    turns = []
    last_idx = 0
    current = None

    for w in words:
        mid = (w["start"] + w["end"]) / 2.0
        spk, last_idx = find_speaker(mid, rttm, last_idx)
        if current is None or current[2] != spk:
            if current is not None:
                turns.append((current[0], current[1], current[2],
                              " ".join(current[3])))
            current = [w["start"], w["end"], spk, [w["word"]]]
        else:
            current[1] = w["end"]
            current[3].append(w["word"])

    if current is not None:
        turns.append((current[0], current[1], current[2],
                      " ".join(current[3])))
    return turns


# ═══════════════════════════════════════════════════════════════════════════════
#   Sortie
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def write_transcript(path: Path, turns: list):
    with open(path, "w", encoding="utf-8") as f:
        for start, end, spk, text in turns:
            f.write(f"[{fmt_time(start)} - {fmt_time(end)}]  {spk} : {text}\n")


def write_words_json(path: Path, words: list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Transcription sherpa-onnx (kroko) + fusion RTTM")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--model-dir", default="sherpa-onnx-streaming-zipformer-fr-kroko")
    ap.add_argument("--rttm", default=None)
    ap.add_argument("--diar", default="diarize_neural",
                    help="Pipeline de diarisation (défaut: diarize_neural)")
    ap.add_argument("--output-dir", "-o", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    file_id = in_path.stem
    pipeline_name = Path(__file__).stem

    if args.rttm:
        rttm_path = Path(args.rttm)
        diar_name = rttm_path.parent.parent.name  # ex: diarize_neural
    else:
        diar_name = args.diar
        rttm_path = (in_path.parent / "outputs" / diar_name
                     / file_id / f"{file_id}.rttm")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = in_path.parent / "outputs" / pipeline_name / diar_name / file_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  TRANSCRIPTION SHERPA-ONNX (KROKO) + FUSION — {in_path.name}")
    print("=" * 78)

    # ── Audio ──
    t0 = time.perf_counter()
    sr = 16000
    print(f"  [1/4] Chargement audio...", flush=True)
    audio = load_audio(in_path, sr=sr)
    duration = len(audio) / sr
    print(f"        {duration:.1f}s ({duration/60:.1f} min)")

    # ── Transcription ──
    t1 = time.perf_counter()
    print(f"  [2/4] Transcription sherpa-onnx...", flush=True)
    words = transcribe_sherpa(audio, Path(args.model_dir), sr=sr)
    print(f"        → {len(words)} mots en {time.perf_counter()-t1:.1f}s")

    # ── RTTM ──
    print(f"  [3/4] Chargement RTTM...", flush=True)
    if rttm_path.exists():
        rttm = load_rttm(rttm_path)
        print(f"        {rttm_path} → {len(rttm)} segments locuteurs")
    else:
        rttm = []
        print(f"        Pas de RTTM ({rttm_path}) → sans locuteurs")

    # ── Fusion ──
    t1 = time.perf_counter()
    print(f"  [4/4] Fusion mots ↔ locuteurs...", flush=True)
    turns = merge_words_to_turns(words, rttm)
    print(f"        {len(turns)} tours de parole — {time.perf_counter()-t1:.1f}s")

    # ── Sortie ──
    txt_path = out_dir / f"{file_id}.transcript.txt"
    json_path = out_dir / f"{file_id}.words.json"
    write_transcript(txt_path, turns)
    write_words_json(json_path, words)

    total = time.perf_counter() - t0
    print()
    print("=" * 78)
    print(f"  RÉSULTAT")
    print("=" * 78)
    print(f"  Tours de parole    : {len(turns)}")
    print(f"  Mots transcrits    : {len(words)}")
    print(f"  RTF total          : {total/duration:.2f}x")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  → {txt_path}")
    print(f"  → {json_path}")
    print()
    print("  Aperçu :")
    for start, end, spk, text in turns[:15]:
        snippet = text if len(text) < 80 else text[:77] + "..."
        print(f"    [{fmt_time(start)} - {fmt_time(end)}]  {spk} : {snippet}")
    if len(turns) > 15:
        print(f"    ... ({len(turns) - 15} de plus)")
    print()


if __name__ == "__main__":
    main()
