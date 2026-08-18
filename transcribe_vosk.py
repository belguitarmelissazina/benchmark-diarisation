#!/usr/bin/env python3
"""
Transcription Vosk + fusion avec un RTTM de diarisation.

Pipeline :
  1. Décode l'audio (n'importe quel format) en PCM 16 kHz mono
  2. Streame l'audio dans Vosk en chunks de 0.25 s (méthode native streaming)
  3. Récupère tous les mots avec timestamps (start/end)
  4. Lit le RTTM produit par diarize_bic_gmm.py
  5. Pour chaque mot, assigne le locuteur correspondant via timestamp
  6. Regroupe en tours de parole et écrit la transcription finale

Usage :
  python transcribe_vosk.py -i dicte_audio_1.m4a
  python transcribe_vosk.py -i dicte_audio_1.m4a --rttm path/to/file.rttm
  python transcribe_vosk.py -i dicte_audio_1.m4a --model vosk-model-fr-0.22

Dépendances :
  pip install vosk imageio-ffmpeg numpy
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model, SetLogLevel


# ═══════════════════════════════════════════════════════════════════════════════
#   Audio loader (autonome via imageio-ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio_pcm16(path: Path, sr: int = 16000) -> bytes:
    """Décode n'importe quel format audio en PCM 16-bit mono brut (bytes)."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return proc.stdout


# ═══════════════════════════════════════════════════════════════════════════════
#   Transcription Vosk (streaming natif, chunks de 0.25 s)
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_vosk(pcm: bytes, model_path: Path, sr: int = 16000,
                    chunk_seconds: float = 0.25) -> list:
    """
    Streame l'audio dans Vosk et retourne la liste de tous les mots :
        [{"word": "...", "start": float, "end": float, "conf": float}, ...]
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle Vosk introuvable : {model_path}")

    print(f"        Chargement du modèle Vosk ({model_path.name})...", flush=True)
    t_load = time.perf_counter()
    SetLogLevel(-1)  # silence Kaldi
    model = Model(str(model_path))
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(True)  # ← essentiel pour avoir les timestamps mot par mot
    print(f"        Modèle chargé en {time.perf_counter()-t_load:.1f}s", flush=True)

    chunk_bytes = int(sr * chunk_seconds) * 2  # 2 bytes/sample (int16)
    total_bytes = len(pcm)
    duration_s = total_bytes / 2 / sr
    words = []

    print(f"        Transcription de {duration_s:.0f}s d'audio en cours...", flush=True)
    t_start = time.perf_counter()
    offset = 0
    next_progress = 5.0  # log tous les 5%
    while offset < total_bytes:
        chunk = pcm[offset:offset + chunk_bytes]
        offset += chunk_bytes
        if rec.AcceptWaveform(chunk):
            res = json.loads(rec.Result())
            if "result" in res:
                words.extend(res["result"])
        # Progression
        pct = offset / total_bytes * 100
        if pct >= next_progress:
            elapsed = time.perf_counter() - t_start
            audio_done = offset / 2 / sr
            rtf = elapsed / audio_done if audio_done > 0 else 0
            eta = (duration_s - audio_done) * rtf
            print(f"        {pct:5.1f}%  |  {audio_done:6.0f}s/{duration_s:.0f}s  "
                  f"|  {len(words)} mots  |  RTF {rtf:.2f}x  |  ETA {eta:.0f}s",
                  flush=True)
            next_progress += 5.0

    # Dernier morceau (utterance non terminée par l'endpointer)
    final = json.loads(rec.FinalResult())
    if "result" in final:
        words.extend(final["result"])

    return words


# ═══════════════════════════════════════════════════════════════════════════════
#   Lecture RTTM
# ═══════════════════════════════════════════════════════════════════════════════

def load_rttm(path: Path) -> list:
    """Retourne [(start, end, speaker_label), ...] trié par start."""
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


def find_speaker(t: float, rttm: list, last_idx: int = 0) -> tuple:
    """Trouve le locuteur actif au temps t. Retourne (label, idx_pour_reprise)."""
    # Recherche linéaire avec mémoire (les mots arrivent dans l'ordre)
    n = len(rttm)
    i = last_idx
    while i < n:
        s, e, spk = rttm[i]
        if t < s:
            # Mot tombe avant le prochain segment → on attribue au plus proche
            if i == 0:
                return rttm[0][2], 0
            # Compare distance au segment précédent vs suivant
            prev_end = rttm[i - 1][1]
            if t - prev_end < s - t:
                return rttm[i - 1][2], i - 1
            return spk, i
        if s <= t < e:
            return spk, i
        i += 1
    # Au-delà du dernier segment
    return rttm[-1][2], n - 1


# ═══════════════════════════════════════════════════════════════════════════════
#   Fusion mots ↔ locuteurs → tours de parole
# ═══════════════════════════════════════════════════════════════════════════════

def merge_words_to_turns(words: list, rttm: list) -> list:
    """
    Assigne un locuteur à chaque mot puis regroupe en tours de parole.
    Retourne [(start, end, speaker, "texte ..."), ...]
    """
    if not rttm:
        # Pas de diarisation → un seul "tour" avec tout le texte
        if not words:
            return []
        text = " ".join(w["word"] for w in words)
        return [(words[0]["start"], words[-1]["end"], "SPK00", text)]

    turns = []
    last_idx = 0
    current = None  # (start, end, spk, [mots])

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
    parser = argparse.ArgumentParser(
        description="Transcription Vosk + fusion avec RTTM de diarisation")
    parser.add_argument("--input", "-i", required=True, help="Fichier audio")
    parser.add_argument("--model", default="vosk-model-fr-0.22",
                        help="Dossier du modèle Vosk")
    parser.add_argument("--rttm", default=None,
                        help="Fichier RTTM")
    parser.add_argument("--diar", default="diarize_bic_gmm",
                        help="Nom du pipeline de diarisation (défaut: diarize_bic_gmm)")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--chunk-seconds", type=float, default=0.25,
                        help="Taille des chunks streamés à Vosk (défaut 0.25 s)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    file_id = in_path.stem
    pipeline_name = Path(__file__).stem

    if args.rttm:
        rttm_path = Path(args.rttm)
        diar_name = rttm_path.parent.parent.name
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
    print(f"  TRANSCRIPTION VOSK + FUSION — {in_path.name}")
    print("=" * 78)

    # ── Audio ──
    t0 = time.perf_counter()
    sr = 16000
    pcm = load_audio_pcm16(in_path, sr=sr)
    duration = len(pcm) / 2 / sr
    print(f"  [1/4] Audio chargé    : {duration:.1f}s ({duration/60:.1f} min)")

    # ── Vosk ──
    t1 = time.perf_counter()
    words = transcribe_vosk(pcm, Path(args.model), sr=sr,
                            chunk_seconds=args.chunk_seconds)
    print(f"  [2/4] Vosk            : {len(words)} mots "
          f"— {time.perf_counter()-t1:.1f}s")

    # ── RTTM ──
    if rttm_path.exists():
        rttm = load_rttm(rttm_path)
        print(f"  [3/4] RTTM            : {len(rttm)} segments locuteurs "
              f"({rttm_path.name})")
    else:
        rttm = []
        print(f"  [3/4] RTTM            : aucun trouvé ({rttm_path}) "
              f"→ transcription sans locuteurs")

    # ── Fusion ──
    t1 = time.perf_counter()
    turns = merge_words_to_turns(words, rttm)
    print(f"  [4/4] Fusion          : {len(turns)} tours de parole "
          f"— {time.perf_counter()-t1:.1f}s")

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
    print(f"  RTF                : {total/duration:.2f}x")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  → {txt_path}")
    print(f"  → {json_path}")
    print()
    print("  Aperçu :")
    for start, end, spk, text in turns[:10]:
        snippet = text if len(text) < 80 else text[:77] + "..."
        print(f"    [{fmt_time(start)} - {fmt_time(end)}]  {spk} : {snippet}")
    if len(turns) > 10:
        print(f"    ... ({len(turns) - 10} de plus)")
    print()


if __name__ == "__main__":
    main()
