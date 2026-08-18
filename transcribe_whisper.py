#!/usr/bin/env python3
"""
Transcription faster-whisper + fusion avec un RTTM de diarisation.

Pipeline :
  1. faster-whisper transcrit l'audio avec timestamps mot par mot
  2. Lecture du RTTM produit par un script de diarisation
  3. Pour chaque mot, assignation du locuteur via timestamp
  4. Regroupement en tours de parole

Usage :
  python transcribe_whisper.py -i dicte_audio_3.m4a
  python transcribe_whisper.py -i dicte_audio_3.m4a --model small
  python transcribe_whisper.py -i dicte_audio_3.m4a --rttm path/to/file.rttm
  python transcribe_whisper.py -i dicte_audio_3.m4a --diar diarize_neural

Tailles disponibles : tiny, base, small, medium, large-v3
  tiny    ~75 Mo   très rapide, qualité moyenne
  base    ~150 Mo  rapide
  small   ~480 Mo  bon compromis (recommandé CPU)
  medium  ~1.5 Go  meilleure qualité, plus lent
  large-v3 ~3 Go   meilleure qualité absolue, lent sur CPU

Dépendances :
  pip install faster-whisper
"""

import argparse
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


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
#   Fusion mots ↔ locuteurs
# ═══════════════════════════════════════════════════════════════════════════════

def merge_words_to_turns(words: list, rttm: list) -> list:
    """words = [{"start","end","word"}], rttm = [(s,e,spk)]"""
    if not rttm:
        if not words:
            return []
        text = "".join(w["word"] for w in words).strip()
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
                              "".join(current[3]).strip()))
            current = [w["start"], w["end"], spk, [w["word"]]]
        else:
            current[1] = w["end"]
            current[3].append(w["word"])

    if current is not None:
        turns.append((current[0], current[1], current[2],
                      "".join(current[3]).strip()))
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
        description="Transcription faster-whisper + fusion RTTM")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--model", default="small",
                    help="tiny, base, small, medium, large-v3 (défaut: small)")
    ap.add_argument("--language", default="fr")
    ap.add_argument("--rttm", default=None,
                    help="Chemin RTTM (sinon cherché via --diar)")
    ap.add_argument("--diar", default="diarize_neural",
                    help="Nom du pipeline de diarisation pour trouver le RTTM "
                         "(défaut: diarize_neural)")
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--compute-type", default="int8",
                    help="int8 (rapide CPU), int8_float16, float16, float32")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

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
    print(f"  TRANSCRIPTION WHISPER + FUSION — {in_path.name}")
    print("=" * 78)
    print(f"  Modèle : {args.model}  |  compute_type : {args.compute_type}")
    print()

    # ── Chargement modèle ──
    t0 = time.perf_counter()
    print(f"  [1/3] Chargement du modèle (téléchargement au 1er lancement)...",
          flush=True)
    t1 = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    print(f"        Modèle chargé en {time.perf_counter()-t1:.1f}s")

    # ── Transcription ──
    print(f"  [2/3] Transcription en cours...", flush=True)
    t1 = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(in_path),
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=True,
        vad_filter=True,
    )
    duration = info.duration
    print(f"        Durée audio : {duration:.0f}s ({duration/60:.1f} min)  "
          f"|  langue détectée : {info.language} (p={info.language_probability:.2f})",
          flush=True)

    words = []
    n_segments = 0
    for seg in segments_iter:
        n_segments += 1
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                })
        # Log de progression
        elapsed = time.perf_counter() - t1
        rtf = elapsed / seg.end if seg.end > 0 else 0
        eta = (duration - seg.end) * rtf
        print(f"        {seg.end/duration*100:5.1f}%  |  "
              f"{seg.end:6.0f}s/{duration:.0f}s  |  "
              f"{len(words)} mots  |  RTF {rtf:.2f}x  |  ETA {eta:.0f}s",
              flush=True)

    print(f"        → {len(words)} mots, {n_segments} segments Whisper "
          f"en {time.perf_counter()-t1:.1f}s")

    # ── Fusion ──
    print(f"  [3/3] Fusion avec RTTM...", flush=True)
    if rttm_path.exists():
        rttm = load_rttm(rttm_path)
        print(f"        RTTM trouvé : {rttm_path}")
        print(f"        {len(rttm)} segments locuteurs")
    else:
        rttm = []
        print(f"        Pas de RTTM ({rttm_path}) → transcription sans locuteurs")

    turns = merge_words_to_turns(words, rttm)
    print(f"        {len(turns)} tours de parole")

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
    for start, end, spk, text in turns[:10]:
        snippet = text if len(text) < 80 else text[:77] + "..."
        print(f"    [{fmt_time(start)} - {fmt_time(end)}]  {spk} : {snippet}")
    if len(turns) > 10:
        print(f"    ... ({len(turns) - 10} de plus)")
    print()


if __name__ == "__main__":
    main()
