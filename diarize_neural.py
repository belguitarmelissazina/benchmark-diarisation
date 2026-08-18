#!/usr/bin/env python3
"""
Diarisation neuronale via le package `diarize` (FoxNoseTech).

Pipeline interne (géré par le package) :
  1. Silero VAD            → segments de parole
  2. WeSpeaker ResNet34-LM → embeddings 256-dim (ONNX, CPU)
  3. GMM BIC               → estimation du nombre de locuteurs
  4. Spectral Clustering   → assignation des labels

Tout tourne sur CPU, sans token HuggingFace, sans GPU.
~10.8% DER sur VoxConverse, ~8x plus rapide que le temps réel sur CPU.

Usage :
  python diarize_neural.py -i dicte_audio_3.m4a
  python diarize_neural.py -i dicte_audio_3.m4a --num-speakers 2
  python diarize_neural.py -i dicte_audio_3.m4a --min-speakers 2 --max-speakers 5

Dépendances :
  pip install diarize imageio-ffmpeg
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from diarize import diarize as run_diarize


# ═══════════════════════════════════════════════════════════════════════════════
#   Conversion audio → WAV 16 kHz mono (autonome via imageio-ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(src: Path, sr: int = 16000) -> Path:
    """Convertit n'importe quel format audio en WAV 16 kHz mono temporaire."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(tempfile.gettempdir()) / f"{src.stem}_16k.wav"
    cmd = [
        ffmpeg_exe, "-y", "-v", "error", "-i", str(src),
        "-ac", "1", "-ar", str(sr), str(tmp),
    ]
    subprocess.run(cmd, check=True)
    return tmp


# ═══════════════════════════════════════════════════════════════════════════════
#   Sortie texte lisible
# ═══════════════════════════════════════════════════════════════════════════════

def write_txt(path: Path, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:7.2f} - {seg.end:7.2f}]  {seg.speaker}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diarisation neuronale via le package `diarize`")
    parser.add_argument("--input", "-i", required=True, help="Fichier audio")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Nombre exact de locuteurs (si connu)")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    file_id = in_path.stem
    pipeline_name = Path(__file__).stem
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = in_path.parent / "outputs" / pipeline_name / file_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DIARISATION NEURONALE (diarize) — {in_path.name}")
    print("=" * 78)

    # ── Conversion ──
    t0 = time.perf_counter()
    wav_path = convert_to_wav(in_path)
    print(f"  [1/2] Audio préparé   : {wav_path.name}")

    # ── Diarisation ──
    t1 = time.perf_counter()
    kwargs = {}
    if args.num_speakers is not None:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        kwargs["max_speakers"] = args.max_speakers

    result = run_diarize(str(wav_path), **kwargs)
    print(f"  [2/2] Diarisation     : {result.num_speakers} locuteurs, "
          f"{len(result.segments)} segments — {time.perf_counter()-t1:.1f}s")

    # ── Sortie ──
    rttm_path = out_dir / f"{file_id}.rttm"
    txt_path = out_dir / f"{file_id}.diarization.txt"
    result.to_rttm(str(rttm_path))
    write_txt(txt_path, result.segments)

    total = time.perf_counter() - t0
    duration = result.audio_duration
    print()
    print("=" * 78)
    print(f"  RÉSULTAT")
    print("=" * 78)
    print(f"  Durée audio        : {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Locuteurs détectés : {result.num_speakers}")
    print(f"  Tours de parole    : {len(result.segments)}")
    print(f"  RTF                : {total/duration:.2f}x")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  → {rttm_path}")
    print(f"  → {txt_path}")
    print()
    print("  Aperçu :")
    for seg in result.segments[:10]:
        print(f"    [{seg.start:7.2f} - {seg.end:7.2f}]  {seg.speaker}")
    if len(result.segments) > 10:
        print(f"    ... ({len(result.segments) - 10} de plus)")
    print()


if __name__ == "__main__":
    main()
