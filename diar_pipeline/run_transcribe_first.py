#!/usr/bin/env python3
"""Transcribe-first pipeline: ASR on raw audio, then diarize, then align.

This is the inverse of the default pipeline (which diarizes first, then
transcribes). The hypothesis is that giving the ASR engine the full
uncut audio (instead of pre-segmented chunks) produces better word
timestamps and fewer hallucinations at speaker boundaries.

Pipeline order:
  1. Audio conversion (16kHz mono WAV)
  2. ASR on full audio → words with timestamps
  3. VAD + Embeddings + Clustering → diarization segments
  4. Align words to speakers (midpoint + boundary modes)
  5. DER evaluation (optional)

All runs are tracked with MLflow.

Usage:
  python -m diar_pipeline.run_transcribe_first -i meeting.wav --transcribe
  python -m diar_pipeline.run_transcribe_first -i meeting.wav --transcribe --reference-rttm ref.rttm
"""

from __future__ import annotations
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import sys
import time
from pathlib import Path

from .audio import convert_to_wav, get_audio_duration
from .vad import run_vad
from .embeddings import extract_embeddings, EMBEDDING_WINDOW, EMBEDDING_STEP
from .clustering import cluster_speakers
from .refinement import refine_vbx
from .segments import build_segments, write_rttm, write_txt
from .tracking import (
    Tracker,
    log_vad_artifacts,
    log_embedding_artifacts,
    log_sim_artifacts,
    log_clustering_artifacts,
    compute_der,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcribe-first pipeline: ASR → diarize → align")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=20)
    ap.add_argument("--embed", choices=["resnet34", "campplus", "ecapa"],
                    default="resnet34")
    ap.add_argument("--estimate", choices=["gmm_bic", "nmesc"], default="gmm_bic")
    ap.add_argument("--cluster", choices=["sc", "ahc", "meanshift"], default="sc")
    ap.add_argument("--no-enhance", action="store_true")
    ap.add_argument("--silhouette-refine", action="store_true")
    ap.add_argument("--win-len", type=float, default=None)
    ap.add_argument("--hop-len", type=float, default=None)
    # VAD
    ap.add_argument("--vad-model", choices=["silero", "pyannote"], default="silero")
    ap.add_argument("--vad-threshold", type=float, default=0.4)
    ap.add_argument("--vad-min-speech-ms", type=int, default=200)
    ap.add_argument("--vad-min-silence-ms", type=int, default=50)
    ap.add_argument("--vad-pad-ms", type=int, default=20)
    ap.add_argument("--refine-vbx", action="store_true")
    ap.add_argument("--vbx-Fa", type=float, default=0.4)
    ap.add_argument("--vbx-Fb", type=float, default=17.0)
    # Transcription
    ap.add_argument("--model-dir", default=None)
    # MLflow
    ap.add_argument("--experiment", default="transcribe_first")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--run-name", default=None)
    # Evaluation
    ap.add_argument("--reference-rttm", default=None)
    ap.add_argument("--der-collar", type=float, default=0.25)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    file_id = in_path.stem
    win_len = args.win_len if args.win_len else EMBEDDING_WINDOW
    hop_len = args.hop_len if args.hop_len else EMBEDDING_STEP

    run_name = args.run_name or f"txfirst__{file_id}"

    out_dir = (Path(args.output_dir) if args.output_dir
               else in_path.parent / "outputs" / "transcribe_first" / file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  TRANSCRIBE-FIRST — {in_path.name}")
    print(f"  Order: ASR → diarize → align")
    print(f"  embed={args.embed} | estimate={args.estimate} | cluster={args.cluster}")
    print("=" * 78)

    tags = {
        "file_id": file_id,
        "pipeline": "transcribe_first",
        "embed": args.embed,
        "cluster": args.cluster,
    }

    with Tracker(experiment=args.experiment, run_name=run_name,
                 tracking_uri=args.tracking_uri, tags=tags) as tr:

        tr.log_params({
            "input_file": str(in_path),
            "pipeline": "transcribe_first",
            "embed": args.embed,
            "estimate": args.estimate,
            "cluster": args.cluster,
            "enhance": not args.no_enhance,
            "win_len": win_len,
            "hop_len": hop_len,
            "vad_model": args.vad_model,
            "vad_threshold": args.vad_threshold,
            "num_speakers": args.num_speakers,
            "refine_vbx": args.refine_vbx,
        })
        print(f"  MLflow run_id: {tr.run_id}")

        timings: dict[str, float] = {}
        t_pipeline = time.perf_counter()

        # ── 1. Audio ────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        wav_path = convert_to_wav(in_path)
        duration = get_audio_duration(str(wav_path))
        timings["time_conversion"] = time.perf_counter() - t0
        tr.ram.mark("conversion")
        tr.log_metric("audio_duration_sec", duration)
        tr.log_metric("time_conversion", timings["time_conversion"])
        tr.log_artifact_file(wav_path, subdir="audio")
        print(f"  [1] audio -> {wav_path.name} ({duration:.1f}s)")

        # ── 2. TRANSCRIPTION FIRST ──────────────────────────────────────────
        from .transcription import (
            load_audio_pcm, transcribe as run_asr,
            align_words_to_speakers, align_words_by_boundaries,
            words_to_turns, format_transcript_txt,
            DEFAULT_MODEL_DIR,
        )
        model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
        t_asr = time.perf_counter()
        print(f"  [2] Transcription (sherpa-onnx) — FIRST...")
        audio_pcm = load_audio_pcm(in_path)
        words = run_asr(audio_pcm, model_dir=model_dir)
        timings["time_transcription"] = time.perf_counter() - t_asr
        tr.ram.mark("transcription")
        tr.log_metric("time_transcription", timings["time_transcription"])
        tr.log_metric("num_words", len(words))
        if duration > 0:
            tr.log_metric("wpm", len(words) / (duration / 60))
        print(f"      -> {len(words)} words — {timings['time_transcription']:.1f}s")

        # Save raw transcription (before speaker assignment)
        tr.log_artifact_json(words, "words_raw.json", subdir="transcript")

        # ── 3. VAD ──────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        speech_segments = run_vad(
            str(wav_path),
            model=args.vad_model,
            threshold=args.vad_threshold,
            min_speech_duration_ms=args.vad_min_speech_ms,
            min_silence_duration_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_pad_ms,
        )
        timings["time_vad"] = time.perf_counter() - t1
        tr.ram.mark("vad")
        tr.log_metric("time_vad", timings["time_vad"])
        log_vad_artifacts(tr, speech_segments, duration)
        print(f"  [3] VAD: {len(speech_segments)} seg — {timings['time_vad']:.1f}s")
        if not speech_segments:
            print("  ERROR: no speech detected")
            sys.exit(1)

        # ── 4. Embeddings ───────────────────────────────────────────────────
        t2 = time.perf_counter()
        embeddings, subsegments = extract_embeddings(
            str(wav_path), speech_segments,
            win_len=win_len, hop_len=hop_len, embed_model=args.embed)
        timings["time_embeddings"] = time.perf_counter() - t2
        tr.ram.mark("embeddings")
        tr.log_metric("time_embeddings", timings["time_embeddings"])
        log_embedding_artifacts(tr, embeddings, subsegments, args.embed)
        print(f"  [4] embeddings: {embeddings.shape[0]} x {embeddings.shape[1]}"
              f" — {timings['time_embeddings']:.1f}s")

        # ── 4b. Similarity ──────────────────────────────────────────────────
        t_sim = time.perf_counter()
        log_sim_artifacts(tr, embeddings, enhance=not args.no_enhance)
        timings["time_similarity"] = time.perf_counter() - t_sim

        # ── 5. Clustering ───────────────────────────────────────────────────
        t3 = time.perf_counter()
        labels, details = cluster_speakers(
            embeddings,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            num_speakers=args.num_speakers,
            method=args.cluster,
            enhance=not args.no_enhance,
            estimate_method=args.estimate,
            silhouette_refine=args.silhouette_refine,
        )
        timings["time_clustering"] = time.perf_counter() - t3
        tr.ram.mark("clustering")
        tr.log_metric("time_clustering", timings["time_clustering"])
        n_spk = len(set(labels.tolist()))
        print(f"  [5] clustering: {n_spk} speakers — {timings['time_clustering']:.1f}s")

        log_clustering_artifacts(tr, labels, embeddings,
                                 stage="pre_vbx" if args.refine_vbx else "final")

        # ── 6. VBx refinement (optional) ────────────────────────────────────
        if args.refine_vbx:
            t4 = time.perf_counter()
            labels = refine_vbx(
                embeddings, labels, subsegments,
                Fa=args.vbx_Fa, Fb=args.vbx_Fb)
            timings["time_vbx"] = time.perf_counter() - t4
            tr.ram.mark("vbx")
            tr.log_metric("time_vbx", timings["time_vbx"])
            n_spk = len(set(labels.tolist()))
            print(f"      VBx -> {n_spk} speakers — {timings['time_vbx']:.1f}s")
            log_clustering_artifacts(tr, labels, embeddings, stage="final")

        # ── 7. Build diarization segments + RTTM ────────────────────────────
        segments = build_segments(speech_segments, subsegments, labels)
        rttm_path = out_dir / f"{file_id}.rttm"
        txt_path = out_dir / f"{file_id}.diarization.txt"
        write_rttm(rttm_path, segments, file_id)
        write_txt(txt_path, segments)

        tr.log_artifact_file(rttm_path)
        tr.log_artifact_file(txt_path)
        tr.log_metric("num_segments", len(segments))
        tr.log_metric("num_speakers_final", n_spk)

        # ── 8. ALIGN words to speakers (post-hoc) ──────────────────────────
        print(f"  [6] Aligning {len(words)} words to {n_spk} speakers...")

        # Midpoint alignment
        words_mid = align_words_to_speakers([dict(w) for w in words], segments)
        turns_mid = words_to_turns(words_mid)

        # Boundary alignment
        words_bnd = align_words_by_boundaries(words, segments, hop_seconds=5.0)
        turns_bnd = words_to_turns(words_bnd)

        tr.log_metric("num_turns_midpoint", len(turns_mid))
        tr.log_metric("num_turns_boundary", len(turns_bnd))

        # Log both alignments
        tr.log_artifact_json(words_mid, "words_midpoint.json", subdir="transcript")
        tr.log_artifact_json(turns_mid, "turns_midpoint.json", subdir="transcript")
        tr.log_artifact_json(words_bnd, "words_boundary.json", subdir="transcript")
        tr.log_artifact_json(turns_bnd, "turns_boundary.json", subdir="transcript")

        # Default pointers
        tr.log_artifact_json(words_mid, "words.json", subdir="transcript")
        tr.log_artifact_json(turns_mid, "turns.json", subdir="transcript")

        for label, t in (("midpoint", turns_mid), ("boundary", turns_bnd)):
            txt = format_transcript_txt(t)
            p = out_dir / f"{file_id}.transcript.{label}.txt"
            p.write_text(txt, encoding="utf-8")
            tr.log_artifact_file(p, subdir="transcript")

        print(f"      midpoint: {len(turns_mid)} turns | "
              f"boundary: {len(turns_bnd)} turns")

        total = time.perf_counter() - t_pipeline
        tr.log_metric("time_total", total)
        tr.log_metric("rtf", total / duration if duration > 0 else 0.0)

        # ── 9. Config dump ──────────────────────────────────────────────────
        config = {
            "pipeline": "transcribe_first",
            "file_id": file_id,
            "input_file": str(in_path),
            "audio_duration": duration,
            "params": {
                "embed": args.embed,
                "estimate": args.estimate,
                "cluster": args.cluster,
                "enhance": not args.no_enhance,
                "win_len": win_len,
                "hop_len": hop_len,
                "num_speakers": args.num_speakers,
                "vad_model": args.vad_model,
                "vad_threshold": args.vad_threshold,
            },
            "timings": timings,
            "results": {
                "num_speakers": n_spk,
                "num_segments": len(segments),
                "num_words": len(words),
                "num_turns_midpoint": len(turns_mid),
                "num_turns_boundary": len(turns_bnd),
                "total_time": total,
                "rtf": total / duration if duration > 0 else 0.0,
            },
        }
        tr.log_artifact_json(config, "config.json")

        # ── 10. DER (optional) ──────────────────────────────────────────────
        if args.reference_rttm:
            ref = Path(args.reference_rttm)
            if not ref.exists():
                print(f"  WARN: reference rttm not found: {ref}")
            else:
                der_result = compute_der(ref, rttm_path, collar=args.der_collar)
                if der_result:
                    print(f"  DER: {der_result['der']*100:.2f}%  "
                          f"(miss={der_result['miss']*100:.2f}%, "
                          f"fa={der_result['false_alarm']*100:.2f}%, "
                          f"conf={der_result['confusion']*100:.2f}%)")
                    tr.log_metric("der", der_result["der"])
                    tr.log_metric("der_miss", der_result["miss"])
                    tr.log_metric("der_false_alarm", der_result["false_alarm"])
                    tr.log_metric("der_confusion", der_result["confusion"])
                    tr.log_artifact_json(der_result, "der.json")
                    tr.log_artifact_file(ref, subdir="reference")

        print()
        print("=" * 78)
        print(f"  RESULT (transcribe-first)")
        print("=" * 78)
        print(f"  Duration       : {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Speakers       : {n_spk}")
        print(f"  Words          : {len(words)}")
        print(f"  Turns (mid)    : {len(turns_mid)}")
        print(f"  Turns (bnd)    : {len(turns_bnd)}")
        print(f"  RTF            : {total/duration:.2f}x")
        print(f"  Total time     : {total:.1f}s")
        print(f"  Peak RAM       : {tr.ram.peak_mb():.0f} MB")
        print(f"  MLflow run_id  : {tr.run_id}")
        print()


if __name__ == "__main__":
    main()
