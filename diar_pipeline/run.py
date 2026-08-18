#!/usr/bin/env python3
"""CLI orchestrator — modular diarization pipeline with MLflow tracking.

Every run logs to MLflow:
  - Parameters: embed, estimate, cluster, enhance, win_len, hop_len, vbx_*, ...
  - Metrics per step: time_vad, time_embeddings, time_clustering, time_vbx,
                      ram_*_mb (snapshot), peak_ram_mb, mean_ram_mb,
                      num_vad_segments, speech_ratio, num_embeddings,
                      num_speakers_final, silhouette_final,
                      der / miss / false_alarm / confusion (if --reference-rttm)
  - Artifacts:
      vad/vad_segments.json
      embeddings/embeddings.npy
      embeddings/chunk_times.json
      similarity/sim_matrix_raw.npy
      similarity/sim_matrix_enhanced.npy (if enhance on)
      clustering/labels_final.npy  (+ labels_pre_vbx.npy if refine-vbx)
      diarization.rttm
      diarization.txt
      ram_timeseries.json
      config.json

Browse runs with:  mlflow ui

Usage:
  python -m diar_pipeline.run -i meeting.m4a
  python -m diar_pipeline.run -i meeting.m4a --embed campplus --estimate nmesc
  python -m diar_pipeline.run -i meeting.m4a --refine-vbx --reference-rttm gt.rttm
"""

from __future__ import annotations
import os as _os
# Force single-threaded BLAS BEFORE numpy/scipy import — otherwise ARPACK /
# eigh results vary run-to-run under multithreaded MKL/OpenBLAS, which caused
# NMESC speaker-count flips (k=3↔5) and DER swings on identical inputs.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import argparse
import json
import sys
import tempfile
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


def build_pipeline_name(args, win_len: float, hop_len: float) -> str:
    """Encode ALL parameters in the run name for benchmarking."""
    parts = ["diar_pipeline", args.vad_model, args.embed, args.estimate, args.cluster]
    if args.cluster == "sc" and not args.no_enhance:
        parts.append("enhanced")
    parts.append(f"w{win_len:.1f}")
    parts.append(f"h{hop_len:.1f}")
    if args.refine_vbx:
        parts.append(f"vbx_Fa{args.vbx_Fa}_Fb{args.vbx_Fb}")
    if args.num_speakers is not None:
        parts.append(f"nspk{args.num_speakers}")
    else:
        parts.append(f"auto_{args.min_speakers}-{args.max_speakers}")
    return "__".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Modular speaker diarization pipeline with MLflow tracking")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=20)
    ap.add_argument("--embed", choices=["resnet34", "campplus", "ecapa"],
                    default="resnet34")
    ap.add_argument("--estimate", choices=["gmm_bic", "nmesc"], default="gmm_bic")
    ap.add_argument("--cluster", choices=["sc", "ahc", "meanshift", "ahc_threshold", "cosine_greedy"],
                    default="sc")
    ap.add_argument("--no-enhance", action="store_true",
                    help="Disable sim enhancement (SC only)")
    ap.add_argument("--silhouette-refine", action="store_true")
    ap.add_argument("--win-len", type=float, default=None)
    ap.add_argument("--hop-len", type=float, default=None)
    # VAD
    ap.add_argument("--vad-model", choices=["silero", "pyannote"], default="silero",
                    help="VAD backend (pyannote needs HF_TOKEN env var)")
    ap.add_argument("--vad-threshold", type=float, default=0.45,
                    help="Speech probability threshold (higher=stricter)")
    ap.add_argument("--vad-min-speech-ms", type=int, default=200,
                    help="Drop speech chunks shorter than this")
    ap.add_argument("--vad-min-silence-ms", type=int, default=50,
                    help="Merge adjacent speech if silence gap shorter than this")
    ap.add_argument("--vad-pad-ms", type=int, default=20,
                    help="Pad speech segments by this much on each side (Silero only)")
    ap.add_argument("--ahc-threshold", type=float, default=0.5,
                    help="Cosine distance threshold for ahc_threshold clustering")
    ap.add_argument("--greedy-threshold", type=float, default=0.5,
                    help="Cosine similarity threshold for cosine_greedy clustering")
    ap.add_argument("--ahc-percentile", type=float, default=None,
                    help="If set (0-100), derive ahc_threshold from this percentile "
                         "of pairwise cosine distances (self-calibrates per audio)")
    ap.add_argument("--greedy-percentile", type=float, default=None,
                    help="If set (0-100), derive greedy_threshold from this percentile "
                         "of pairwise cosine similarities (self-calibrates per audio)")
    ap.add_argument("--min-cluster-size", type=int, default=1,
                    help="Drop clusters with fewer than N embeddings; reassign to nearest "
                         "(AHC threshold methods). Reduces over-segmentation.")
    ap.add_argument("--merge-threshold", type=float, default=None,
                    help="Post-AHC: iteratively merge clusters whose centroid cosine "
                         "distance is below this (e.g. 0.5). Reduces over-segmentation.")
    ap.add_argument("--refine-vbx", action="store_true")
    ap.add_argument("--vbx-Fa", type=float, default=0.05)
    ap.add_argument("--vbx-Fb", type=float, default=5.0)
    # MLflow
    ap.add_argument("--experiment", default="diarization",
                    help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow tracking URI (default: ./mlruns)")
    ap.add_argument("--run-name", default=None,
                    help="Override auto-generated run name")
    # Transcription (optional)
    ap.add_argument("--transcribe", action="store_true",
                    help="Run ASR (sherpa-onnx kroko Zipformer) after diarization")
    ap.add_argument("--model-dir", default=None,
                    help="Path to sherpa-onnx model dir (default: auto-detect)")
    # Evaluation
    ap.add_argument("--reference-rttm", default=None,
                    help="Ground-truth RTTM for DER computation")
    ap.add_argument("--der-collar", type=float, default=0.25)
    # Legacy diagnostics (optional, still supported)
    ap.add_argument("--diag", action="store_true",
                    help="Also render legacy static diagnostics via diagnostics.py")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    file_id = in_path.stem
    win_len = args.win_len if args.win_len else EMBEDDING_WINDOW
    hop_len = args.hop_len if args.hop_len else EMBEDDING_STEP

    pipeline_name = build_pipeline_name(args, win_len, hop_len)
    run_name = args.run_name or f"{pipeline_name}__{file_id}"

    out_dir = (Path(args.output_dir) if args.output_dir
               else in_path.parent / "outputs" / pipeline_name / file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DIARIZATION — {in_path.name}")
    print(f"  embed={args.embed} | estimate={args.estimate} | cluster={args.cluster}"
          f" | enhance={'OFF' if args.no_enhance else 'ON'}"
          f" | vbx={'ON' if args.refine_vbx else 'OFF'}"
          f" | window={win_len}s/{hop_len}s")
    print("=" * 78)

    tags = {
        "file_id": file_id,
        "pipeline": pipeline_name,
        "embed": args.embed,
        "cluster": args.cluster,
        "estimate": args.estimate,
    }

    with Tracker(experiment=args.experiment, run_name=run_name,
                 tracking_uri=args.tracking_uri, tags=tags) as tr:

        # ── Log parameters ─────────────────────────────────────────────────
        tr.log_params({
            "input_file": str(in_path),
            "embed": args.embed,
            "estimate": args.estimate,
            "cluster": args.cluster,
            "enhance": not args.no_enhance,
            "silhouette_refine": args.silhouette_refine,
            "win_len": win_len,
            "hop_len": hop_len,
            "refine_vbx": args.refine_vbx,
            "vbx_Fa": args.vbx_Fa,
            "vbx_Fb": args.vbx_Fb,
            "num_speakers": args.num_speakers,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "ahc_threshold": args.ahc_threshold,
            "greedy_threshold": args.greedy_threshold,
            "ahc_percentile": args.ahc_percentile,
            "greedy_percentile": args.greedy_percentile,
            "min_cluster_size": args.min_cluster_size,
            "merge_threshold": args.merge_threshold,
            "vad_model": args.vad_model,
            "vad_threshold": args.vad_threshold,
            "vad_min_speech_ms": args.vad_min_speech_ms,
            "vad_min_silence_ms": args.vad_min_silence_ms,
            "vad_pad_ms": args.vad_pad_ms,
            "transcribe": args.transcribe,
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
        # Log the 16 kHz mono WAV as an artifact so the backend always has
        # a universally-decodable source for waveform rendering + playback.
        tr.log_artifact_file(wav_path, subdir="audio")
        print(f"  [1] audio -> {wav_path.name} ({duration:.1f}s)")

        # ── 2. VAD ──────────────────────────────────────────────────────────
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
        print(f"  [2] VAD: {len(speech_segments)} seg — {timings['time_vad']:.1f}s")
        if not speech_segments:
            print("  ERROR: no speech detected")
            sys.exit(1)

        # ── 3. Embeddings ───────────────────────────────────────────────────
        t2 = time.perf_counter()
        embeddings, subsegments = extract_embeddings(
            str(wav_path), speech_segments,
            win_len=win_len, hop_len=hop_len, embed_model=args.embed)
        timings["time_embeddings"] = time.perf_counter() - t2
        tr.ram.mark("embeddings")
        tr.log_metric("time_embeddings", timings["time_embeddings"])
        log_embedding_artifacts(tr, embeddings, subsegments, args.embed)
        model_label = {
            "campplus": "CAM++",
            "ecapa": "ECAPA-TDNN",
            "resnet34": "ResNet34-LM",
        }[args.embed]
        print(f"  [3] embeddings ({model_label}): {embeddings.shape[0]} x "
              f"{embeddings.shape[1]} — {timings['time_embeddings']:.1f}s")
        if len(embeddings) == 0:
            print("  ERROR: no embeddings extracted")
            sys.exit(1)

        # ── 3b. Similarity matrices (standalone artifacts) ──────────────────
        t_sim = time.perf_counter()
        log_sim_artifacts(tr, embeddings, enhance=not args.no_enhance)
        timings["time_similarity"] = time.perf_counter() - t_sim
        tr.log_metric("time_similarity", timings["time_similarity"])

        # ── 4. Clustering ───────────────────────────────────────────────────
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
            ahc_threshold=args.ahc_threshold,
            greedy_threshold=args.greedy_threshold,
            ahc_percentile=args.ahc_percentile,
            greedy_percentile=args.greedy_percentile,
            min_cluster_size=args.min_cluster_size,
            merge_threshold=args.merge_threshold,
        )
        timings["time_clustering"] = time.perf_counter() - t3
        tr.ram.mark("clustering")
        tr.log_metric("time_clustering", timings["time_clustering"])
        n_spk = len(set(labels.tolist()))
        print(f"  [4] clustering: {n_spk} speakers — {timings['time_clustering']:.1f}s")
        if details:
            print(f"      estimation: {details.method} -> k={details.best_k}")

        # Log pre-VBx labels if we are going to run VBx (for before/after compare)
        pre_vbx_stage = "pre_vbx" if args.refine_vbx else "final"
        log_clustering_artifacts(tr, labels, embeddings, stage=pre_vbx_stage)

        # ── 5. VBx refinement ───────────────────────────────────────────────
        if args.refine_vbx:
            t4 = time.perf_counter()
            print(f"  [5] VBx (Fa={args.vbx_Fa}, Fb={args.vbx_Fb})...")
            labels = refine_vbx(
                embeddings, labels, subsegments,
                Fa=args.vbx_Fa, Fb=args.vbx_Fb)
            timings["time_vbx"] = time.perf_counter() - t4
            tr.ram.mark("vbx")
            tr.log_metric("time_vbx", timings["time_vbx"])
            n_spk = len(set(labels.tolist()))
            print(f"      -> {n_spk} speakers — {timings['time_vbx']:.1f}s")
            log_clustering_artifacts(tr, labels, embeddings, stage="final")

        # ── 6. Assemble + write ─────────────────────────────────────────────
        segments = build_segments(speech_segments, subsegments, labels)
        rttm_path = out_dir / f"{file_id}.rttm"
        txt_path = out_dir / f"{file_id}.diarization.txt"
        write_rttm(rttm_path, segments, file_id)
        write_txt(txt_path, segments)

        tr.log_artifact_file(rttm_path)
        tr.log_artifact_file(txt_path)
        tr.log_metric("num_segments", len(segments))
        tr.log_metric("num_speakers_final", n_spk)

        # ── 7. Transcription (optional) ─────────────────────────────────────
        words = None
        turns = None
        if args.transcribe:
            from .transcription import (
                load_audio_pcm, transcribe as run_asr,
                align_words_to_speakers, align_words_by_boundaries,
                words_to_turns, format_transcript_txt,
                DEFAULT_MODEL_DIR,
            )
            model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
            t_asr = time.perf_counter()
            print(f"  [6] Transcription (sherpa-onnx)...")
            audio_pcm = load_audio_pcm(in_path)
            words = run_asr(audio_pcm, model_dir=model_dir)
            timings["time_transcription"] = time.perf_counter() - t_asr
            tr.ram.mark("transcription")
            tr.log_metric("time_transcription", timings["time_transcription"])
            if duration > 0:
                tr.log_metric("wpm", len(words) / (duration / 60))

            # Two alignment modes from the SAME transcription:
            #  (a) midpoint  — per-word speaker = speaker at word midpoint
            #  (b) boundary  — transcribe-first; insert speaker changes
            #                  at the nearest punctuation to each diarization
            #                  speaker-change time
            words_mid = align_words_to_speakers([dict(w) for w in words], segments)
            turns_mid = words_to_turns(words_mid)
            words_bnd = align_words_by_boundaries(words, segments, hop_seconds=5.0)
            turns_bnd = words_to_turns(words_bnd)

            tr.log_metric("num_words", len(words))
            tr.log_metric("num_turns_midpoint", len(turns_mid))
            tr.log_metric("num_turns_boundary", len(turns_bnd))

            # Log both alignments
            tr.log_artifact_json(words_mid, "words_midpoint.json", subdir="transcript")
            tr.log_artifact_json(turns_mid, "turns_midpoint.json", subdir="transcript")
            tr.log_artifact_json(words_bnd, "words_boundary.json", subdir="transcript")
            tr.log_artifact_json(turns_bnd, "turns_boundary.json", subdir="transcript")

            # Default pointers (for backward compat with frontend/backend)
            tr.log_artifact_json(words_mid, "words.json", subdir="transcript")
            tr.log_artifact_json(turns_mid, "turns.json", subdir="transcript")

            for label, t in (("midpoint", turns_mid), ("boundary", turns_bnd)):
                txt = format_transcript_txt(t)
                p = out_dir / f"{file_id}.transcript.{label}.txt"
                p.write_text(txt, encoding="utf-8")
                tr.log_artifact_file(p, subdir="transcript")

            # Per-speaker word counts (based on midpoint)
            spk_counts = {}
            for w in words_mid:
                spk_counts[w["speaker"]] = spk_counts.get(w["speaker"], 0) + 1
            tr.log_artifact_json(spk_counts, "words_per_speaker.json",
                                 subdir="transcript")
            words = words_mid
            turns = turns_mid
            print(f"      -> {len(words)} words | "
                  f"midpoint: {len(turns_mid)} turns | "
                  f"boundary: {len(turns_bnd)} turns "
                  f"-- {timings['time_transcription']:.1f}s")

        total = time.perf_counter() - t_pipeline
        tr.log_metric("time_total", total)
        tr.log_metric("rtf", total / duration if duration > 0 else 0.0)

        # ── 8. Config dump (for audit) ──────────────────────────────────────
        config = {
            "pipeline_name": pipeline_name,
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
                "refine_vbx": args.refine_vbx,
                "vbx_Fa": args.vbx_Fa,
                "vbx_Fb": args.vbx_Fb,
                "num_speakers": args.num_speakers,
                "min_speakers": args.min_speakers,
                "max_speakers": args.max_speakers,
            },
            "timings": timings,
            "results": {
                "num_speakers": n_spk,
                "num_segments": len(segments),
                "transcribed": args.transcribe,
                "num_words": len(words) if words else 0,
                "num_turns": len(turns) if turns else 0,
                "total_time": total,
                "rtf": total / duration if duration > 0 else 0.0,
            },
        }
        tr.log_artifact_json(config, "config.json")

        # ── 8. DER (optional) ───────────────────────────────────────────────
        if args.reference_rttm:
            ref = Path(args.reference_rttm)
            if not ref.exists():
                print(f"  WARN: reference rttm not found: {ref}")
            else:
                ref_speakers = set()
                for line in ref.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 8:
                        ref_speakers.add(parts[7])
                n_ref = len(ref_speakers)
                tr.log_metric("num_speakers_ref", n_ref)
                print(f"  Reference: {n_ref} speakers")
                der_result = compute_der(ref, rttm_path,
                                         collar=args.der_collar)
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

        # ── 9. Legacy diagnostics (opt-in) ──────────────────────────────────
        if args.diag:
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from diagnostics import Diagnostics  # type: ignore
                diag = Diagnostics(out_dir, file_id)
                diag.plot_vad(str(wav_path), speech_segments)
                diag.plot_affinity(embeddings, labels)
                diag.plot_embeddings(embeddings, subsegments, labels)
                diag.plot_timeline(segments, duration)
                diag.plot_timing(timings, total, duration)
                diag.plot_ram()
                diag.generate_report(pipeline_name, {
                    "File": str(in_path),
                    "Duration": f"{duration:.1f}s",
                    "Detected": str(n_spk),
                })
                # push diagnostics folder to mlflow as well
                if (out_dir / "diagnostics").exists():
                    tr.log_artifact_file(out_dir / "diagnostics",
                                         subdir="legacy_diagnostics")
            except Exception as e:
                print(f"  (legacy diag failed: {e})")

        print()
        print("=" * 78)
        print(f"  RESULT")
        print("=" * 78)
        print(f"  Duration       : {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Speakers       : {n_spk}")
        print(f"  Segments       : {len(segments)}")
        print(f"  RTF            : {total/duration:.2f}x")
        print(f"  Total time     : {total:.1f}s")
        print(f"  Peak RAM       : {tr.ram.peak_mb():.0f} MB")
        print(f"  MLflow run_id  : {tr.run_id}")
        print(f"  -> {rttm_path}")
        print(f"  -> {txt_path}")
        print()
        print("  Browse runs: mlflow ui")
        print()


if __name__ == "__main__":
    main()
