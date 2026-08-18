#!/usr/bin/env python3
"""Run pyannote-onnx-extended (full ONNX pipeline) on a single file.

This is a standalone experiment runner that uses the pyannote-onnx-extended
library for diarization and tracks results with MLflow via our Tracker.

The library downloads segmentation + embedding ONNX models from HuggingFace
automatically (onnx-community repos, no HF_TOKEN needed).

Usage:
  python -m diar_pipeline.run_pyannote_onnx -i audio.wav --reference-rttm ref.rttm
  python -m diar_pipeline.run_pyannote_onnx -i audio.wav --num-speakers 4
"""

from __future__ import annotations
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import sys
import time
from pathlib import Path

from .audio import convert_to_wav, get_audio_duration
from .tracking import Tracker, compute_der


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pyannote-onnx-extended diarization with MLflow tracking")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    # pyannote-onnx-extended segmentation params
    ap.add_argument("--onset", type=float, default=0.5,
                    help="Onset threshold for speaker activation")
    ap.add_argument("--offset", type=float, default=0.5,
                    help="Offset threshold for speaker deactivation")
    ap.add_argument("--min-duration-on", type=float, default=0.5,
                    help="Minimum speech duration (seconds)")
    ap.add_argument("--min-duration-off", type=float, default=0.3,
                    help="Minimum silence gap to split segments (seconds)")
    # Transcription (optional)
    ap.add_argument("--transcribe", action="store_true",
                    help="Run ASR (sherpa-onnx kroko Zipformer) after diarization")
    ap.add_argument("--model-dir", default=None,
                    help="Path to sherpa-onnx model dir (default: auto-detect)")
    # Evaluation
    ap.add_argument("--reference-rttm", default=None)
    ap.add_argument("--der-collar", type=float, default=0.25)
    # MLflow
    ap.add_argument("--experiment", default="pyannote_onnx",
                    help="MLflow experiment name")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    file_id = in_path.stem
    run_name = args.run_name or f"pyannote_onnx__{file_id}"

    out_dir = (Path(args.output_dir) if args.output_dir
               else in_path.parent / "outputs" / "pyannote_onnx" / file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  PYANNOTE-ONNX DIARIZATION — {in_path.name}")
    print(f"  onset={args.onset} | offset={args.offset}"
          f" | min_on={args.min_duration_on}s | min_off={args.min_duration_off}s"
          f" | num_speakers={args.num_speakers or 'auto'}")
    print("=" * 78)

    tags = {
        "file_id": file_id,
        "pipeline": "pyannote_onnx_extended",
    }

    with Tracker(experiment=args.experiment, run_name=run_name,
                 tracking_uri=args.tracking_uri, tags=tags) as tr:

        tr.log_params({
            "input_file": str(in_path),
            "pipeline": "pyannote_onnx_extended",
            "onset": args.onset,
            "offset": args.offset,
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
            "num_speakers": args.num_speakers,
            "transcribe": args.transcribe,
        })
        print(f"  MLflow run_id: {tr.run_id}")

        timings: dict[str, float] = {}
        t_pipeline = time.perf_counter()

        # ── 1. Audio conversion ─────────────────────────────────────────────
        t0 = time.perf_counter()
        wav_path = convert_to_wav(in_path)
        duration = get_audio_duration(str(wav_path))
        timings["time_conversion"] = time.perf_counter() - t0
        tr.ram.mark("conversion")
        tr.log_metric("audio_duration_sec", duration)
        tr.log_metric("time_conversion", timings["time_conversion"])
        tr.log_artifact_file(wav_path, subdir="audio")
        print(f"  [1] audio -> {wav_path.name} ({duration:.1f}s)")

        # ── 2. Load pipeline (downloads ONNX models on first run) ───────────
        t1 = time.perf_counter()
        from onnx_pyannote import ONNXSpeakerDiarization
        pipeline = ONNXSpeakerDiarization(
            model_name="speaker-diarization-3.1",
            providers=["CPUExecutionProvider"],
            onset=args.onset,
            offset=args.offset,
            min_duration_on=args.min_duration_on,
            min_duration_off=args.min_duration_off,
        )
        timings["time_model_load"] = time.perf_counter() - t1
        tr.ram.mark("model_load")
        tr.log_metric("time_model_load", timings["time_model_load"])
        print(f"  [2] model loaded — {timings['time_model_load']:.1f}s")

        # ── 3. Run diarization ──────────────────────────────────────────────
        t2 = time.perf_counter()
        annotation = pipeline(str(wav_path), num_speakers=args.num_speakers)
        timings["time_diarization"] = time.perf_counter() - t2
        tr.ram.mark("diarization")
        tr.log_metric("time_diarization", timings["time_diarization"])

        # Count speakers and segments
        speakers = set()
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speakers.add(speaker)
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        n_spk = len(speakers)
        tr.log_metric("num_speakers_final", n_spk)
        tr.log_metric("num_segments", len(segments))
        print(f"  [3] diarization: {n_spk} speakers, {len(segments)} segments"
              f" — {timings['time_diarization']:.1f}s")

        # ── 4. Write RTTM ──────────────────────────────────────────────────
        rttm_path = out_dir / f"{file_id}.rttm"
        with open(rttm_path, "w", encoding="utf-8") as f:
            for seg in segments:
                dur = seg["end"] - seg["start"]
                f.write(f"SPEAKER {file_id} 1 {seg['start']:.3f} {dur:.3f}"
                        f" <NA> <NA> {seg['speaker']} <NA> <NA>\n")
        tr.log_artifact_file(rttm_path)

        # Write human-readable txt
        txt_path = out_dir / f"{file_id}.diarization.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg['start']:8.2f} - {seg['end']:8.2f}]  "
                        f"{seg['speaker']}\n")
        tr.log_artifact_file(txt_path)

        # Log segments JSON
        tr.log_artifact_json(segments, "segments.json")

        # ── 5. Transcription (optional) ─────────────────────────────────────
        words = None
        turns = None
        if args.transcribe:
            from .transcription import (
                load_audio_pcm, transcribe as run_asr,
                align_words_to_speakers, words_to_turns,
                format_transcript_txt, DEFAULT_MODEL_DIR,
            )
            # Convert our segments list to the format align_words_to_speakers expects
            diar_segments = [
                {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                for s in segments
            ]
            model_dir = Path(args.model_dir) if args.model_dir else DEFAULT_MODEL_DIR
            t_asr = time.perf_counter()
            print(f"  [4] Transcription (sherpa-onnx)...")
            audio_pcm = load_audio_pcm(in_path)
            words = run_asr(audio_pcm, model_dir=model_dir)
            timings["time_transcription"] = time.perf_counter() - t_asr
            tr.ram.mark("transcription")
            tr.log_metric("time_transcription", timings["time_transcription"])
            if duration > 0:
                tr.log_metric("wpm", len(words) / (duration / 60))

            from types import SimpleNamespace
            diar_objs = [SimpleNamespace(**s) for s in diar_segments]
            words_aligned = align_words_to_speakers(
                [dict(w) for w in words], diar_objs)
            turns = words_to_turns(words_aligned)

            tr.log_metric("num_words", len(words))
            tr.log_metric("num_turns", len(turns))
            tr.log_artifact_json(words_aligned, "words.json", subdir="transcript")
            tr.log_artifact_json(turns, "turns.json", subdir="transcript")

            txt = format_transcript_txt(turns)
            p = out_dir / f"{file_id}.transcript.txt"
            p.write_text(txt, encoding="utf-8")
            tr.log_artifact_file(p, subdir="transcript")

            print(f"      -> {len(words)} words, {len(turns)} turns"
                  f" — {timings['time_transcription']:.1f}s")

        total = time.perf_counter() - t_pipeline
        tr.log_metric("time_total", total)
        tr.log_metric("rtf", total / duration if duration > 0 else 0.0)

        # ── 6. Config dump ──────────────────────────────────────────────────
        config = {
            "pipeline": "pyannote_onnx_extended",
            "file_id": file_id,
            "input_file": str(in_path),
            "audio_duration": duration,
            "params": {
                "onset": args.onset,
                "offset": args.offset,
                "min_duration_on": args.min_duration_on,
                "min_duration_off": args.min_duration_off,
                "num_speakers": args.num_speakers,
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

        # ── 7. DER (optional) ───────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
