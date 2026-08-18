#!/usr/bin/env python3
"""Test timestamps kroko — accès via rec.tokens / rec.timestamps / rec.get_result_as_json_string"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx


def load_audio_pcm_float(path: Path, sr: int = 16000) -> np.ndarray:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--model-dir", default="sherpa-onnx-streaming-zipformer-fr-kroko")
    args = ap.parse_args()

    sr = 16000
    audio = load_audio_pcm_float(Path(args.input), sr=sr)
    audio = audio[: int(args.seconds * sr)]
    print(f"Audio chargé : {len(audio)/sr:.1f}s")

    md = Path(args.model_dir)
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
    print("Modèle kroko chargé.")

    stream = rec.create_stream()
    chunk = int(0.5 * sr)
    for i in range(0, len(audio), chunk):
        stream.accept_waveform(sr, audio[i:i + chunk])
        while rec.is_ready(stream):
            rec.decode_stream(stream)

    # Flush
    tail = np.zeros(int(0.5 * sr), dtype=np.float32)
    stream.accept_waveform(sr, tail)
    while rec.is_ready(stream):
        rec.decode_stream(stream)

    print()
    print("=" * 70)

    # ── Test 1 : rec.tokens(stream) et rec.timestamps(stream)
    print("TEST 1 — rec.tokens(stream) / rec.timestamps(stream)")
    try:
        tokens = rec.tokens(stream)
        print(f"  rec.tokens type={type(tokens).__name__}")
        if isinstance(tokens, (list, tuple)):
            print(f"  len={len(tokens)}, premiers={tokens[:20]}")
        else:
            print(f"  valeur={tokens!r}"[:200])
    except Exception as e:
        print(f"  rec.tokens(stream) → ERROR: {e}")

    try:
        timestamps = rec.timestamps(stream)
        print(f"  rec.timestamps type={type(timestamps).__name__}")
        if isinstance(timestamps, (list, tuple)):
            print(f"  len={len(timestamps)}, premiers={timestamps[:20]}")
        else:
            print(f"  valeur={timestamps!r}"[:200])
    except Exception as e:
        print(f"  rec.timestamps(stream) → ERROR: {e}")

    # ── Test 2 : rec.get_result_as_json_string(stream)
    print()
    print("TEST 2 — rec.get_result_as_json_string(stream)")
    try:
        j = rec.get_result_as_json_string(stream)
        print(f"  type = {type(j).__name__}")
        d = json.loads(j)
        print(f"  clés = {list(d.keys())}")
        for k in ("text", "tokens", "timestamps", "start_time"):
            if k in d:
                v = d[k]
                if isinstance(v, list):
                    print(f"  {k} (len={len(v)}) = {v[:15]}")
                else:
                    print(f"  {k} = {v!r}"[:120])
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Test 3 : rec.get_result_all(stream)
    print()
    print("TEST 3 — rec.get_result_all(stream)")
    try:
        r = rec.get_result_all(stream)
        print(f"  type = {type(r).__name__}")
        if isinstance(r, dict):
            print(f"  clés = {list(r.keys())}")
            for k in ("tokens", "timestamps"):
                if k in r and r[k]:
                    print(f"  {k} (len={len(r[k])}) = {r[k][:15]}")
        elif isinstance(r, str):
            try:
                d = json.loads(r)
                print(f"  clés JSON = {list(d.keys())}")
            except Exception:
                print(f"  valeur = {r[:200]}")
        else:
            attrs = [a for a in dir(r) if not a.startswith('_')]
            print(f"  dir = {attrs}")
            for attr in ("text", "tokens", "timestamps"):
                if hasattr(r, attr):
                    v = getattr(r, attr)
                    print(f"  {attr} = {v!r}"[:150] if not isinstance(v, (list, tuple))
                          else f"  {attr} (len={len(v)}) = {v[:15]}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── VERDICT
    print()
    print("=" * 70)
    print("VERDICT :")
    # Try to show paired tokens+timestamps
    try:
        j = rec.get_result_as_json_string(stream)
        d = json.loads(j)
        if d.get("timestamps") and d.get("tokens"):
            toks, ts = d["tokens"], d["timestamps"]
            print(f"  ✓ {len(ts)} timestamps / {len(toks)} tokens via JSON")
            print()
            for tok, t in list(zip(toks, ts))[:30]:
                print(f"    {t:6.2f}s  {tok}")
            return
    except Exception:
        pass
    try:
        toks = rec.tokens(stream)
        ts = rec.timestamps(stream)
        if ts and toks:
            print(f"  ✓ {len(ts)} timestamps / {len(toks)} tokens via rec.tokens/timestamps")
            print()
            for tok, t in list(zip(toks, ts))[:30]:
                print(f"    {t:6.2f}s  {tok}")
            return
    except Exception:
        pass
    print("  ✗ Pas de timestamps exploitables.")


if __name__ == "__main__":
    main()
