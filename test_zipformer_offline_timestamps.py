#!/usr/bin/env python3
"""Test timestamps sur le modèle offline sherpa-onnx-zipformer-fr-2023-10-02."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
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
    ap.add_argument("--model-dir", default="sherpa-onnx-zipformer-fr-2023-10-02")
    args = ap.parse_args()

    sr = 16000
    audio = load_audio(Path(args.input), sr=sr)[: int(args.seconds * sr)]
    print(f"Audio chargé : {len(audio)/sr:.1f}s")

    md = Path(args.model_dir)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(md / "encoder-epoch-29-avg-9-with-averaged-model.onnx"),
        decoder=str(md / "decoder-epoch-29-avg-9-with-averaged-model.onnx"),
        joiner=str(md / "joiner-epoch-29-avg-9-with-averaged-model.onnx"),
        tokens=str(md / "tokens.txt"),
        num_threads=4,
        sample_rate=sr,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    print("Modèle offline chargé.")

    s = rec.create_stream()
    s.accept_waveform(sr, audio)
    rec.decode_stream(s)
    r = s.result

    print()
    print("=" * 60)
    print(f"type     = {type(r).__name__}")
    print(f"repr     = {r!r}")
    print(f"dir      = {[a for a in dir(r) if not a.startswith('_')]}")
    print("=" * 60)
    for attr in ("text", "tokens", "timestamps", "words"):
        if hasattr(r, attr):
            v = getattr(r, attr)
            if isinstance(v, (list, tuple)):
                print(f"  {attr:10s} (len={len(v)}) = {v[:20]}")
            else:
                print(f"  {attr:10s} = {v!r}")

    print()
    ts = getattr(r, "timestamps", None)
    tk = getattr(r, "tokens", None)
    if ts:
        print(f"  ✓ {len(ts)} timestamps / {len(tk)} tokens")
        print()
        print("  Aperçu :")
        for tok, t in list(zip(tk, ts))[:25]:
            print(f"    {t:6.2f}s  {tok}")
    else:
        print("  ✗ Pas de timestamps même sur l'offline. Étrange.")


if __name__ == "__main__":
    main()
