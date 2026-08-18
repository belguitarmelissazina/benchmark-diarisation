#!/usr/bin/env python3
"""
Diarisation via SpeechBrain ECAPA-TDNN (directement, sans simple_diarizer).

Pipeline :
  1. Conversion audio WAV 16 kHz mono (via imageio-ffmpeg)
  2. Silero VAD → segments de parole
  3. SpeechBrain ECAPA-TDNN → embeddings par segment (fenêtres glissantes)
  4. Clustering : spectral (sc) ou hiérarchique (ahc) via scikit-learn
  5. Sortie RTTM + TXT

Usage :
  python diarize_simple.py -i dicte_audio_3.m4a
  python diarize_simple.py -i dicte_audio_3.m4a --num-speakers 6
  python diarize_simple.py -i dicte_audio_3.m4a --embed ecapa --cluster sc
  python diarize_simple.py -i dicte_audio_3.m4a --cluster ahc --threshold 0.7

Dépendances :
  pip install speechbrain torch torchaudio scikit-learn imageio-ffmpeg silero-vad
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio


# ═══════════════════════════════════════════════════════════════════════════════
#   Audio
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(src: Path, sr: int = 16000) -> Path:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = Path(tempfile.gettempdir()) / f"{src.stem}_16k.wav"
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src),
           "-ac", "1", "-ar", str(sr), str(tmp)]
    subprocess.run(cmd, check=True)
    return tmp


def load_wav(path: Path, sr: int = 16000) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(str(path))
    if orig_sr != sr:
        wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # (samples,)


# ═══════════════════════════════════════════════════════════════════════════════
#   VAD — Silero
# ═══════════════════════════════════════════════════════════════════════════════

def run_silero_vad(wav: torch.Tensor, sr: int = 16000,
                   threshold: float = 0.5, min_speech_ms: int = 250,
                   min_silence_ms: int = 100) -> list:
    """Retourne [(start_s, end_s), ...] via Silero VAD."""
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad', trust_repo=True)
    get_speech_timestamps = utils[0]
    speech_timestamps = get_speech_timestamps(
        wav, model, sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    return [(s['start'] / sr, s['end'] / sr) for s in speech_timestamps]


# ═══════════════════════════════════════════════════════════════════════════════
#   Speaker Embeddings — SpeechBrain ECAPA-TDNN
# ═══════════════════════════════════════════════════════════════════════════════

def load_embedding_model(model_name: str = "ecapa"):
    """Charge le modèle d'embeddings SpeechBrain."""
    import os
    os.environ["SB_FETCH_STRATEGY"] = "copy"  # évite les symlinks sur Windows

    from speechbrain.inference.speaker import SpeakerRecognition
    if model_name == "ecapa":
        source = "speechbrain/spkrec-ecapa-voxceleb"
    else:  # xvec
        source = "speechbrain/spkrec-xvect-voxceleb"
    model = SpeakerRecognition.from_hparams(
        source=source,
        savedir=f"pretrained_models/{model_name}",
    )
    return model


def extract_embeddings(wav: torch.Tensor, speech_segments: list,
                       model, sr: int = 16000,
                       win_len: float = 1.5, hop_len: float = 0.75) -> tuple:
    """
    Extrait un embedding par fenêtre glissante à l'intérieur de chaque
    segment de parole. Retourne (embeddings, segment_info).
    """
    embeddings = []
    seg_info = []  # (start_s, end_s) de chaque fenêtre

    win_samples = int(win_len * sr)
    hop_samples = int(hop_len * sr)

    for seg_start, seg_end in speech_segments:
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)
        seg_wav = wav[start_sample:end_sample]

        if len(seg_wav) < win_samples:
            # Segment plus court que la fenêtre → un seul embedding
            chunk = seg_wav.unsqueeze(0)
            emb = model.encode_batch(chunk).squeeze().detach().numpy()
            embeddings.append(emb)
            seg_info.append((seg_start, seg_end))
        else:
            for offset in range(0, len(seg_wav) - win_samples + 1, hop_samples):
                chunk = seg_wav[offset:offset + win_samples].unsqueeze(0)
                emb = model.encode_batch(chunk).squeeze().detach().numpy()
                embeddings.append(emb)
                t_start = seg_start + offset / sr
                t_end = t_start + win_len
                seg_info.append((t_start, min(t_end, seg_end)))

    return np.array(embeddings), seg_info


# ═══════════════════════════════════════════════════════════════════════════════
#   Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_embeddings(embeddings: np.ndarray, method: str = "sc",
                       num_speakers: int = None,
                       threshold: float = 0.7) -> np.ndarray:
    """Clustering des embeddings. Retourne labels."""
    from sklearn.cluster import SpectralClustering, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    if num_speakers is not None:
        n = num_speakers
    else:
        # Estimation auto via eigenvalues (pour spectral) ou seuil (pour AHC)
        n = estimate_num_speakers(embeddings, max_speakers=10)

    if method == "sc":
        # Matrice d'affinité cosine
        sim = cosine_similarity(embeddings)
        sim = (sim + 1) / 2  # normaliser [0, 1]
        np.fill_diagonal(sim, 1.0)
        clustering = SpectralClustering(
            n_clusters=n, affinity='precomputed',
            random_state=0, n_init=10,
        )
        labels = clustering.fit_predict(sim)
    else:  # ahc
        clustering = AgglomerativeClustering(
            n_clusters=n if num_speakers else None,
            distance_threshold=threshold if num_speakers is None else None,
            metric='cosine', linkage='average',
        )
        labels = clustering.fit_predict(embeddings)

    return labels


def estimate_num_speakers(embeddings: np.ndarray, max_speakers: int = 10) -> int:
    """Estime le nombre de locuteurs via la méthode des eigenvalues."""
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(embeddings)
    sim = (sim + 1) / 2
    np.fill_diagonal(sim, 1.0)

    eigenvalues = np.sort(np.linalg.eigvalsh(sim))[::-1]
    # Chercher le plus grand gap entre eigenvalues consécutives
    max_k = min(max_speakers, len(eigenvalues) - 1)
    if max_k < 2:
        return 2
    gaps = [eigenvalues[i] - eigenvalues[i + 1] for i in range(max_k)]
    n = np.argmax(gaps) + 1
    return max(2, n)


# ═══════════════════════════════════════════════════════════════════════════════
#   Reconstruction des segments diarisés
# ═══════════════════════════════════════════════════════════════════════════════

def build_diarization(seg_info: list, labels: np.ndarray,
                      merge_gap: float = 0.5) -> list:
    """Fusionne les fenêtres en segments continus par locuteur."""
    segments = []
    for (start, end), label in zip(seg_info, labels):
        spk = f"SPEAKER_{int(label):02d}"
        if segments and segments[-1][2] == spk and start - segments[-1][1] < merge_gap:
            segments[-1] = (segments[-1][0], end, spk)
        else:
            segments.append((start, end, spk))
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
#   Sortie
# ═══════════════════════════════════════════════════════════════════════════════

def write_rttm(path: Path, file_id: str, segments: list):
    with open(path, "w", encoding="utf-8") as f:
        for start, end, spk in segments:
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {end - start:.3f} "
                    f"<NA> <NA> {spk} <NA> <NA>\n")


def write_txt(path: Path, segments: list):
    with open(path, "w", encoding="utf-8") as f:
        for start, end, spk in segments:
            f.write(f"[{start:7.2f} - {end:7.2f}]  {spk}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Diarisation via SpeechBrain ECAPA-TDNN")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--embed", default="ecapa", choices=["xvec", "ecapa"],
                    help="Modèle d'embeddings (défaut: ecapa)")
    ap.add_argument("--cluster", default="sc", choices=["sc", "ahc"],
                    help="Méthode de clustering (défaut: sc = spectral)")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Seuil AHC (si num-speakers non fourni, défaut: 0.7)")
    ap.add_argument("--win-len", type=float, default=1.5,
                    help="Fenêtre d'embedding en secondes (défaut: 1.5)")
    ap.add_argument("--hop-len", type=float, default=0.75,
                    help="Hop de la fenêtre en secondes (défaut: 0.75)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    file_id = in_path.stem
    config = f"{args.embed}_{args.cluster}"
    pipeline_name = f"{Path(__file__).stem}_{config}"
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = in_path.parent / "outputs" / pipeline_name / file_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DIARISATION SPEECHBRAIN — {in_path.name}")
    print(f"  embed={args.embed}  cluster={args.cluster}  "
          f"num_speakers={args.num_speakers}  threshold={args.threshold}")
    print(f"  win={args.win_len}s  hop={args.hop_len}s")
    print("=" * 78)

    # ── Conversion WAV ──
    t0 = time.perf_counter()
    wav_path = convert_to_wav(in_path)
    wav = load_wav(wav_path)
    duration = len(wav) / 16000
    print(f"  [1/4] Audio       : {duration:.1f}s ({duration/60:.1f} min)")

    # ── VAD ──
    t1 = time.perf_counter()
    speech = run_silero_vad(wav)
    speech_total = sum(e - s for s, e in speech)
    print(f"  [2/4] Silero VAD  : {len(speech)} segments de parole "
          f"({speech_total:.1f}s) — {time.perf_counter()-t1:.1f}s")

    # ── Embeddings ──
    t1 = time.perf_counter()
    print(f"  [3/4] Embeddings {args.embed.upper()}...", flush=True)
    model = load_embedding_model(args.embed)
    embeddings, seg_info = extract_embeddings(
        wav, speech, model, win_len=args.win_len, hop_len=args.hop_len)
    print(f"        {len(embeddings)} embeddings — {time.perf_counter()-t1:.1f}s")

    # ── Clustering ──
    t1 = time.perf_counter()
    labels = cluster_embeddings(
        embeddings, method=args.cluster,
        num_speakers=args.num_speakers,
        threshold=args.threshold)
    n_spk = len(set(labels))
    print(f"  [4/4] Clustering  : {n_spk} locuteurs — {time.perf_counter()-t1:.1f}s")

    # ── Sortie ──
    diarization = build_diarization(seg_info, labels)
    rttm_path = out_dir / f"{file_id}.rttm"
    txt_path = out_dir / f"{file_id}.diarization.txt"
    write_rttm(rttm_path, file_id, diarization)
    write_txt(txt_path, diarization)

    total = time.perf_counter() - t0
    print()
    print("=" * 78)
    print(f"  RÉSULTAT")
    print("=" * 78)
    print(f"  Config             : {config}")
    print(f"  Locuteurs détectés : {n_spk}")
    print(f"  Segments           : {len(diarization)}")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  → {rttm_path}")
    print(f"  → {txt_path}")
    print()
    print("  Aperçu :")
    for start, end, spk in diarization[:15]:
        print(f"    [{start:7.2f} - {end:7.2f}]  {spk}")
    if len(diarization) > 15:
        print(f"    ... ({len(diarization) - 15} de plus)")
    print()


if __name__ == "__main__":
    main()
