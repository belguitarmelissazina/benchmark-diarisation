#!/usr/bin/env python3
"""
Diarisation classique BIC + GMM (Pipeline 1) — 100% Python, sans deep learning,
sans modèle pré-entraîné. Conçu pour CPU / 8 Go RAM / enregistrements Teams.

Pipeline :
  1. Chargement WAV 16 kHz mono (utiliser preprocess_audio.py en amont)
  2. VAD WebRTC          → segments de parole
  3. MFCC (+ Δ + ΔΔ)     → features par frame (10 ms hop)
  4. BIC segmentation    → détection des changements de locuteur
  5. GMM par segment     → modèle acoustique local
  6. Distance ΔBIC       → matrice de similarité entre segments
  7. AHC                 → clustering en N locuteurs
  8. Sortie RTTM + TXT lisible

Usage :
  python diarize_bic_gmm.py -i meeting.wav
  python diarize_bic_gmm.py -i meeting.wav --num-speakers 3
  python diarize_bic_gmm.py -i meeting.wav --bic-threshold 1.0

Dépendances :
  pip install webrtcvad librosa scikit-learn soundfile numpy
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad
from sklearn.mixture import GaussianMixture


def load_audio_ffmpeg(path: Path, sr: int = 16000) -> np.ndarray:
    """Décode n'importe quel format audio (m4a, mp3, wav...) en float32 mono
    via le ffmpeg embarqué par imageio-ffmpeg (pipeline 100% autonome)."""
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


# ═══════════════════════════════════════════════════════════════════════════════
#   1. VAD — WebRTC
# ═══════════════════════════════════════════════════════════════════════════════

def run_vad(audio: np.ndarray, sr: int, aggressiveness: int = 2,
            frame_ms: int = 30, min_speech_ms: int = 300,
            merge_gap_ms: int = 200) -> list:
    """
    VAD WebRTC. Retourne une liste de (start_s, end_s) de parole.
    aggressiveness : 0 (permissif) → 3 (strict). 2 = bon défaut pour Teams.
    """
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError("WebRTC VAD exige 8/16/32/48 kHz")
    vad = webrtcvad.Vad(aggressiveness)

    # WebRTC veut du PCM 16-bit
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_len = int(sr * frame_ms / 1000) * 2  # 2 bytes/sample
    n_frames = len(pcm16) // frame_len

    flags = []
    for i in range(n_frames):
        chunk = pcm16[i * frame_len:(i + 1) * frame_len]
        flags.append(vad.is_speech(chunk, sr))

    # Regrouper les frames consécutives en segments
    segments = []
    in_speech = False
    start = 0
    for i, f in enumerate(flags):
        t = i * frame_ms / 1000.0
        if f and not in_speech:
            start = t
            in_speech = True
        elif not f and in_speech:
            segments.append((start, t))
            in_speech = False
    if in_speech:
        segments.append((start, n_frames * frame_ms / 1000.0))

    # Fusionner les trous courts
    merged = []
    for s, e in segments:
        if merged and (s - merged[-1][1]) * 1000 < merge_gap_ms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filtrer les segments trop courts
    return [(s, e) for s, e in merged if (e - s) * 1000 >= min_speech_ms]


# ═══════════════════════════════════════════════════════════════════════════════
#   2. Features — MFCC + Δ + ΔΔ
# ═══════════════════════════════════════════════════════════════════════════════

def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 19,
                 win_ms: int = 25, hop_ms: int = 10) -> np.ndarray:
    """Retourne (n_frames, 3*n_mfcc) — MFCC + delta + delta-delta, CMN par fichier."""
    n_fft = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2]).T  # (n_frames, 3*n_mfcc)
    # CMN — Cepstral Mean Normalization globale
    feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float64)


def time_to_frame(t: float, hop_ms: int = 10) -> int:
    return int(round(t * 1000 / hop_ms))


# ═══════════════════════════════════════════════════════════════════════════════
#   3. BIC segmentation — détection de changement de locuteur
# ═══════════════════════════════════════════════════════════════════════════════

def _logdet_cov(X: np.ndarray, reg: float = 1e-3) -> float:
    """log|Σ| robuste pour matrice diagonale (rapide, stable)."""
    if len(X) < 2:
        return 0.0
    var = X.var(axis=0) + reg
    return float(np.sum(np.log(var)))


def delta_bic(X1: np.ndarray, X2: np.ndarray, lam: float = 1.0) -> float:
    """
    ΔBIC entre deux fenêtres adjacentes (covariances diagonales).
    > 0 → deux distributions différentes (changement de locuteur)
    """
    N1, N2 = len(X1), len(X2)
    N = N1 + N2
    if N1 < 2 or N2 < 2:
        return -np.inf
    d = X1.shape[1]
    X = np.vstack([X1, X2])
    ld = _logdet_cov(X)
    ld1 = _logdet_cov(X1)
    ld2 = _logdet_cov(X2)
    bic = 0.5 * (N * ld - N1 * ld1 - N2 * ld2)
    # Pénalité (covariance diagonale → d paramètres par Gaussienne)
    penalty = lam * 0.5 * (2 * d) * np.log(N)
    return bic - penalty


def bic_segmentation(feats: np.ndarray, win_frames: int = 200,
                     shift_frames: int = 20, lam: float = 1.5) -> list:
    """
    Détecte les frames de changement à l'intérieur d'un bloc de parole.
    win_frames=200 → 2s de chaque côté ; shift=20 → résolution 200 ms.
    Retourne une liste d'indices de frames (changements).
    """
    changes = []
    n = len(feats)
    if n < 2 * win_frames:
        return changes
    i = win_frames
    last_change = 0
    while i + win_frames <= n:
        X1 = feats[i - win_frames:i]
        X2 = feats[i:i + win_frames]
        if delta_bic(X1, X2, lam=lam) > 0:
            if i - last_change >= win_frames // 2:  # évite les doublons
                changes.append(i)
                last_change = i
        i += shift_frames
    return changes


def split_speech_into_segments(speech_intervals: list, feats: np.ndarray,
                               hop_ms: int = 10, lam: float = 1.5,
                               win_ms: int = 2000) -> list:
    """
    Pour chaque intervalle de parole VAD, applique BIC pour le découper.
    Retourne une liste de segments (start_s, end_s, frame_start, frame_end).
    """
    win_frames = int(win_ms / hop_ms)
    segments = []
    for s, e in speech_intervals:
        f_start = time_to_frame(s, hop_ms)
        f_end = min(time_to_frame(e, hop_ms), len(feats))
        if f_end - f_start < 50:  # < 0.5 s : trop court, on ignore
            continue
        local = feats[f_start:f_end]
        local_changes = bic_segmentation(local, win_frames=win_frames,
                                         shift_frames=20, lam=lam)
        boundaries = [0] + local_changes + [len(local)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b - a < 50:  # < 0.5 s
                continue
            segments.append((
                (f_start + a) * hop_ms / 1000.0,
                (f_start + b) * hop_ms / 1000.0,
                f_start + a,
                f_start + b,
            ))
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
#   4. Clustering AHC avec distance ΔBIC entre segments
# ═══════════════════════════════════════════════════════════════════════════════

def bic_distance_matrix(feats: np.ndarray, segments: list,
                        lam: float = 1.0) -> np.ndarray:
    """Matrice de distances ΔBIC entre tous les segments."""
    n = len(segments)
    D = np.zeros((n, n))
    seg_feats = [feats[s[2]:s[3]] for s in segments]
    for i in range(n):
        for j in range(i + 1, n):
            # Distance = -ΔBIC : plus c'est petit, plus c'est similaire
            d = -delta_bic(seg_feats[i], seg_feats[j], lam=lam)
            D[i, j] = d
            D[j, i] = d
    return D


def ahc_bic(D: np.ndarray, threshold: float = 0.0,
            num_speakers: int = None) -> np.ndarray:
    """
    Clustering hiérarchique agglomératif manuel (linkage 'average')
    avec critère d'arrêt sur ΔBIC ou nombre de locuteurs cible.
    """
    n = D.shape[0]
    clusters = [[i] for i in range(n)]
    # Travailler sur une copie modifiable
    dist = D.copy().astype(float)
    np.fill_diagonal(dist, np.inf)

    while len(clusters) > 1:
        # Trouver la paire la plus proche
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = sorted(idx)
        min_d = dist[i, j]

        # Critère d'arrêt
        if num_speakers is not None:
            if len(clusters) <= num_speakers:
                break
        else:
            # Distance = -ΔBIC. Si la plus petite distance > -threshold,
            # ça veut dire que ΔBIC < threshold pour toutes les paires
            # → plus de fusion justifiée.
            if min_d > -threshold:
                break

        # Fusionner i et j
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        # Mettre à jour la matrice (linkage average)
        new_row = (dist[i] + dist[j]) / 2.0
        dist[i] = new_row
        dist[:, i] = new_row
        dist[i, i] = np.inf
        dist = np.delete(dist, j, axis=0)
        dist = np.delete(dist, j, axis=1)

    # Construire les labels
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            labels[m] = cid
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#   5. (Optionnel) Re-clustering GMM pour stabiliser
# ═══════════════════════════════════════════════════════════════════════════════

def refine_with_gmm(feats: np.ndarray, segments: list, labels: np.ndarray,
                    n_components: int = 8) -> np.ndarray:
    """
    Pour chaque cluster, entraîne une GMM, puis réassigne chaque segment
    au modèle qui le score le mieux. Une seule itération.
    """
    n_speakers = labels.max() + 1
    if n_speakers < 2:
        return labels
    gmms = {}
    for spk in range(n_speakers):
        X = np.vstack([feats[s[2]:s[3]] for s, l in zip(segments, labels) if l == spk])
        if len(X) < n_components * 5:
            n_comp = max(2, len(X) // 10)
        else:
            n_comp = n_components
        if len(X) < 10:
            continue
        gmm = GaussianMixture(n_components=n_comp, covariance_type="diag",
                              max_iter=50, reg_covar=1e-3, random_state=0)
        gmm.fit(X)
        gmms[spk] = gmm

    new_labels = labels.copy()
    for i, seg in enumerate(segments):
        X = feats[seg[2]:seg[3]]
        if len(X) < 5 or not gmms:
            continue
        scores = {spk: g.score(X) for spk, g in gmms.items()}
        new_labels[i] = max(scores, key=scores.get)
    return new_labels


# ═══════════════════════════════════════════════════════════════════════════════
#   6. Sortie
# ═══════════════════════════════════════════════════════════════════════════════

def merge_consecutive(segments: list, labels: np.ndarray) -> list:
    """Fusionne les segments adjacents partageant le même locuteur."""
    out = []
    for (s, e, _, _), lab in zip(segments, labels):
        if out and out[-1][2] == lab and s - out[-1][1] < 0.5:
            out[-1] = (out[-1][0], e, lab)
        else:
            out.append((s, e, lab))
    return out


def write_rttm(path: Path, file_id: str, diarization: list):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, lab in diarization:
            f.write(f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} "
                    f"<NA> <NA> SPK{lab:02d} <NA> <NA>\n")


def write_txt(path: Path, diarization: list):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, lab in diarization:
            f.write(f"[{s:7.2f} - {e:7.2f}]  Speaker {chr(65 + lab)}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Diarisation BIC + GMM (Pipeline 1, 100% classique)")
    parser.add_argument("--input", "-i", required=True, help="WAV 16 kHz mono")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Dossier de sortie (défaut : à côté de l'input)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Nombre de locuteurs (sinon auto via seuil BIC)")
    parser.add_argument("--bic-threshold", type=float, default=0.0,
                        help="Seuil d'arrêt AHC (défaut 0.0)")
    parser.add_argument("--bic-lambda-seg", type=float, default=1.5,
                        help="Pénalité BIC pour la segmentation (défaut 1.5)")
    parser.add_argument("--bic-lambda-cluster", type=float, default=1.0,
                        help="Pénalité BIC pour le clustering (défaut 1.0)")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--no-refine", action="store_true",
                        help="Désactiver le raffinement GMM")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    pipeline_name = Path(__file__).stem  # ex: "diarize_bic_gmm"
    file_id = in_path.stem
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = in_path.parent / "outputs" / pipeline_name / file_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DIARISATION BIC + GMM — {in_path.name}")
    print("=" * 78)

    # ── Chargement (via ffmpeg, accepte m4a/mp3/wav...) ──
    t0 = time.perf_counter()
    sr = 16000
    audio = load_audio_ffmpeg(in_path, sr=sr)
    duration = len(audio) / sr
    print(f"  [1/6] Audio chargé    : {duration:.1f}s ({duration/60:.1f} min)")

    # ── VAD ──
    t1 = time.perf_counter()
    speech = run_vad(audio, sr, aggressiveness=args.vad_aggressiveness)
    speech_total = sum(e - s for s, e in speech)
    print(f"  [2/6] VAD             : {len(speech)} zones de parole "
          f"({speech_total:.1f}s, {speech_total/duration*100:.0f}%) "
          f"— {time.perf_counter()-t1:.1f}s")

    if not speech:
        print("  Aucune parole détectée. Abandon.")
        sys.exit(1)

    # ── MFCC ──
    t1 = time.perf_counter()
    feats = extract_mfcc(audio, sr)
    print(f"  [3/6] MFCC+Δ+ΔΔ       : {feats.shape} "
          f"— {time.perf_counter()-t1:.1f}s")

    # ── Segmentation BIC ──
    t1 = time.perf_counter()
    segments = split_speech_into_segments(speech, feats, lam=args.bic_lambda_seg)
    print(f"  [4/6] Segmentation BIC: {len(segments)} segments "
          f"— {time.perf_counter()-t1:.1f}s")

    if len(segments) < 2:
        print("  Pas assez de segments pour le clustering.")
        sys.exit(1)

    # ── Distance matrix + AHC ──
    t1 = time.perf_counter()
    D = bic_distance_matrix(feats, segments, lam=args.bic_lambda_cluster)
    labels = ahc_bic(D, threshold=args.bic_threshold,
                     num_speakers=args.num_speakers)
    n_spk = len(set(labels))
    print(f"  [5/6] Clustering AHC  : {n_spk} locuteurs "
          f"— {time.perf_counter()-t1:.1f}s")

    # ── Refinement ──
    if not args.no_refine and n_spk >= 2:
        t1 = time.perf_counter()
        labels = refine_with_gmm(feats, segments, labels)
        n_spk = len(set(labels))
        print(f"  [6/6] Raffinement GMM : {n_spk} locuteurs "
              f"— {time.perf_counter()-t1:.1f}s")
    else:
        print(f"  [6/6] Raffinement GMM : désactivé")

    # ── Sortie ──
    diarization = merge_consecutive(segments, labels)
    rttm_path = out_dir / f"{file_id}.rttm"
    txt_path = out_dir / f"{file_id}.diarization.txt"
    write_rttm(rttm_path, file_id, diarization)
    write_txt(txt_path, diarization)

    total = time.perf_counter() - t0
    print()
    print("=" * 78)
    print(f"  RÉSULTAT")
    print("=" * 78)
    print(f"  Locuteurs détectés : {n_spk}")
    print(f"  Tours de parole    : {len(diarization)}")
    print(f"  RTF                : {total/duration:.2f}x")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  → {rttm_path.name}")
    print(f"  → {txt_path.name}")
    print()
    print("  Aperçu :")
    for s, e, lab in diarization[:10]:
        print(f"    [{s:7.2f} - {e:7.2f}]  Speaker {chr(65 + lab)}")
    if len(diarization) > 10:
        print(f"    ... ({len(diarization) - 10} de plus)")
    print()


if __name__ == "__main__":
    main()
