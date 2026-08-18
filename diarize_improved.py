#!/usr/bin/env python3
"""
Diarisation améliorée : pipeline `diarize` (WeSpeaker + Silero VAD)
avec améliorations par rapport au package de base :

  1. Sim Enhancement — raffinement de la matrice d'affinité avant
     spectral clustering (diagonal fill → gaussian blur → row threshold
     → symmetrization → diffusion → row max norm).  Technique empruntée
     à SpectralCluster (Google) / simple_diarizer.

  2. Choix du clustering — Spectral Clustering (SC, défaut),
     Agglomerative Hierarchical Clustering (AHC), ou MeanShift
     (auto-détection du nombre de speakers par densité, sans GMM-BIC).

  3. Fenêtres configurables — par défaut 1.2s/0.6s (comme diarize),
     mais possibilité d'utiliser des fenêtres plus larges (ex: 3.0s/1.5s)
     pour des embeddings plus stables et moins de fragmentation.
     (Inspiré de l'article "Towards Approximate Fast Diarization")

  4. VBx (Variational Bayes HMM) — raffinement post-clustering qui
     modélise la continuité temporelle via un HMM. Corrige les micro-
     erreurs (confettis, interjections mal attribuées) avec précision.
     (Ref: Landini et al., 2022)

  5. Choix du modèle d'embeddings — WeSpeaker ResNet34-LM (256-dim,
     défaut) ou CAM++ (512-dim, plus récent et performant). Les deux
     tournent en ONNX sur CPU.

  6. NME-SC (Normalized Maximum Eigengap) — estimation du nombre de
     speakers par analyse du spectre du Laplacien de la matrice
     d'affinité, en alternative au GMM-BIC. Plus robuste sur de
     grands nombres de speakers.

Le reste du pipeline est identique à `diarize_neural.py` :
  Silero VAD → WeSpeaker ResNet34-LM (256-dim ONNX) → GMM BIC → clustering

Usage :
  python diarize_improved.py -i meeting.m4a
  python diarize_improved.py -i meeting.m4a --cluster ahc
  python diarize_improved.py -i meeting.m4a --cluster meanshift
  python diarize_improved.py -i meeting.m4a --cluster sc --no-enhance
  python diarize_improved.py -i meeting.m4a --num-speakers 3
  python diarize_improved.py -i meeting.m4a --win-len 3.0 --hop-len 1.5
  python diarize_improved.py -i meeting.m4a --refine-vbx
  python diarize_improved.py -i meeting.m4a --cluster sc --refine-vbx --vbx-Fa 0.3

Dépendances :
  pip install diarize imageio-ffmpeg scipy
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from diarize.clustering import estimate_speakers
from diarize.embeddings import extract_embeddings, EMBEDDING_WINDOW, EMBEDDING_STEP
from diarize.utils import (
    DiarizeResult,
    Segment,
    SpeakerEstimationDetails,
    SpeechSegment,
    SubSegment,
    get_audio_duration,
)
from diarize.vad import run_vad


# ═══════════════════════════════════════════════════════════════════════════════
#   1. Sim Enhancement (emprunté à simple_diarizer / SpectralCluster)
# ═══════════════════════════════════════════════════════════════════════════════

def _diagonal_fill(A: np.ndarray) -> np.ndarray:
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(A.shape[0])] = np.max(A, axis=1)
    return A


def _gaussian_blur(A: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(A, sigma=sigma)


def _row_threshold_mult(A: np.ndarray, p: float = 0.95,
                        mult: float = 0.01) -> np.ndarray:
    percentiles = np.percentile(A, p * 100, axis=1)
    mask = A < percentiles[:, np.newaxis]
    return (mask * mult * A) + (~mask * A)


def _symmetrization(A: np.ndarray) -> np.ndarray:
    return np.maximum(A, A.T)


def _diffusion(A: np.ndarray) -> np.ndarray:
    return A @ A.T


def _row_max_norm(A: np.ndarray) -> np.ndarray:
    maxes = np.amax(A, axis=1, keepdims=True)
    maxes = np.where(maxes == 0, 1.0, maxes)
    return A / maxes


def sim_enhancement(A: np.ndarray) -> np.ndarray:
    """Raffine la matrice d'affinité pour réduire le bruit inter-speaker."""
    for fn in [_diagonal_fill, _gaussian_blur, _row_threshold_mult,
               _symmetrization, _diffusion, _row_max_norm]:
        A = fn(A)
    return A


# ═══════════════════════════════════════════════════════════════════════════════
#   NME-SC — Normalized Maximum Eigengap pour estimation du nb de speakers
#
#   Analyse les valeurs propres du Laplacien normalisé de la matrice
#   d'affinité. Le plus grand "saut" (gap) entre valeurs propres
#   consécutives indique le nombre de clusters.
#   Plus robuste que GMM-BIC, surtout avec beaucoup de speakers.
#   Ref: Park et al., "Auto-Tuning Spectral Clustering for Speaker
#        Diarization Using Normalized Maximum Eigengap", 2020.
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_speakers_nmesc(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: int = 20,
    enhance: bool = True,
) -> int:
    """Estime le nombre de speakers via Normalized Maximum Eigengap.

    1. Construit la matrice d'affinité (cosine similarity → [0,1])
    2. Optionnel : sim enhancement
    3. Calcule le Laplacien normalisé symétrique
    4. Trouve le plus grand gap entre valeurs propres consécutives
    """
    n = len(embeddings)
    if n < 2:
        return max(1, min_k)

    emb = normalize(embeddings, norm="l2")

    # Matrice d'affinité
    affinity = (cosine_similarity(emb) + 1) / 2
    np.fill_diagonal(affinity, 1.0)
    affinity = np.maximum(affinity, 0)

    if enhance:
        affinity = sim_enhancement(affinity.copy())

    # Laplacien normalisé symétrique : L_sym = I - D^{-1/2} A D^{-1/2}
    degree = np.sum(affinity, axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt

    # Valeurs propres (triées croissant)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_sym)))

    # Chercher le max eigengap entre min_k et max_k
    # Les k plus petites valeurs propres → k clusters
    # Le gap est entre eigenvalue[k-1] et eigenvalue[k]
    upper = min(max_k, n - 1)
    best_k = min_k
    max_gap = -1.0

    for k in range(min_k, upper + 1):
        if k < n:
            gap = eigenvalues[k] - eigenvalues[k - 1]
            # Normaliser par la valeur propre pour éviter le biais
            norm_gap = gap / (eigenvalues[k] + 1e-10) if eigenvalues[k] > 1e-10 else gap
            if norm_gap > max_gap:
                max_gap = norm_gap
                best_k = k

    return best_k


# ═══════════════════════════════════════════════════════════════════════════════
#   2. Clustering amélioré (SC avec sim enhancement + AHC)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_sc(embeddings: np.ndarray, k: int,
               enhance: bool = True) -> np.ndarray:
    """Spectral Clustering avec sim enhancement optionnel."""
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=int)

    affinity = (cosine_similarity(embeddings) + 1) / 2
    np.fill_diagonal(affinity, 1.0)
    affinity = np.maximum(affinity, 0)

    if enhance:
        affinity = sim_enhancement(affinity)

    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_init=10,
    )
    return sc.fit_predict(affinity)


def cluster_ahc(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative Hierarchical Clustering (cosine distance, average)."""
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=int)

    S = pairwise_distances(embeddings, metric="cosine")
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(S)


def cluster_meanshift(embeddings: np.ndarray) -> np.ndarray:
    """MeanShift clustering — auto-détection du nombre de speakers par densité.

    Pas besoin de spécifier k ni de GMM-BIC. L'algorithme trouve les modes
    de la distribution d'embeddings automatiquement.
    Inspiré de l'article "Towards Approximate Fast Diarization".
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    ms = MeanShift()
    return ms.fit_predict(embeddings)


def cluster_improved(
    embeddings: np.ndarray,
    min_speakers: int = 1,
    max_speakers: int = 20,
    num_speakers: int | None = None,
    method: str = "sc",
    enhance: bool = True,
    estimate_method: str = "gmm_bic",
) -> tuple[np.ndarray, SpeakerEstimationDetails | None]:
    """Estimation (GMM-BIC ou NME-SC) + clustering (SC, AHC, MeanShift)."""

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int), None

    emb = normalize(embeddings, norm="l2")

    # MeanShift : pas besoin de k, auto-détection
    if method == "meanshift":
        labels = cluster_meanshift(emb)
        n_found = len(set(labels))
        print(f"  MeanShift → {n_found} locuteurs détectés (auto)")
        return labels, None

    # Estimation du nombre de speakers
    details = None
    if num_speakers is not None:
        k = num_speakers
        print(f"  Nombre de locuteurs fixé : {k}")
    elif estimate_method == "nmesc":
        k = estimate_speakers_nmesc(emb, min_speakers, max_speakers, enhance=enhance)
        print(f"  NME-SC (eigengap) → {k} locuteurs estimés")
    else:
        k, details = estimate_speakers(emb, min_speakers, max_speakers)
        print(f"  GMM-BIC → {k} locuteurs estimés")

    # Clustering
    if method == "ahc":
        labels = cluster_ahc(emb, k)
    else:
        labels = cluster_sc(emb, k, enhance=enhance)

    return labels, details


# ═══════════════════════════════════════════════════════════════════════════════
#   VBx — Variational Bayes HMM refinement
#
#   Post-traitement qui raffine les frontières speaker en modélisant
#   la continuité temporelle via un HMM :
#     - États = speakers
#     - Émissions = similarité cosinus embedding↔centroïde speaker
#     - Transitions = pénalité pour changement de speaker (paramètre Fa)
#
#   Algorithme EM : forward-backward → posteriors → mise à jour centroïdes.
#   Corrige les micro-erreurs de clustering (confettis, interjections mal
#   attribuées) avec précision.
#
#   Ref: Landini et al., "Bayesian HMM clustering of x-vector sequences
#        (VBx) for speaker diarization", 2022.
# ═══════════════════════════════════════════════════════════════════════════════

def refine_vbx(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subsegments: list[SubSegment],
    Fa: float = 0.4,
    Fb: float = 17.0,
    max_iter: int = 40,
) -> np.ndarray:
    """Raffine les labels de clustering via un HMM Bayésien variationnel.

    Args:
        embeddings: (N, D) L2-normalisés.
        labels: labels initiaux du clustering.
        subsegments: pour trier par temps.
        Fa: probabilité de changement de speaker (0→1). Plus bas = moins
            de changements = segments plus longs. Typique: 0.3–0.5.
        Fb: concentration du modèle speaker (similarité cosinus × Fb).
            Plus haut = plus confiant dans les centroïdes. Typique: 10–20.
        max_iter: nombre max d'itérations EM.

    Returns:
        Labels raffinés (même shape que labels).
    """
    n = len(embeddings)
    speakers = sorted(set(labels))
    K = len(speakers)

    if K < 2 or n < 2:
        return labels

    # Mapper les labels vers 0..K-1
    spk_to_idx = {spk: i for i, spk in enumerate(speakers)}
    idx_labels = np.array([spk_to_idx[l] for l in labels])

    # Trier par temps
    order = np.argsort([s.start for s in subsegments])
    emb_sorted = normalize(embeddings[order], norm="l2")
    lab_sorted = idx_labels[order]

    # Initialiser les centroïdes speakers
    means = np.zeros((K, emb_sorted.shape[1]))
    for k in range(K):
        mask = lab_sorted == k
        if mask.sum() > 0:
            means[k] = emb_sorted[mask].mean(axis=0)
    means = normalize(means, norm="l2")

    # Matrice de transition (log)
    # P(rester) = 1 - Fa, P(changer vers un speaker spécifique) = Fa/(K-1)
    log_stay = np.log(max(1.0 - Fa, 1e-10))
    log_switch = np.log(max(Fa / (K - 1), 1e-10)) if K > 1 else -np.inf
    log_trans = np.full((K, K), log_switch)
    np.fill_diagonal(log_trans, log_stay)

    # EM iterations
    for iteration in range(max_iter):
        # Log-vraisemblance : Fb × cosine_similarity(embedding, centroïde)
        log_likes = Fb * (emb_sorted @ means.T)  # (N, K)

        # Forward pass (log domain)
        log_alpha = np.full((n, K), -np.inf)
        log_alpha[0] = log_likes[0] + np.log(1.0 / K)
        for t in range(1, n):
            for k in range(K):
                log_alpha[t, k] = log_likes[t, k] + np.logaddexp.reduce(
                    log_alpha[t - 1] + log_trans[:, k]
                )

        # Backward pass (log domain)
        log_beta = np.zeros((n, K))
        for t in range(n - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_beta[t + 1] + log_likes[t + 1] + log_trans[k, :]
                )

        # Posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # M-step : mise à jour des centroïdes
        new_means = gamma.T @ emb_sorted  # (K, D)
        # Éviter division par zéro
        row_norms = np.linalg.norm(new_means, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        new_means = new_means / row_norms

        # Convergence ?
        if np.allclose(means, new_means, atol=1e-4):
            print(f"    VBx convergé en {iteration + 1} itérations")
            break
        means = new_means
    else:
        print(f"    VBx max itérations atteint ({max_iter})")

    # Labels finaux
    refined_sorted = np.argmax(gamma, axis=1)

    # Re-mapper vers les labels originaux et dé-trier
    idx_to_spk = {i: spk for spk, i in spk_to_idx.items()}
    result = np.empty_like(labels)
    result[order] = np.array([idx_to_spk[l] for l in refined_sorted])

    n_changes = int(np.sum(result != labels))
    print(f"    VBx : {n_changes}/{n} labels modifiés "
          f"({100*n_changes/n:.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#   3. Extraction d'embeddings avec fenêtre configurable
# ═══════════════════════════════════════════════════════════════════════════════

CAMPP_ONNX_PATH = str(Path(__file__).parent / "pretrained_models" / "campplus" / "voxceleb_CAM++_LM.onnx")


def extract_embeddings_custom(
    audio_path: str,
    speech_segments: list[SpeechSegment],
    win_len: float = EMBEDDING_WINDOW,
    hop_len: float = EMBEDDING_STEP,
    min_seg: float = 0.4,
    embed_model: str = "resnet34",
) -> tuple[np.ndarray, list[SubSegment]]:
    """Extraction d'embeddings avec fenêtre et modèle configurables.

    embed_model: "resnet34" (WeSpeaker ResNet34-LM, 256-dim) ou
                 "campplus" (CAM++, 512-dim, plus performant).
    """
    import os
    import tempfile
    import wespeakerruntime as wespeaker_rt
    import soundfile as sf

    if embed_model == "campplus":
        model = wespeaker_rt.Speaker(onnx_path=CAMPP_ONNX_PATH)
    else:
        model = wespeaker_rt.Speaker(lang="en")
    audio_data, sr = sf.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    embeddings = []
    subsegments = []

    for idx, seg in enumerate(speech_segments):
        seg_dur = seg.duration
        if seg_dur < min_seg:
            continue

        if seg_dur <= win_len * 1.5:
            windows = [(seg.start, seg.end)]
        else:
            windows = []
            ws = seg.start
            while ws + min_seg < seg.end:
                we = min(ws + win_len, seg.end)
                windows.append((ws, we))
                ws += hop_len

        for ws, we in windows:
            s_sample = int(ws * sr)
            e_sample = int(we * sr)
            chunk = audio_data[s_sample:e_sample]

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, chunk, sr)
                emb = model.extract_embedding(tmp_path)
            except Exception:
                continue
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            if emb is not None:
                if emb.ndim == 2:
                    emb = emb[0]
                embeddings.append(emb)
                subsegments.append(SubSegment(start=ws, end=we, parent_idx=idx))

    if not embeddings:
        dim = 512 if embed_model == "campplus" else 256
        return np.empty((0, dim), dtype=np.float32), []

    return np.stack(embeddings), subsegments


# ═══════════════════════════════════════════════════════════════════════════════
#   Construction des segments (repris de diarize.__init__)
# ═══════════════════════════════════════════════════════════════════════════════

def build_segments(
    speech_segments: list[SpeechSegment],
    subsegments: list[SubSegment],
    labels: np.ndarray,
    merge_gap: float = 0.7,
) -> list[Segment]:
    """Assemble les segments de diarisation à partir des labels de clustering."""
    raw = []
    for sub, label in zip(subsegments, labels):
        raw.append((sub.start, sub.end, f"SPEAKER_{int(label):02d}"))

    # Segments VAD courts sans embedding → speaker le plus proche
    covered = {sub.parent_idx for sub in subsegments}
    for idx, seg in enumerate(speech_segments):
        if idx in covered:
            continue
        mid = (seg.start + seg.end) / 2
        best_spk, best_dist = "SPEAKER_00", float("inf")
        for sub, label in zip(subsegments, labels):
            d = abs(mid - (sub.start + sub.end) / 2)
            if d < best_dist:
                best_dist = d
                best_spk = f"SPEAKER_{int(label):02d}"
        raw.append((seg.start, seg.end, best_spk))

    raw.sort(key=lambda x: x[0])
    if not raw:
        return []

    # Merge adjacent same-speaker
    merged = [[raw[0][0], raw[0][1], raw[0][2]]]
    for start, end, spk in raw[1:]:
        prev = merged[-1]
        if spk == prev[2] and (start - prev[1]) < merge_gap:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end, spk])

    return [Segment(start=m[0], end=m[1], speaker=m[2]) for m in merged]


# ═══════════════════════════════════════════════════════════════════════════════
#   Conversion audio → WAV 16 kHz mono
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_wav(src: Path, sr: int = 16000) -> Path:
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
#   Sortie
# ═══════════════════════════════════════════════════════════════════════════════

def write_txt(path: Path, segments: list[Segment]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:7.2f} - {seg.end:7.2f}]  {seg.speaker}\n")


def write_rttm(path: Path, segments: list[Segment], file_id: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            dur = seg.end - seg.start
            f.write(f"SPEAKER {file_id} 1 {seg.start:.6f} {dur:.6f} "
                    f"<NA> <NA> {seg.speaker} <NA> <NA>\n")


# ═══════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Diarisation améliorée (WeSpeaker + sim enhancement + AHC/SC/MeanShift)")
    ap.add_argument("--input", "-i", required=True, help="Fichier audio")
    ap.add_argument("--output-dir", "-o", default=None)
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="Nombre exact de locuteurs (si connu)")
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=20)
    ap.add_argument("--embed", choices=["resnet34", "campplus"], default="resnet34",
                    help="Modèle d'embeddings : resnet34 (256-dim) ou campplus (512-dim)")
    ap.add_argument("--estimate", choices=["gmm_bic", "nmesc"], default="gmm_bic",
                    help="Méthode d'estimation nb speakers : gmm_bic ou nmesc (eigengap)")
    ap.add_argument("--cluster", choices=["sc", "ahc", "meanshift"], default="sc",
                    help="Méthode de clustering (défaut: sc)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="Désactiver le sim enhancement (SC uniquement)")
    ap.add_argument("--win-len", type=float, default=None,
                    help=f"Fenêtre d'embedding en secondes (défaut: {EMBEDDING_WINDOW}s)")
    ap.add_argument("--hop-len", type=float, default=None,
                    help=f"Hop d'embedding en secondes (défaut: {EMBEDDING_STEP}s)")
    ap.add_argument("--refine-vbx", action="store_true",
                    help="Appliquer le raffinement VBx (Variational Bayes HMM) "
                         "après le clustering pour corriger les frontières")
    ap.add_argument("--vbx-Fa", type=float, default=0.4,
                    help="VBx : probabilité de changement de speaker (défaut: 0.4)")
    ap.add_argument("--vbx-Fb", type=float, default=17.0,
                    help="VBx : concentration du modèle speaker (défaut: 17.0)")
    ap.add_argument("--diag", action="store_true",
                    help="Générer les diagnostics visuels (VAD, embeddings, "
                         "affinité, timeline, RAM, timing) dans un sous-dossier")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERREUR : {in_path} introuvable")
        sys.exit(1)

    file_id = in_path.stem
    use_custom_win = args.win_len is not None or args.hop_len is not None
    win_len = args.win_len if args.win_len else EMBEDDING_WINDOW
    hop_len = args.hop_len if args.hop_len else EMBEDDING_STEP

    # Construire un nom de pipeline qui encode TOUS les paramètres
    # Ex: diarize_improved__campplus__nmesc__sc_enhanced__w3.0__h1.5__vbx_Fa0.3_Fb17__nspk2
    parts = ["diarize_improved"]
    parts.append(args.embed)
    parts.append(args.estimate)
    parts.append(args.cluster)
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
    pipeline_name = "__".join(parts)
    out_dir = Path(args.output_dir) if args.output_dir else (
        in_path.parent / "outputs" / pipeline_name / file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  DIARISATION AMÉLIORÉE — {in_path.name}")
    print(f"  Embeddings: {args.embed.upper()}"
          f" | Estimation: {args.estimate.upper()}"
          f" | Clustering: {args.cluster.upper()}")
    print(f"  Sim Enhancement: {'OFF' if args.no_enhance else 'ON'}"
          f" | VBx: {'ON' if args.refine_vbx else 'OFF'}"
          f" | Fenêtre: {win_len}s/{hop_len}s")
    print("=" * 78)

    # ── Diagnostics (optionnel) ──
    diag = None
    if args.diag:
        from diagnostics import Diagnostics
        diag = Diagnostics(out_dir, file_id)

    timings = {}
    t0 = time.perf_counter()

    # 1. Conversion
    wav_path = convert_to_wav(in_path)
    duration = get_audio_duration(str(wav_path))
    timings["1_conversion"] = time.perf_counter() - t0
    print(f"  [1/4] Audio préparé     : {wav_path.name} ({duration:.1f}s)")
    if diag:
        diag.snapshot_ram("conversion")

    # 2. VAD
    t1 = time.perf_counter()
    speech_segments = run_vad(str(wav_path))
    timings["2_vad"] = time.perf_counter() - t1
    total_speech = sum(s.duration for s in speech_segments)
    print(f"  [2/4] VAD (Silero)      : {len(speech_segments)} segments, "
          f"{total_speech:.1f}s de parole — {timings['2_vad']:.1f}s")
    if diag:
        diag.snapshot_ram("vad")
        diag.plot_vad(str(wav_path), speech_segments)
        print(f"         → diag: 01_vad.png")

    if not speech_segments:
        print("  ERREUR : aucune parole détectée")
        sys.exit(1)

    # 3. Embeddings
    t2 = time.perf_counter()
    use_custom_embed = args.embed != "resnet34"
    if use_custom_win or use_custom_embed:
        embeddings, subsegments = extract_embeddings_custom(
            str(wav_path), speech_segments, win_len=win_len, hop_len=hop_len,
            embed_model=args.embed)
    else:
        embeddings, subsegments = extract_embeddings(str(wav_path), speech_segments)
    timings["3_embeddings"] = time.perf_counter() - t2
    model_name = "CAM++" if args.embed == "campplus" else "ResNet34-LM"
    print(f"  [3/4] Embeddings ({model_name}) : {embeddings.shape[0]} vecteurs "
          f"({embeddings.shape[1]}-dim) — {timings['3_embeddings']:.1f}s")
    if diag:
        diag.snapshot_ram("embeddings")

    if len(embeddings) == 0:
        print("  ERREUR : aucun embedding extrait")
        sys.exit(1)

    # 4. Clustering amélioré
    t3 = time.perf_counter()
    labels, details = cluster_improved(
        embeddings,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        method=args.cluster,
        enhance=not args.no_enhance,
        estimate_method=args.estimate,
    )
    timings["4_clustering"] = time.perf_counter() - t3
    n_spk = len(set(labels))
    print(f"  [4/4] Clustering        : {n_spk} locuteurs — "
          f"{timings['4_clustering']:.1f}s")
    if diag:
        diag.snapshot_ram("clustering")
        diag.plot_affinity(embeddings, labels)
        diag.plot_embeddings(embeddings, subsegments, labels,
                             title_suffix="(après clustering)")
        print(f"         → diag: 02_embeddings, 03_affinity")

    # 5. Raffinement VBx (optionnel)
    labels_before_vbx = labels.copy() if args.refine_vbx else None
    if args.refine_vbx:
        t4 = time.perf_counter()
        print(f"  [5/5] Raffinement VBx (Fa={args.vbx_Fa}, Fb={args.vbx_Fb})...")
        labels = refine_vbx(
            embeddings, labels, subsegments,
            Fa=args.vbx_Fa, Fb=args.vbx_Fb,
        )
        timings["5_vbx"] = time.perf_counter() - t4
        n_spk = len(set(labels))
        print(f"        → {n_spk} locuteurs après VBx — "
              f"{timings['5_vbx']:.1f}s")
        if diag:
            diag.snapshot_ram("vbx")
            diag.plot_embeddings(embeddings, subsegments, labels,
                                 title_suffix="(après VBx)")
            print(f"         → diag: 02_embeddings (après VBx)")

    # Construction des segments
    segments = build_segments(speech_segments, subsegments, labels)

    # Sortie
    rttm_path = out_dir / f"{file_id}.rttm"
    txt_path = out_dir / f"{file_id}.diarization.txt"
    write_rttm(rttm_path, segments, file_id)
    write_txt(txt_path, segments)

    total = time.perf_counter() - t0

    # ── Diagnostics finaux ──
    if diag:
        diag.snapshot_ram("fin")
        diag.plot_timeline(segments, duration)
        diag.plot_timing(timings, total, duration)
        diag.plot_ram()
        report_path = diag.generate_report(pipeline_name, {
            "Fichier": str(in_path),
            "Durée audio": f"{duration:.1f}s ({duration/60:.1f} min)",
            "Embeddings": f"{args.embed} ({embeddings.shape[1]}-dim)",
            "Estimation": args.estimate,
            "Clustering": args.cluster,
            "Sim Enhancement": "OFF" if args.no_enhance else "ON",
            "Fenêtre / Hop": f"{win_len}s / {hop_len}s",
            "VBx": f"Fa={args.vbx_Fa}, Fb={args.vbx_Fb}" if args.refine_vbx else "OFF",
            "Nb speakers": str(args.num_speakers) if args.num_speakers else "auto",
            "Locuteurs détectés": str(n_spk),
            "Segments": str(len(segments)),
            "Temps total": f"{total:.1f}s",
            "RTF": f"{total/duration:.2f}x",
        })
        print(f"\n  DIAGNOSTICS → {report_path}")

    print()
    print("=" * 78)
    print(f"  RÉSULTAT")
    print("=" * 78)
    print(f"  Durée audio        : {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Locuteurs détectés : {n_spk}")
    print(f"  Segments           : {len(segments)}")
    print(f"  RTF                : {total/duration:.2f}x")
    print(f"  Temps total        : {total:.1f}s")
    print(f"  Pipeline           : {pipeline_name}")
    print(f"  → {rttm_path}")
    print(f"  → {txt_path}")
    if diag:
        print(f"  → {diag.out_dir / 'report.html'}")
    print()
    print("  Aperçu :")
    for seg in segments[:10]:
        print(f"    [{seg.start:7.2f} - {seg.end:7.2f}]  {seg.speaker}")
    if len(segments) > 10:
        print(f"    ... ({len(segments) - 10} de plus)")
    print()


if __name__ == "__main__":
    main()
