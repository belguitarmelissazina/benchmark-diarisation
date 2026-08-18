"""Clustering step: speaker count estimation + label assignment.

Provides:
  - estimate_speakers_gmm_bic : GMM + BIC (classic, from diarize lib)
  - estimate_speakers_nmesc   : Normalized Maximum Eigengap (Park et al. 2020)
  - sim_enhancement           : affinity refinement (SpectralCluster/simple_diarizer)
  - cluster_sc                : Spectral Clustering (+optional enhance)
  - cluster_ahc               : Agglomerative Hierarchical Clustering
  - cluster_meanshift         : MeanShift (auto k via density)
  - cluster_speakers          : high-level wrapper (estimate + cluster)
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from .models import SpeakerEstimationDetails


# ── Sim enhancement ──────────────────────────────────────────────────────────

def _diagonal_fill(A: np.ndarray) -> np.ndarray:
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(A.shape[0])] = np.max(A, axis=1)
    return A


def _gaussian_blur(A: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(A, sigma=sigma)


def _row_threshold_mult(A: np.ndarray, p: float = 0.95, mult: float = 0.01) -> np.ndarray:
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
    """Refine affinity matrix to reduce inter-speaker noise.

    The Laplacian requires a symmetric matrix, so we re-symmetrize after
    every step that can break symmetry (row_threshold_mult, row_max_norm).
    """
    for fn in (_diagonal_fill, _gaussian_blur, _row_threshold_mult,
               _symmetrization, _diffusion, _row_max_norm,
               _symmetrization):          # <-- final symmetrization
        A = fn(A)
    return A


# ── Speaker count estimation: GMM-BIC ────────────────────────────────────────

def estimate_speakers_gmm_bic(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: int = 20,
) -> tuple[int, SpeakerEstimationDetails]:
    """Estimate speaker count via GMM with BIC selection.

    1. L2-normalize → PCA(8) → sweep k ∈ [min_k, max_k] → pick argmin BIC.
    2. Single-speaker pre-check: if cosine sim p10 ≥ 0.16, return 1.
    """
    n = embeddings.shape[0]
    if n == 0:
        return max(1, min_k), SpeakerEstimationDetails(
            method="gmm_bic", best_k=max(1, min_k), reason="no_embeddings")

    emb = normalize(embeddings, norm="l2")

    if n < 4:
        return max(1, min_k), SpeakerEstimationDetails(
            method="gmm_bic", best_k=max(1, min_k), reason="too_few_samples")

    # Single-speaker pre-check: if the MEDIAN pairwise cosine similarity
    # is very high, all embeddings look alike → likely 1 speaker.
    # Using median (not p10) makes this robust across embedding models
    # with different baseline similarity distributions.
    sim_matrix = cosine_similarity(emb)
    mask = ~np.eye(n, dtype=bool)
    sim_median = float(np.median(sim_matrix[mask]))
    if sim_median >= 0.75 and min_k <= 1:
        return 1, SpeakerEstimationDetails(
            method="gmm_bic", best_k=1,
            reason="cosine_similarity_single_speaker",
            cosine_sim_p10=round(sim_median, 4))

    n_pca = 8
    actual_pca = min(n_pca, n - 1, emb.shape[1])
    emb_pca = PCA(n_components=actual_pca, random_state=42).fit_transform(emb)

    k_upper = max(min_k + 1, min(max_k + 1, n // 2 + 1))
    k_to_bic: dict[int, float] = {}
    for k in range(min_k, k_upper):
        try:
            gmm = GaussianMixture(
                n_components=k, covariance_type="full",
                random_state=42, n_init=5, max_iter=300)
            gmm.fit(emb_pca)
            k_to_bic[k] = gmm.bic(emb_pca)
        except Exception:
            continue

    if not k_to_bic:
        return min_k, SpeakerEstimationDetails(
            method="gmm_bic", best_k=min_k, reason="gmm_failed")

    best_k = min(k_to_bic, key=k_to_bic.get)  # type: ignore[arg-type]
    return best_k, SpeakerEstimationDetails(
        method="gmm_bic", best_k=best_k, pca_dim=actual_pca,
        k_bics={k: round(b, 1) for k, b in sorted(k_to_bic.items())})


# ── Speaker count estimation: NME-SC (Park et al. 2020) ─────────────────────
# Faithful reimplementation of Auto-Tuning Spectral Clustering.
# Reference: https://github.com/tango4j/Auto-Tuning-Spectral-Clustering

def _p_neighbor_binarize(A: np.ndarray, p: int) -> np.ndarray:
    """Keep only the top-p similarity values per row (set to 1), zero the rest.

    Vectorized: for each row, find the p-th largest value as a threshold,
    then set all entries >= threshold to 1 (column-write per reference impl).
    """
    n = A.shape[0]
    p = min(p, n)
    # For each row, find the p-th largest value
    # np.partition is O(N) per row vs O(N log N) for full sort
    kth = n - p  # index for partition (kth smallest = (n-p)-th)
    thresholds = np.partition(A, kth, axis=1)[:, kth]  # (n,)
    # Column-write: X_out[idx, i] = 1 means "row i of A selects column idx"
    # This is equivalent to: for each row i, mark top-p columns → write into those rows of X_out at column i
    mask = A >= thresholds[:, np.newaxis]  # (n, n) bool: row i has True at top-p columns
    X_out = mask.astype(np.float64).T  # transpose: column-write
    return X_out


def _unnormalized_laplacian(A: np.ndarray) -> np.ndarray:
    """L = D - A (unnormalized), with diagonal of A zeroed first."""
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    D = np.diag(np.sum(np.abs(A), axis=1))
    return D - A


def _eigengaps(L: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Eigendecomposition + consecutive eigengap list."""
    lambdas = np.sort(np.real(np.linalg.eigvalsh(L)))
    gaps = [float(lambdas[i + 1] - lambdas[i]) for i in range(len(lambdas) - 1)]
    return lambdas, gaps


def _is_fully_connected(A: np.ndarray) -> bool:
    """Check if adjacency graph is fully connected via BFS from node 0."""
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        neighbors = np.where(A[node] > 0)[0]
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)
    return visited.all()


def estimate_speakers_nmesc(
    embeddings: np.ndarray,
    min_k: int = 1,
    max_k: int = 20,
    enhance: bool = False,
    max_rp_ratio: float = 0.25,
    sparse_search_volume: int = 30,
) -> int:
    """Estimate speaker count via NME-SC (Park et al. 2020).

    Implements Algorithm 1 from the paper exactly:
      for p = 1 to P:
          Ap = binarize(A, p)           # top-p per row → 1, rest → 0
          Āp = (Ap + Ap^T) / 2          # symmetrize
          Lp = Laplacian(Āp)            # unnormalized: D - Āp
          Up, Σp, Vp^T = SVD(Lp)        # full eigendecomposition
          ep = eigengap(Σp)             # consecutive eigenvalue gaps
          gp = max(ep) / max(Σp)        # NME value
          r[p] = p / gp                 # ratio
      p̂ = argmin(r)
      k = argmax(e_p̂)
    """
    n = len(embeddings)
    if n < 2:
        return max(1, min_k)

    eps = 1e-10
    emb = normalize(embeddings, norm="l2")
    # Step 1: raw cosine similarity — no kernel, no scaling (paper eq. 4)
    affinity = cosine_similarity(emb)

    # p-sweep range: 1 to P where P = max_rp_ratio * N (paper default ~25%)
    max_N = max(2, int(n * max_rp_ratio))
    # Sparse search: sample up to sparse_search_volume evenly-spaced p values
    n_search = min(max_N, sparse_search_volume)
    p_list = np.unique(
        np.linspace(1, max_N, n_search, endpoint=True).astype(int)
    ).tolist()

    def _eval_p(p: int) -> tuple[int, float, int]:
        # Ap = binarize(A, p)  — paper step 2
        Ap = _p_neighbor_binarize(affinity, p)
        # Āp = (Ap + Ap^T) / 2  — paper step 3
        A_bar = 0.5 * (Ap + Ap.T)

        # Lp = D - Āp  (unnormalized Laplacian) — paper step 4
        A_work = A_bar.copy()
        np.fill_diagonal(A_work, 0.0)
        D_diag = np.sum(A_work, axis=1)
        Lp_dense = np.diag(D_diag) - A_work

        # Step 5 (paper): SVD to get eigenvalues. We only use the smallest
        # (max_k+1) eigenvalues (for the eigengap vector capped at max_k) and
        # λ_N = max eigenvalue (for the NME denominator in eq. 10). Compute
        # exactly those using a sparse eigensolver — numerically identical to
        # dense eigh for the quantities that feed r(p) and k.
        n_local = Lp_dense.shape[0]
        k_small = min(max_k + 1, n_local - 1)
        # Dense eigh only: ARPACK (sparse eigsh) is non-deterministic under
        # multithreaded BLAS and caused run-to-run DER swings (e.g. FR-069
        # flipped k=3↔5 → DER 0.13↔0.27). Dense is slower on large N but
        # stable bit-to-bit.
        all_lam = np.sort(np.real(scipy.linalg.eigh(Lp_dense, eigvals_only=True)))
        smallest = all_lam[:k_small + 1]
        lambda_N = float(all_lam[-1])

        # ep = eigengap(Σp) capped at max_k  — paper step 6
        ep_capped = np.diff(smallest)[:max_k]

        # gp = max(ep) / max(Σp)  — paper step 7 (eq. 10)
        gp = float(np.max(ep_capped)) / (lambda_N + eps)

        # r[p] = p / gp  — paper step 8 (eq. 11)
        r_p = p / (gp + eps)
        # k = argmax(ep) + 1  — paper eq. 14
        k_p = int(np.argmax(ep_capped)) + 1
        return p, r_p, k_p

    # Sequential p-sweep (parallelization causes BLAS oversubscription on Windows)
    r_list: list[float] = []
    est_spk_dict: dict[int, int] = {}
    for p in p_list:
        _, r_p, k_p = _eval_p(p)
        r_list.append(r_p)
        est_spk_dict[p] = k_p

    # p̂ = argmin(r)  — paper eq. 13
    best_idx = int(np.argmin(r_list))
    best_p = p_list[best_idx]
    best_k = est_spk_dict[best_p]

    # Ensure graph is fully connected at p̂ (reference impl safety check)
    X_conn = 0.5 * (_p_neighbor_binarize(affinity, best_p)
                     + _p_neighbor_binarize(affinity, best_p).T)
    if not _is_fully_connected(X_conn):
        for p in p_list[best_idx:]:
            X_bin = _p_neighbor_binarize(affinity, p)
            X_conn = 0.5 * (X_bin + X_bin.T)
            if _is_fully_connected(X_conn):
                best_p = p
                best_k = est_spk_dict[p]
                break

    # Clamp to [min_k, max_k]
    best_k = max(min_k, min(max_k, best_k))
    return best_k


# ── Clustering algorithms ────────────────────────────────────────────────────

def cluster_sc(embeddings: np.ndarray, k: int, *, enhance: bool = True) -> np.ndarray:
    """Spectral Clustering on cosine affinity, with optional sim enhancement."""
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
        n_clusters=k, affinity="precomputed",
        assign_labels="kmeans", random_state=42, n_init=10)
    return sc.fit_predict(affinity)


def cluster_ahc(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative Hierarchical Clustering (cosine distance, average linkage)."""
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    k = min(k, n)
    if k == 1:
        return np.zeros(n, dtype=int)

    D = pairwise_distances(embeddings, metric="cosine")
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average")
    return model.fit_predict(D)


def _merge_small_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_cluster_size: int = 3,
) -> np.ndarray:
    """Reassign embeddings in tiny clusters to their nearest larger cluster."""
    labels = labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    small = set(uniq[counts < min_cluster_size].tolist())
    large = [c for c in uniq if c not in small]
    if not small or not large:
        return labels
    centroids = {c: embeddings[labels == c].mean(axis=0) for c in large}
    C = np.stack([centroids[c] for c in large])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-10)
    for i, lbl in enumerate(labels):
        if lbl in small:
            e = embeddings[i]
            e = e / (np.linalg.norm(e) + 1e-10)
            labels[i] = large[int(np.argmax(C @ e))]
    # Relabel 0..k-1
    _, new = np.unique(labels, return_inverse=True)
    return new


def _merge_close_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    merge_threshold: float,
) -> np.ndarray:
    """Iteratively merge the two clusters with closest centroids until the
    closest pair exceeds `merge_threshold` (cosine distance)."""
    labels = labels.copy()
    while True:
        uniq = np.unique(labels)
        if len(uniq) < 2:
            break
        C = np.stack([embeddings[labels == c].mean(axis=0) for c in uniq])
        D = pairwise_distances(C, metric="cosine")
        np.fill_diagonal(D, np.inf)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if D[i, j] > merge_threshold:
            break
        labels[labels == uniq[j]] = uniq[i]
    _, new = np.unique(labels, return_inverse=True)
    return new


def cluster_ahc_threshold(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    min_speakers: int = 1,
    max_speakers: int = 20,
    percentile: float | None = None,
    min_cluster_size: int = 1,
    merge_threshold: float | None = None,
) -> np.ndarray:
    """AHC with cosine distance threshold — no k estimation needed.

    Merges clusters until the closest pair exceeds `threshold`.
    This is the approach used by pyannote 3.x internally.
    Lower threshold = more speakers, higher = fewer speakers.

    If `percentile` is given (0-100), threshold is computed as that
    percentile of the pairwise-distance distribution for this audio —
    self-calibrating across embedding models and recordings.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    D = pairwise_distances(embeddings, metric="cosine")
    if percentile is not None:
        iu = np.triu_indices_from(D, k=1)
        threshold = float(np.percentile(D[iu], percentile))
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="precomputed", linkage="average")
    labels = model.fit_predict(D)

    # Clamp to min/max speakers
    n_found = len(set(labels))
    if n_found < min_speakers:
        model = AgglomerativeClustering(
            n_clusters=min_speakers, metric="precomputed", linkage="average")
        labels = model.fit_predict(D)
    elif n_found > max_speakers:
        model = AgglomerativeClustering(
            n_clusters=max_speakers, metric="precomputed", linkage="average")
        labels = model.fit_predict(D)

    # Post-processing: drop tiny clusters, then merge close centroids.
    if min_cluster_size > 1:
        labels = _merge_small_clusters(embeddings, labels, min_cluster_size)
    if merge_threshold is not None:
        labels = _merge_close_centroids(embeddings, labels, merge_threshold)

    return labels


def cluster_cosine_greedy(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    max_speakers: int = 20,
    percentile: float | None = None,
) -> np.ndarray:
    """Simple online-style greedy clustering by cosine similarity.

    Process embeddings sequentially:
      - First embedding starts speaker 0.
      - For each next embedding, compute cosine similarity to all
        existing speaker centroids.
      - If max similarity >= threshold, assign to that speaker.
      - Otherwise, create a new speaker.

    No matrices, no eigengaps, no estimation. Just similarity comparison.
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)

    if percentile is not None and n >= 2:
        # Use cosine SIMILARITY percentile (greedy uses similarity, not distance).
        # High percentile (e.g. 90) = threshold near the most-similar pairs =
        # stricter merging = more speakers. Inverse semantics from AHC distance.
        S = cosine_similarity(embeddings)
        iu = np.triu_indices_from(S, k=1)
        threshold = float(np.percentile(S[iu], percentile))

    labels = np.zeros(n, dtype=int)
    # Each centroid is the running mean of assigned embeddings
    centroids = [embeddings[0].copy()]
    centroid_counts = [1]

    for i in range(1, n):
        emb = embeddings[i]
        # Cosine similarity to all centroids
        sims = np.array([
            np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-10)
            for c in centroids
        ])
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        if best_sim >= threshold and len(centroids) >= 1:
            labels[i] = best_idx
            # Update centroid (running mean)
            centroid_counts[best_idx] += 1
            centroids[best_idx] += (emb - centroids[best_idx]) / centroid_counts[best_idx]
        elif len(centroids) < max_speakers:
            labels[i] = len(centroids)
            centroids.append(emb.copy())
            centroid_counts.append(1)
        else:
            # Max speakers reached, assign to closest
            labels[i] = best_idx
            centroid_counts[best_idx] += 1
            centroids[best_idx] += (emb - centroids[best_idx]) / centroid_counts[best_idx]

    return labels


def cluster_meanshift(embeddings: np.ndarray) -> np.ndarray:
    """MeanShift — auto-detects k by density. No estimation needed."""
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    return MeanShift().fit_predict(embeddings)


# ── High-level wrapper ───────────────────────────────────────────────────────

def cluster_speakers(
    embeddings: np.ndarray,
    *,
    min_speakers: int = 1,
    max_speakers: int = 20,
    num_speakers: int | None = None,
    method: str = "sc",
    enhance: bool = True,
    estimate_method: str = "gmm_bic",
    silhouette_refine: bool = False,
    ahc_threshold: float = 0.5,
    greedy_threshold: float = 0.5,
    ahc_percentile: float | None = None,
    greedy_percentile: float | None = None,
    min_cluster_size: int = 1,
    merge_threshold: float | None = None,
) -> tuple[np.ndarray, SpeakerEstimationDetails | None]:
    """Estimate speaker count (if needed) then cluster.

    Args:
        method: "sc" | "ahc" | "meanshift" | "ahc_threshold" | "cosine_greedy"
        estimate_method: "gmm_bic" | "nmesc" (ignored for threshold/greedy methods)
        silhouette_refine: try k, k+1, k+2 and pick best silhouette (SC only).
        ahc_threshold: cosine distance threshold for ahc_threshold method.
        greedy_threshold: cosine similarity threshold for cosine_greedy method.
    """
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int), None

    emb = normalize(embeddings, norm="l2")

    # Methods that don't need k estimation
    if method == "meanshift":
        labels = cluster_meanshift(emb)
        return labels, None

    if method == "ahc_threshold":
        labels = cluster_ahc_threshold(
            emb, threshold=ahc_threshold,
            min_speakers=min_speakers, max_speakers=max_speakers,
            percentile=ahc_percentile,
            min_cluster_size=min_cluster_size,
            merge_threshold=merge_threshold)
        n_spk = len(set(labels.tolist()))
        reason = (f"distance_percentile={ahc_percentile}" if ahc_percentile is not None
                  else f"distance_threshold={ahc_threshold}")
        details = SpeakerEstimationDetails(
            method="ahc_threshold", best_k=n_spk, reason=reason)
        return labels, details

    if method == "cosine_greedy":
        labels = cluster_cosine_greedy(
            emb, threshold=greedy_threshold, max_speakers=max_speakers,
            percentile=greedy_percentile)
        n_spk = len(set(labels.tolist()))
        reason = (f"similarity_percentile={greedy_percentile}" if greedy_percentile is not None
                  else f"similarity_threshold={greedy_threshold}")
        details = SpeakerEstimationDetails(
            method="cosine_greedy", best_k=n_spk, reason=reason)
        return labels, details

    # Estimate k
    details: SpeakerEstimationDetails | None = None
    if num_speakers is not None:
        k = num_speakers
    elif estimate_method == "nmesc":
        k = estimate_speakers_nmesc(emb, min_speakers, max_speakers)
        details = SpeakerEstimationDetails(method="nmesc", best_k=k)
    else:
        k, details = estimate_speakers_gmm_bic(emb, min_speakers, max_speakers)

    # Optional silhouette refinement: cluster at k, then try merging the two
    # closest centroids iteratively (k→k-1→k-2) and keep whichever k gives
    # the best silhouette. One SC call + cheap centroid merges, no re-clustering.
    if (silhouette_refine and method == "sc"
            and num_speakers is None and k >= 3 and len(emb) >= 4):
        labels_k = cluster_sc(emb, k, enhance=enhance)
        distance = np.maximum(1 - (cosine_similarity(emb) + 1) / 2, 0)
        best_labels = labels_k
        best_sil = silhouette_score(distance, labels_k, metric="precomputed")
        best_k = k
        current_labels = labels_k.copy()
        for attempt_k in range(k - 1, max(1, k - 3), -1):
            uniq = np.unique(current_labels)
            if len(uniq) < 2:
                break
            C = np.stack([emb[current_labels == c].mean(axis=0) for c in uniq])
            D = pairwise_distances(C, metric="cosine")
            np.fill_diagonal(D, np.inf)
            i, j = np.unravel_index(np.argmin(D), D.shape)
            current_labels = current_labels.copy()
            current_labels[current_labels == uniq[j]] = uniq[i]
            _, current_labels = np.unique(current_labels, return_inverse=True)
            sil = silhouette_score(distance, current_labels, metric="precomputed")
            if sil > best_sil:
                best_sil = sil
                best_labels = current_labels.copy()
                best_k = attempt_k
        if details is not None:
            details.best_k = best_k
        return best_labels, details  # type: ignore[return-value]

    if method == "ahc":
        labels = cluster_ahc(emb, k)
    else:
        labels = cluster_sc(emb, k, enhance=enhance)
    return labels, details
