"""VBx — Variational Bayes HMM refinement of cluster labels.

Post-clustering refinement that models temporal continuity via a HMM:
  - States = speakers
  - Emissions = cos(embedding, centroid) scaled by Fb
  - Transitions: stay prob = 1 − Fa ; switch prob = Fa / (K − 1)
Forward-backward in log domain → posteriors → update centroids (M-step).
Ref: Landini et al., "VBx for speaker diarization", 2022.
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import normalize

from .models import SubSegment


def refine_vbx(
    embeddings: np.ndarray,
    labels: np.ndarray,
    subsegments: list[SubSegment],
    *,
    Fa: float = 0.05,
    Fb: float = 5.0,
    max_iter: int = 40,
    verbose: bool = True,
) -> np.ndarray:
    """Refine labels via Variational Bayes HMM.

    Args:
        embeddings: (N, D), any normalization (re-normalized inside).
        labels: initial cluster labels from clustering step.
        subsegments: used to sort by time.
        Fa: speaker-change probability (0..1). Lower → longer segments.
        Fb: speaker-model concentration. Higher → more confident centroids.
    """
    n = len(embeddings)
    speakers = sorted(set(labels))
    K = len(speakers)
    if K < 2 or n < 2:
        return labels

    spk_to_idx = {spk: i for i, spk in enumerate(speakers)}
    idx_labels = np.array([spk_to_idx[l] for l in labels])

    order = np.argsort([s.start for s in subsegments])
    emb_sorted = normalize(embeddings[order], norm="l2")
    lab_sorted = idx_labels[order]

    means = np.zeros((K, emb_sorted.shape[1]))
    for k in range(K):
        mask = lab_sorted == k
        if mask.sum() > 0:
            means[k] = emb_sorted[mask].mean(axis=0)
    means = normalize(means, norm="l2")

    log_stay = np.log(max(1.0 - Fa, 1e-10))
    log_switch = np.log(max(Fa / (K - 1), 1e-10)) if K > 1 else -np.inf
    log_trans = np.full((K, K), log_switch)
    np.fill_diagonal(log_trans, log_stay)

    gamma = None
    for iteration in range(max_iter):
        log_likes = Fb * (emb_sorted @ means.T)  # (N, K)

        # Forward
        log_alpha = np.full((n, K), -np.inf)
        log_alpha[0] = log_likes[0] + np.log(1.0 / K)
        for t in range(1, n):
            for k in range(K):
                log_alpha[t, k] = log_likes[t, k] + np.logaddexp.reduce(
                    log_alpha[t - 1] + log_trans[:, k])

        # Backward
        log_beta = np.zeros((n, K))
        for t in range(n - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_beta[t + 1] + log_likes[t + 1] + log_trans[k, :])

        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        new_means = gamma.T @ emb_sorted
        row_norms = np.linalg.norm(new_means, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        new_means = new_means / row_norms

        if np.allclose(means, new_means, atol=1e-4):
            if verbose:
                print(f"    VBx converged in {iteration + 1} iterations")
            break
        means = new_means
    else:
        if verbose:
            print(f"    VBx max iterations reached ({max_iter})")

    refined_sorted = np.argmax(gamma, axis=1)
    idx_to_spk = {i: spk for spk, i in spk_to_idx.items()}
    result = np.empty_like(labels)
    result[order] = np.array([idx_to_spk[l] for l in refined_sorted])

    if verbose:
        n_changes = int(np.sum(result != labels))
        print(f"    VBx: {n_changes}/{n} labels changed ({100*n_changes/n:.1f}%)")
    return result
