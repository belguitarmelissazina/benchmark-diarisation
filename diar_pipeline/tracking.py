"""MLflow tracking utilities for the diarization pipeline.

Provides:
  - Tracker           : context-managed MLflow run + RAM sampler thread
  - log_vad_artifacts
  - log_embedding_artifacts
  - log_sim_artifacts
  - log_clustering_artifacts
  - compute_der       : diarization error rate vs. reference RTTM
  - compute_silhouette

All artifacts are written as raw, machine-readable files (JSON / .npy)
so the frontend can rebuild visualizations dynamically instead of relying
on pre-rendered PNGs.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#   RAM sampler — background thread that records (t, rss_mb) every interval
# ─────────────────────────────────────────────────────────────────────────────

class RamSampler:
    """Background thread sampling process RSS at a fixed interval.

    Use as: sampler.start(); ...work...; sampler.mark("vad"); ...; sampler.stop()
    The `marks` list records phase boundaries for later alignment with metrics.
    """

    def __init__(self, interval: float = 0.25):
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process(os.getpid())
        self.interval = interval
        self.samples: list[tuple[float, float]] = []  # (t_rel, rss_mb)
        self.marks: list[dict[str, Any]] = []          # [{t, label}]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: float = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                self.samples.append((time.perf_counter() - self._t0, rss_mb))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark(self, label: str) -> None:
        self.marks.append({"t": time.perf_counter() - self._t0, "label": label})

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def peak_mb(self) -> float:
        return max((s[1] for s in self.samples), default=0.0)

    def mean_mb(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s[1] for s in self.samples]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_sec": self.interval,
            "samples": [{"t": round(t, 3), "rss_mb": round(m, 2)}
                        for t, m in self.samples],
            "marks": self.marks,
            "peak_mb": round(self.peak_mb(), 2),
            "mean_mb": round(self.mean_mb(), 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
#   Tracker — thin wrapper around mlflow that encapsulates run lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class Tracker:
    """Context manager for a single MLflow run.

    Usage:
        with Tracker(experiment="diarization", run_name="...") as tr:
            tr.log_params({...})
            tr.log_metric("time_vad", 1.2)
            tr.log_artifact_json({...}, "vad_segments.json")
            tr.log_artifact_npy(arr, "embeddings.npy")
            tr.ram.mark("vad")
    """

    def __init__(
        self,
        experiment: str = "diarization",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        ram_interval: float = 0.25,
    ):
        import mlflow
        self._mlflow = mlflow
        # Default to SQLite backend store (required for the MLflow UI Overview
        # tab) with artifacts living in ./mlruns at the repo root. Both the
        # tracking code and `mlflow ui` must point at the same mlflow.db.
        repo_root = Path(__file__).resolve().parent.parent
        if tracking_uri is None:
            db_path = (repo_root / "mlflow.db").as_posix()
            tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        # Ensure the experiment exists with artifacts rooted at ./mlruns, so
        # the backend API can resolve artifact files on disk directly.
        if mlflow.get_experiment_by_name(experiment) is None:
            artifact_location = (repo_root / "mlruns").as_uri()
            mlflow.create_experiment(experiment, artifact_location=artifact_location)
        mlflow.set_experiment(experiment)
        self._run_name = run_name
        self._tags = tags or {}
        self.ram = RamSampler(interval=ram_interval)
        self._active = None

    def __enter__(self) -> "Tracker":
        self._active = self._mlflow.start_run(run_name=self._run_name, tags=self._tags)
        self.run_id: str = self._active.info.run_id
        self.ram.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ram.stop()
        # Always log the RAM timeseries before closing the run
        try:
            self.log_artifact_json(self.ram.to_dict(), "ram_timeseries.json")
            self.log_metric("peak_ram_mb", self.ram.peak_mb())
            self.log_metric("mean_ram_mb", self.ram.mean_mb())
        except Exception:
            pass
        status = "FAILED" if exc_type else "FINISHED"
        self._mlflow.end_run(status=status)

    # ── logging helpers ──
    def log_params(self, params: dict[str, Any]) -> None:
        clean = {k: ("" if v is None else str(v)) for k, v in params.items()}
        self._mlflow.log_params(clean)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        try:
            self._mlflow.log_metric(key, float(value), step=step)
        except Exception:
            pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_artifact_json(self, obj: Any, filename: str, subdir: str = "") -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            self._mlflow.log_artifact(str(p), artifact_path=subdir or None)

    def log_artifact_npy(self, arr: np.ndarray, filename: str, subdir: str = "") -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            np.save(p, arr)
            self._mlflow.log_artifact(str(p), artifact_path=subdir or None)

    def log_artifact_file(self, path: str | Path, subdir: str = "") -> None:
        self._mlflow.log_artifact(str(path), artifact_path=subdir or None)


# ─────────────────────────────────────────────────────────────────────────────
#   Step-level artifact helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_vad_artifacts(tracker: Tracker, speech_segments, audio_duration: float) -> None:
    data = {
        "audio_duration": audio_duration,
        "num_segments": len(speech_segments),
        "segments": [{"start": s.start, "end": s.end} for s in speech_segments],
    }
    tracker.log_artifact_json(data, "vad_segments.json", subdir="vad")
    total_speech = sum(s.duration for s in speech_segments)
    tracker.log_metric("num_vad_segments", len(speech_segments))
    tracker.log_metric("speech_ratio",
                       total_speech / audio_duration if audio_duration > 0 else 0.0)
    tracker.log_metric("speech_duration_sec", total_speech)


def log_embedding_artifacts(
    tracker: Tracker,
    embeddings: np.ndarray,
    subsegments,
    embed_model: str,
    *,
    compute_umap: bool = True,
) -> None:
    tracker.log_artifact_npy(embeddings.astype(np.float32), "embeddings.npy",
                             subdir="embeddings")
    chunk_times = {
        "embed_model": embed_model,
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "n_chunks": int(embeddings.shape[0]),
        "chunks": [
            {"idx": i, "start": s.start, "end": s.end, "parent_idx": s.parent_idx}
            for i, s in enumerate(subsegments)
        ],
    }
    tracker.log_artifact_json(chunk_times, "chunk_times.json", subdir="embeddings")
    tracker.log_metric("num_embeddings", int(embeddings.shape[0]))
    tracker.log_metric("embedding_dim",
                       int(embeddings.shape[1]) if embeddings.ndim == 2 else 0)

    # Precompute 2D UMAP for the frontend (cosine metric, L2-normalised input).
    if compute_umap and len(embeddings) >= 4:
        try:
            import umap
            from sklearn.preprocessing import normalize
            emb_norm = normalize(embeddings, norm="l2")
            n_neighbors = min(15, max(2, len(embeddings) - 1))
            reducer = umap.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=42,
            )
            umap_2d = reducer.fit_transform(emb_norm).astype(np.float32)
            tracker.log_artifact_npy(umap_2d, "umap_2d.npy", subdir="embeddings")
        except Exception as e:
            print(f"    UMAP computation failed: {e}")


def log_sim_artifacts(
    tracker: Tracker,
    embeddings: np.ndarray,
    enhance: bool,
) -> None:
    """Compute and log raw (+enhanced if applicable) similarity matrices."""
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    if len(embeddings) < 2:
        return
    emb = normalize(embeddings, norm="l2")
    sim_raw = ((cosine_similarity(emb) + 1) / 2).astype(np.float32)
    np.fill_diagonal(sim_raw, 1.0)
    sim_raw = np.maximum(sim_raw, 0)
    tracker.log_artifact_npy(sim_raw, "sim_matrix_raw.npy", subdir="similarity")

    if enhance:
        from .clustering import sim_enhancement
        sim_enh = sim_enhancement(sim_raw.copy()).astype(np.float32)
        tracker.log_artifact_npy(sim_enh, "sim_matrix_enhanced.npy",
                                 subdir="similarity")


def log_clustering_artifacts(
    tracker: Tracker,
    labels: np.ndarray,
    embeddings: np.ndarray,
    stage: str = "final",
) -> None:
    """Log labels + silhouette score. `stage` distinguishes pre/post-VBx."""
    tracker.log_artifact_npy(labels.astype(np.int32),
                             f"labels_{stage}.npy", subdir="clustering")
    tracker.log_metric(f"num_speakers_{stage}", len(set(labels.tolist())))
    sil = compute_silhouette(embeddings, labels)
    if sil is not None:
        tracker.log_metric(f"silhouette_{stage}", sil)


# ─────────────────────────────────────────────────────────────────────────────
#   Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_silhouette(embeddings: np.ndarray,
                       labels: np.ndarray) -> Optional[float]:
    """Cosine-distance silhouette score. Returns None if undefined."""
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_similarity
        n_clusters = len(set(labels.tolist()))
        if n_clusters < 2 or len(embeddings) < n_clusters + 1:
            return None
        distance = np.maximum(1 - (cosine_similarity(embeddings) + 1) / 2, 0)
        return float(silhouette_score(distance, labels, metric="precomputed"))
    except Exception:
        return None


def compute_der(
    reference_rttm: str | Path,
    hypothesis_rttm: str | Path,
    collar: float = 0.25,
    skip_overlap: bool = True,
) -> Optional[dict[str, float]]:
    """Diarization Error Rate using pyannote.metrics.

    Returns a dict with {der, miss, false_alarm, confusion} or None on failure.
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
        from pyannote.database.util import load_rttm

        ref_map = load_rttm(str(reference_rttm))
        hyp_map = load_rttm(str(hypothesis_rttm))
        if not ref_map or not hyp_map:
            return None
        ref = next(iter(ref_map.values()))
        hyp = next(iter(hyp_map.values()))

        metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        der = metric(ref, hyp, detailed=True)
        total = der.get("total", 0.0) or 1.0
        return {
            "der": float(der.get("diarization error rate",
                                 der.get("der", 0.0))),
            "miss": float(der.get("missed detection", 0.0)) / total,
            "false_alarm": float(der.get("false alarm", 0.0)) / total,
            "confusion": float(der.get("confusion", 0.0)) / total,
            "total_sec": float(total),
        }
    except Exception as e:
        print(f"    DER computation failed: {e}")
        return None
