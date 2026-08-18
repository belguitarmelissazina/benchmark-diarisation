"""FastAPI backend — data access layer over MLflow for the React dashboard.

Responsibilities:
  - List / filter MLflow runs
  - Serve per-run metrics, params, tags
  - Stream raw artifacts (VAD, embeddings, UMAP, similarity, labels, RTTM)
  - Stream audio (full file or a segment for click-to-play)
  - Downsample heavy arrays so the frontend stays responsive on a 16GB laptop

Run with:
  cd backend
  python -m uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import mlflow
from mlflow.tracking import MlflowClient

# ── Config ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
# SQLite backend store (metadata) + ./mlruns artifact root. Must match the
# URI used by diar_pipeline.tracking and the `mlflow ui` command.
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}",
)
mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(title="Diarization Analysis API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


def client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


# ── Artifact helpers ─────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _cached_npy(path_str: str) -> np.ndarray:
    """LRU cache for numpy artifacts — avoids rereading large matrices."""
    return np.load(path_str)


def _run_artifact_root(run_id: str) -> Path:
    """Resolve on-disk path to a run's artifacts folder (file-store only)."""
    from urllib.parse import unquote, urlparse
    r = client().get_run(run_id)
    uri = r.info.artifact_uri
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        # On Windows, parsed.path looks like "/C:/Users/..."; strip the leading "/"
        raw = unquote(parsed.path)
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        p = Path(raw)
    else:
        # Non-file store: fall through to mlflow's download helper.
        p = Path(client().download_artifacts(run_id, ""))
    return p


def _artifact_path(run_id: str, rel: str) -> Path:
    root = _run_artifact_root(run_id)
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise HTTPException(status_code=400, detail="Path traversal blocked")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {rel}")
    return p


def _load_json(run_id: str, rel: str) -> Any:
    p = _artifact_path(run_id, rel)
    return json.loads(p.read_text(encoding="utf-8"))


def _load_npy(run_id: str, rel: str) -> np.ndarray:
    p = _artifact_path(run_id, rel)
    return _cached_npy(str(p))


# ── Schemas-ish ──────────────────────────────────────────────────────────────

def _run_summary(r) -> dict:
    info, data = r.info, r.data
    return {
        "run_id": info.run_id,
        "run_name": data.tags.get("mlflow.runName", ""),
        "experiment_id": info.experiment_id,
        "status": info.status,
        "start_time": info.start_time,
        "end_time": info.end_time,
        "duration_ms": (info.end_time - info.start_time) if info.end_time else None,
        "tags": {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")},
        "params": dict(data.params),
        "metrics": dict(data.metrics),
    }


# ── Endpoints: experiments & runs ────────────────────────────────────────────

@app.get("/api/experiments")
def list_experiments():
    exps = client().search_experiments()
    return [
        {"experiment_id": e.experiment_id, "name": e.name,
         "artifact_location": e.artifact_location}
        for e in exps
    ]


@app.get("/api/runs")
def list_runs(
    experiment: str = Query(""),
    limit: int = Query(200, ge=1, le=1000),
    order_by: str = Query("start_time DESC"),
    filter_string: str = Query("attributes.status = 'FINISHED'"),
):
    """List runs for an experiment (or all experiments if empty), newest first."""
    if experiment:
        exp = client().get_experiment_by_name(experiment)
        if exp is None:
            return []
        exp_ids = [exp.experiment_id]
    else:
        exps = client().search_experiments()
        exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        return []
    runs = client().search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string,
        max_results=limit,
        order_by=[order_by],
    )
    return [_run_summary(r) for r in runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        r = client().get_run(run_id)
    except Exception:
        raise HTTPException(status_code=404, detail="run not found")
    summary = _run_summary(r)
    # Attach metric history for anything time-series-y (all metrics are single-point
    # in our pipeline except per-step; still cheap to expose full history).
    history: dict[str, list[dict]] = {}
    for key in r.data.metrics.keys():
        hist = client().get_metric_history(run_id, key)
        history[key] = [{"step": h.step, "value": h.value, "timestamp": h.timestamp}
                        for h in hist]
    summary["metric_history"] = history
    # Also surface the config.json if it exists
    try:
        summary["config"] = _load_json(run_id, "config.json")
    except HTTPException:
        summary["config"] = None
    return summary


# ── Endpoints: per-step data ─────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vad")
def get_vad(run_id: str):
    return _load_json(run_id, "vad/vad_segments.json")


@app.get("/api/runs/{run_id}/chunks")
def get_chunks(run_id: str):
    return _load_json(run_id, "embeddings/chunk_times.json")


@app.get("/api/runs/{run_id}/embeddings")
def get_embeddings(
    run_id: str,
    downsample: int = Query(1, ge=1, le=100,
                            description="Keep every Nth embedding"),
):
    """Return embeddings as a JSON list of lists. Use downsample for huge runs."""
    X = _load_npy(run_id, "embeddings/embeddings.npy")
    if downsample > 1:
        X = X[::downsample]
    return {
        "shape": list(X.shape),
        "downsample": downsample,
        "values": X.tolist(),
    }


@app.get("/api/runs/{run_id}/umap")
def get_umap(run_id: str, downsample: int = Query(1, ge=1, le=100)):
    """Return precomputed 2D UMAP (from the pipeline). One point per embedding chunk."""
    try:
        X = _load_npy(run_id, "embeddings/umap_2d.npy")
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail="umap_2d.npy not found for this run (older run?).",
        )
    if downsample > 1:
        X = X[::downsample]

    # Also pair with labels and chunk times for frontend convenience
    labels = None
    for candidate in ("clustering/labels_final.npy", "clustering/labels_pre_vbx.npy"):
        try:
            labels = _load_npy(run_id, candidate)
            if downsample > 1:
                labels = labels[::downsample]
            break
        except HTTPException:
            continue

    chunks = None
    try:
        chunk_meta = _load_json(run_id, "embeddings/chunk_times.json")
        chunks = chunk_meta.get("chunks", [])
        if downsample > 1:
            chunks = chunks[::downsample]
    except HTTPException:
        pass

    return {
        "shape": list(X.shape),
        "points": X.tolist(),
        "labels": labels.tolist() if labels is not None else None,
        "chunks": chunks,
        "downsample": downsample,
    }


@app.get("/api/runs/{run_id}/labels")
def get_labels(run_id: str, stage: str = Query("final")):
    """Return cluster labels. stage = 'final' | 'pre_vbx'."""
    rel = f"clustering/labels_{stage}.npy"
    X = _load_npy(run_id, rel)
    return {"stage": stage, "shape": list(X.shape), "labels": X.tolist()}


@app.get("/api/runs/{run_id}/sim")
def get_sim(
    run_id: str,
    variant: str = Query("raw", pattern="^(raw|enhanced)$"),
    max_size: int = Query(512, ge=32, le=4096,
                          description="Max N for returned NxN matrix"),
):
    """Return the similarity matrix. Auto-downsample if larger than max_size."""
    rel = f"similarity/sim_matrix_{variant}.npy"
    M = _load_npy(run_id, rel)
    n = M.shape[0]
    stride = max(1, (n + max_size - 1) // max_size)
    if stride > 1:
        M = M[::stride, ::stride]
    return {
        "variant": variant,
        "original_shape": [n, n],
        "shape": list(M.shape),
        "stride": stride,
        "values": M.astype(float).round(4).tolist(),
    }


@app.get("/api/runs/{run_id}/ram")
def get_ram(run_id: str):
    return _load_json(run_id, "ram_timeseries.json")


@app.get("/api/runs/{run_id}/transcript")
def get_transcript(run_id: str):
    """Return speaker-attributed transcript, both alignment modes.

    Response shape:
      {
        "modes": {
          "midpoint": {"words": [...], "turns": [...]},
          "boundary": {"words": [...], "turns": [...]}
        },
        "words_per_speaker": {...},
        # Legacy (= midpoint) for back-compat:
        "words": [...],
        "turns": [...]
      }
    """
    def try_load(path: str):
        try:
            return _load_json(run_id, path)
        except HTTPException:
            return None

    modes: dict = {}
    for name in ("midpoint", "boundary"):
        w = try_load(f"transcript/words_{name}.json")
        t = try_load(f"transcript/turns_{name}.json")
        if w is not None or t is not None:
            modes[name] = {"words": w, "turns": t}

    # Legacy fallback: older runs only have words.json / turns.json
    legacy_words = try_load("transcript/words.json")
    legacy_turns = try_load("transcript/turns.json")
    if not modes and (legacy_words is not None or legacy_turns is not None):
        modes["midpoint"] = {"words": legacy_words, "turns": legacy_turns}

    if not modes:
        raise HTTPException(404, "No transcript artifacts for this run")

    default = modes.get("midpoint") or next(iter(modes.values()))
    return {
        "modes": modes,
        "words_per_speaker": try_load("transcript/words_per_speaker.json"),
        "words": default.get("words"),
        "turns": default.get("turns"),
    }


@app.get("/api/runs/{run_id}/rttm")
def get_rttm(run_id: str):
    """Return RTTM as parsed JSON segments (easy to consume in React)."""
    root = _run_artifact_root(run_id)
    rttm_files = list(root.glob("*.rttm"))
    if not rttm_files:
        raise HTTPException(status_code=404, detail="No RTTM artifact")
    lines = rttm_files[0].read_text(encoding="utf-8").splitlines()
    segments = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 9 and parts[0] == "SPEAKER":
            segments.append({
                "start": float(parts[3]),
                "duration": float(parts[4]),
                "end": float(parts[3]) + float(parts[4]),
                "speaker": parts[7],
            })
    return {"file": rttm_files[0].name, "segments": segments}


# ── Endpoints: audio streaming ───────────────────────────────────────────────

def _resolve_audio(run_id: str) -> Path:
    """Prefer the 16 kHz mono WAV logged under audio/; else convert input_file on the fly."""
    root = _run_artifact_root(run_id)
    audio_dir = root / "audio"
    if audio_dir.exists():
        wavs = sorted(audio_dir.glob("*.wav"))
        if wavs:
            return wavs[0]
    cfg = _load_json(run_id, "config.json")
    p = Path(cfg.get("input_file", ""))
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Audio not found: {p}")
    # If source isn't wav, convert to a cached wav in the artifacts dir
    if p.suffix.lower() not in (".wav",):
        import subprocess, tempfile
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise HTTPException(500, "imageio_ffmpeg needed to decode non-wav audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        cached = audio_dir / f"{p.stem}_16k.wav"
        if not cached.exists():
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(p),
                 "-ac", "1", "-ar", "16000", str(cached)],
                check=True,
            )
        return cached
    return p


@app.get("/api/runs/{run_id}/audio")
def get_audio(
    run_id: str,
    start: Optional[float] = Query(None, ge=0),
    end: Optional[float] = Query(None, ge=0),
):
    """Stream the source audio, or a [start, end] slice for click-to-play."""
    audio_path = _resolve_audio(run_id)

    # Full file: just stream the file directly (browser handles Range).
    if start is None and end is None:
        return FileResponse(str(audio_path),
                            media_type=_guess_audio_mime(audio_path))

    # Slice: read → trim → return as WAV in memory.
    try:
        import soundfile as sf
    except ImportError:
        raise HTTPException(500, "soundfile not installed")
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    s0 = int((start or 0.0) * sr)
    s1 = int((end or (len(data) / sr)) * sr)
    s0 = max(0, min(s0, len(data)))
    s1 = max(s0, min(s1, len(data)))
    clip = data[s0:s1]
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


def _guess_audio_mime(p: Path) -> str:
    ext = p.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


@app.get("/api/runs/{run_id}/waveform")
def get_waveform(
    run_id: str,
    max_points: int = Query(2000, ge=100, le=20000),
):
    """Return a downsampled waveform (min/max envelope per bucket) for drawing.

    Much smaller than raw audio; the frontend just draws polylines.
    """
    audio_path = _resolve_audio(run_id)
    import soundfile as sf
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    bucket = max(1, n // max_points)
    # Trim to whole buckets and reshape
    usable = (n // bucket) * bucket
    reshaped = data[:usable].reshape(-1, bucket)
    mn = reshaped.min(axis=1)
    mx = reshaped.max(axis=1)
    times = (np.arange(reshaped.shape[0]) * bucket / sr).astype(np.float32)
    return {
        "sample_rate": sr,
        "duration": n / sr,
        "bucket_size": bucket,
        "times": times.tolist(),
        "min": mn.astype(float).round(5).tolist(),
        "max": mx.astype(float).round(5).tolist(),
    }


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"ok": True, "tracking_uri": TRACKING_URI}
