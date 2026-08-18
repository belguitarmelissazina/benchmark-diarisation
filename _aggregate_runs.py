import json, os, glob, csv

ROOT = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/mlruns"
rows = []
for run_dir in sorted(os.listdir(ROOT)):
    art = os.path.join(ROOT, run_dir, "artifacts")
    if not os.path.isdir(art):
        continue
    cfg_path = os.path.join(art, "config.json")
    der_path = os.path.join(art, "der.json")
    if not os.path.exists(cfg_path):
        continue
    try:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    except Exception as e:
        print("FAIL cfg", run_dir, e); continue
    der = {}
    if os.path.exists(der_path):
        try: der = json.load(open(der_path, encoding="utf-8"))
        except Exception: pass
    p = cfg.get("params", {})
    t = cfg.get("timings", {})
    r = cfg.get("results", {})
    row = {
        "run_id": run_dir,
        "pipeline_name": cfg.get("pipeline_name", ""),
        "file_id": cfg.get("file_id", ""),
        "audio_duration": cfg.get("audio_duration", 0),
        "embed": p.get("embed", ""),
        "estimate": p.get("estimate", ""),
        "cluster": p.get("cluster", ""),
        "enhance": p.get("enhance", False),
        "win_len": p.get("win_len", None),
        "hop_len": p.get("hop_len", None),
        "refine_vbx": p.get("refine_vbx", False),
        "vbx_Fa": p.get("vbx_Fa", None),
        "vbx_Fb": p.get("vbx_Fb", None),
        "num_speakers_param": p.get("num_speakers", None),
        "min_speakers": p.get("min_speakers", None),
        "max_speakers": p.get("max_speakers", None),
        "t_conversion": t.get("time_conversion", 0),
        "t_vad": t.get("time_vad", 0),
        "t_embeddings": t.get("time_embeddings", 0),
        "t_similarity": t.get("time_similarity", 0),
        "t_clustering": t.get("time_clustering", 0),
        "t_refine": t.get("time_refinement", 0),
        "num_speakers_pred": r.get("num_speakers", None),
        "num_segments": r.get("num_segments", None),
        "total_time": r.get("total_time", None),
        "rtf": r.get("rtf", None),
        "der": der.get("der", None),
        "miss": der.get("miss", None),
        "false_alarm": der.get("false_alarm", None),
        "confusion": der.get("confusion", None),
        "total_sec_ref": der.get("total_sec", None),
    }
    # Detect pyannote/onnx pipelines (different params)
    if "pyannote" in cfg.get("pipeline_name", "").lower():
        row["embed"] = row["embed"] or "pyannote"
    rows.append(row)

# Write CSV
out_csv = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_runs_summary.csv"
fields = list(rows[0].keys()) if rows else []
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in rows:
        w.writerow(row)
print(f"Wrote {len(rows)} runs to {out_csv}")

# Quick stats
files = sorted({r["file_id"] for r in rows})
print("Files:", files)
embeds = sorted({r["embed"] for r in rows})
print("Embeds:", embeds)
clusters = sorted({r["cluster"] for r in rows})
print("Clusters:", clusters)
estimates = sorted({r["estimate"] for r in rows})
print("Estimates:", estimates)
