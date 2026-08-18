"""Compute aggregated tables / statistics for the benchmark report."""
import csv, json, os
from collections import defaultdict
import statistics as st

CSV = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_runs_summary.csv"
OUT_DIR = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_report_data"
os.makedirs(OUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

def f(x):
    try:
        return float(x)
    except Exception:
        return None

for r in rows:
    for k in ("audio_duration","t_conversion","t_vad","t_embeddings","t_similarity","t_clustering","t_refine","total_time","rtf","der","miss","false_alarm","confusion","total_sec_ref","win_len","hop_len","vbx_Fa","vbx_Fb"):
        r[k] = f(r[k])
    r["enhance"] = (r["enhance"]=="True")
    r["refine_vbx"] = (r["refine_vbx"]=="True")

# Tag method short names
def method_label(r):
    if not r["embed"]:
        return "pyannote_onnx"
    parts = []
    parts.append(r["embed"])
    parts.append(r["estimate"] or "-")
    parts.append(r["cluster"] or "-")
    if r["enhance"] and r["cluster"] == "sc":
        parts.append("enh")
    if r["refine_vbx"]:
        parts.append(f"vbx({r['vbx_Fa']},{r['vbx_Fb']})")
    if r["num_speakers_param"]:
        parts.append(f"k={r['num_speakers_param']}")
    return "_".join(parts)

for r in rows:
    r["method"] = method_label(r)

# Durations of files (audio length in s)
file_durations = {}
for r in rows:
    if r["file_id"] and r["audio_duration"]:
        file_durations[r["file_id"]] = r["audio_duration"]

print("File durations (s):")
for fid, d in sorted(file_durations.items()):
    print(f"  {fid:25s} {d:>9.1f}  ({d/60:5.2f} min)")

# Average duplicates: group by (file, method) and average DER, time
groups = defaultdict(list)
for r in rows:
    groups[(r["file_id"], r["method"])].append(r)

agg = []
for (fid, m), rs in groups.items():
    der_vals = [r["der"] for r in rs if r["der"] is not None]
    if not der_vals:
        continue
    rec = {
        "file_id": fid,
        "method": m,
        "n_runs": len(rs),
        "der_mean": st.mean(der_vals),
        "der_std": st.stdev(der_vals) if len(der_vals)>1 else 0.0,
        "der_min": min(der_vals),
        "der_max": max(der_vals),
        "miss_mean": st.mean([r["miss"] for r in rs if r["miss"] is not None]) if any(r["miss"] is not None for r in rs) else None,
        "fa_mean": st.mean([r["false_alarm"] for r in rs if r["false_alarm"] is not None]) if any(r["false_alarm"] is not None for r in rs) else None,
        "conf_mean": st.mean([r["confusion"] for r in rs if r["confusion"] is not None]) if any(r["confusion"] is not None for r in rs) else None,
        "rtf_mean": st.mean([r["rtf"] for r in rs if r["rtf"] is not None]) if any(r["rtf"] is not None for r in rs) else None,
        "total_time_mean": st.mean([r["total_time"] for r in rs if r["total_time"] is not None]) if any(r["total_time"] is not None for r in rs) else None,
        "t_vad_mean": st.mean([r["t_vad"] for r in rs if r["t_vad"] is not None]) if any(r["t_vad"] is not None for r in rs) else None,
        "t_embeddings_mean": st.mean([r["t_embeddings"] for r in rs if r["t_embeddings"] is not None]) if any(r["t_embeddings"] is not None for r in rs) else None,
        "t_clustering_mean": st.mean([r["t_clustering"] for r in rs if r["t_clustering"] is not None]) if any(r["t_clustering"] is not None for r in rs) else None,
        "t_refine_mean": st.mean([r["t_refine"] for r in rs if r["t_refine"] is not None]) if any(r["t_refine"] is not None for r in rs) else 0.0,
        "num_spk_pred": st.mode([int(r["num_speakers_pred"]) for r in rs if r["num_speakers_pred"]]) if any(r["num_speakers_pred"] for r in rs) else None,
    }
    agg.append(rec)

with open(os.path.join(OUT_DIR, "aggregated.json"), "w", encoding="utf-8") as fh:
    json.dump(agg, fh, indent=2)

print(f"\n# Aggregated rows: {len(agg)}")

# Best per file
print("\n# Best DER per file (averaged across duplicates):")
by_file = defaultdict(list)
for r in agg:
    by_file[r["file_id"]].append(r)
for fid in sorted(by_file):
    rs = sorted(by_file[fid], key=lambda x: x["der_mean"])
    print(f"  {fid}:")
    for r in rs[:3]:
        print(f"    {r['method']:55s}  DER={r['der_mean']*100:6.2f}% (n={r['n_runs']})")

# Group by method only - average DER across files
m_groups = defaultdict(list)
for r in agg:
    m_groups[r["method"]].append(r)

print("\n# Method overall (averaged across files where evaluated):")
method_summary = []
for m, rs in m_groups.items():
    rec = {
        "method": m,
        "n_files": len(set(r["file_id"] for r in rs)),
        "n_runs_total": sum(r["n_runs"] for r in rs),
        "der_mean": st.mean([r["der_mean"] for r in rs]),
        "der_std": st.stdev([r["der_mean"] for r in rs]) if len(rs)>1 else 0.0,
        "rtf_mean": st.mean([r["rtf_mean"] for r in rs if r["rtf_mean"] is not None]) if any(r["rtf_mean"] is not None for r in rs) else None,
        "files": sorted({r["file_id"] for r in rs}),
    }
    method_summary.append(rec)
method_summary.sort(key=lambda x: x["der_mean"])

for r in method_summary:
    print(f"  {r['method']:55s}  DER={r['der_mean']*100:6.2f}%  RTF={(r['rtf_mean'] or 0):.3f}  files={r['n_files']}  runs={r['n_runs_total']}")

with open(os.path.join(OUT_DIR, "method_summary.json"), "w", encoding="utf-8") as fh:
    json.dump(method_summary, fh, indent=2)
with open(os.path.join(OUT_DIR, "by_file.json"), "w", encoding="utf-8") as fh:
    json.dump({k: v for k,v in by_file.items()}, fh, indent=2)

# Time-per-step share — averaged on the BEST_CONFIG (resnet34, gmm_bic, sc, enh)
best = [r for r in rows if r["embed"]=="resnet34" and r["estimate"]=="gmm_bic" and r["cluster"]=="sc" and r["enhance"] and not r["refine_vbx"] and r["num_speakers_param"] in (None,"")]
print(f"\n# resnet34/gmm_bic/sc-enh runs (no vbx, auto-k): {len(best)}")
if best:
    by_file_best = defaultdict(list)
    for r in best:
        by_file_best[r["file_id"]].append(r)
    print(f"\n# Time per step on BEST_CONFIG (avg per file):")
    print(f"  {'file':25s}  {'audio_s':>8s}  {'vad':>6s}  {'emb':>7s}  {'clu':>6s}  {'tot':>7s}  {'rtf':>6s}  {'der%':>7s}")
    rows_best_step = []
    for fid in sorted(by_file_best):
        rs = by_file_best[fid]
        d = file_durations.get(fid, 0)
        ts_vad = st.mean([r["t_vad"] for r in rs if r["t_vad"] is not None]) if any(r["t_vad"] for r in rs) else 0
        ts_emb = st.mean([r["t_embeddings"] for r in rs if r["t_embeddings"] is not None]) if any(r["t_embeddings"] for r in rs) else 0
        ts_clu = st.mean([r["t_clustering"] for r in rs if r["t_clustering"] is not None]) if any(r["t_clustering"] for r in rs) else 0
        ts_tot = st.mean([r["total_time"] for r in rs if r["total_time"] is not None]) if any(r["total_time"] for r in rs) else 0
        rtf = st.mean([r["rtf"] for r in rs if r["rtf"] is not None]) if any(r["rtf"] for r in rs) else 0
        der = st.mean([r["der"] for r in rs if r["der"] is not None]) if any(r["der"] for r in rs) else None
        print(f"  {fid:25s}  {d:>8.1f}  {ts_vad:>6.1f}  {ts_emb:>7.1f}  {ts_clu:>6.1f}  {ts_tot:>7.1f}  {rtf:>6.3f}  {der*100 if der else 0:>7.2f}")
        rows_best_step.append({
            "file_id": fid, "audio_s": d, "t_vad": ts_vad, "t_emb": ts_emb,
            "t_clu": ts_clu, "t_tot": ts_tot, "rtf": rtf, "der": der})
    with open(os.path.join(OUT_DIR, "best_per_file_steps.json"), "w", encoding="utf-8") as fh:
        json.dump(rows_best_step, fh, indent=2)

# Compare embeddings — only on files where all 3 were tried (and same cluster=sc)
emb_compare = defaultdict(dict)
for r in rows:
    if r["cluster"]!="sc" or not r["enhance"] or r["refine_vbx"] or r["num_speakers_param"]:
        continue
    if r["embed"] not in ("resnet34","ecapa","campplus"):
        continue
    if r["der"] is None: continue
    key = (r["file_id"], r["estimate"])
    emb_compare[key].setdefault(r["embed"], []).append(r)

print("\n# Embedding compare (per file, per estimate, sc+enh, auto-k):")
embcomp_records = []
for (fid, est), e in sorted(emb_compare.items()):
    if len(e) < 2: continue
    line = f"  {fid:25s} est={est:9s} | "
    rec = {"file_id": fid, "estimate": est}
    for em in ("resnet34","ecapa","campplus"):
        if em in e:
            ders = [r["der"] for r in e[em]]
            rtfs = [r["rtf"] for r in e[em] if r["rtf"] is not None]
            der = st.mean(ders)
            rtf = st.mean(rtfs) if rtfs else 0
            line += f"{em}: DER={der*100:5.2f}% RTF={rtf:.2f}  "
            rec[f"{em}_der"] = der
            rec[f"{em}_rtf"] = rtf
    embcomp_records.append(rec)
    print(line)
with open(os.path.join(OUT_DIR, "emb_compare.json"), "w", encoding="utf-8") as fh:
    json.dump(embcomp_records, fh, indent=2)

# Cluster method compare on common files (resnet34)
print("\n# Cluster-method compare (resnet34, gmm_bic, auto-k):")
cl_compare = defaultdict(dict)
for r in rows:
    if r["embed"]!="resnet34" or r["estimate"]!="gmm_bic" or r["refine_vbx"] or r["num_speakers_param"]:
        continue
    if r["der"] is None: continue
    cl = r["cluster"]
    if r["enhance"] and cl == "sc": cl="sc_enh"
    elif cl=="sc": cl="sc"
    cl_compare[r["file_id"]].setdefault(cl, []).append(r)

cluster_recs = []
for fid, d in sorted(cl_compare.items()):
    if len(d) < 2: continue
    line = f"  {fid:25s} | "
    rec = {"file_id": fid}
    for cl in ("sc_enh","sc","ahc_threshold","cosine_greedy","ahc"):
        if cl in d:
            dr = st.mean([r["der"] for r in d[cl]])
            line += f"{cl}: DER={dr*100:5.2f}% (n={len(d[cl])})  "
            rec[cl] = dr
    cluster_recs.append(rec)
    print(line)
with open(os.path.join(OUT_DIR, "cluster_compare.json"), "w", encoding="utf-8") as fh:
    json.dump(cluster_recs, fh, indent=2)

# nspk vs auto: compare same file/method with diff num_speakers_param
print("\n# Effect of forcing num_speakers (resnet34, sc enhance):")
ns_groups = defaultdict(list)
for r in rows:
    if r["embed"]!="resnet34" or r["cluster"]!="sc" or not r["enhance"] or r["refine_vbx"]: continue
    if r["der"] is None: continue
    ns = r["num_speakers_param"] or "auto"
    ns_groups[(r["file_id"], r["estimate"])].append((ns, r))
nspk_recs = []
for (fid, est), rs in sorted(ns_groups.items()):
    nss = sorted({x[0] for x in rs})
    if len(nss) < 2: continue
    line = f"  {fid:25s} est={est:9s} | "
    rec = {"file_id": fid, "estimate": est}
    for ns in nss:
        ders = [x[1]["der"] for x in rs if x[0]==ns]
        line += f"k={ns}: DER={st.mean(ders)*100:5.2f}% (n={len(ders)})  "
        rec[f"k_{ns}"] = st.mean(ders)
    nspk_recs.append(rec)
    print(line)
with open(os.path.join(OUT_DIR, "nspk_compare.json"), "w", encoding="utf-8") as fh:
    json.dump(nspk_recs, fh, indent=2)

# VBx compare
print("\n# VBx vs no-VBx (resnet34, gmm_bic, sc enh):")
vbx_groups = defaultdict(dict)
for r in rows:
    if r["embed"]!="resnet34" or r["estimate"]!="nmesc" or r["cluster"]!="sc" or not r["enhance"]: continue
    if r["num_speakers_param"]: continue
    if r["der"] is None: continue
    key = "vbx" if r["refine_vbx"] else "no_vbx"
    vbx_groups[r["file_id"]].setdefault(key, []).append(r)
vbx_recs = []
for fid, d in sorted(vbx_groups.items()):
    if len(d)<2: continue
    line = f"  {fid:25s} | "
    rec = {"file_id": fid}
    for k in ("no_vbx","vbx"):
        if k in d:
            dr = st.mean([r["der"] for r in d[k]])
            line += f"{k}: DER={dr*100:5.2f}%  "
            rec[k] = dr
    vbx_recs.append(rec); print(line)
with open(os.path.join(OUT_DIR, "vbx_compare.json"), "w", encoding="utf-8") as fh:
    json.dump(vbx_recs, fh, indent=2)

# Pyannote-onnx baseline vs custom pipeline
print("\n# pyannote-onnx baseline vs custom best per file:")
pyann = {r["file_id"]: r for r in rows if r["embed"]=="" and r["der"] is not None}
custom_best = {}
for r in agg:
    fid = r["file_id"]
    if r["method"]=="pyannote_onnx": continue
    if fid not in custom_best or r["der_mean"] < custom_best[fid]["der_mean"]:
        custom_best[fid] = r
baseline_recs = []
for fid in sorted(pyann.keys() | custom_best.keys()):
    p = pyann.get(fid)
    c = custom_best.get(fid)
    rec = {"file_id": fid,
           "pyannote_der": p["der"] if p else None,
           "pyannote_rtf": p["rtf"] if p else None,
           "best_method": c["method"] if c else None,
           "best_der": c["der_mean"] if c else None,
           "best_rtf": c["rtf_mean"] if c else None}
    baseline_recs.append(rec)
    print(f"  {fid:25s} | pyannote: DER={(p['der']*100 if p else None)} RTF={(p['rtf'] if p else None)}  | best custom: {c['method'] if c else '-':45s} DER={(c['der_mean']*100 if c else None)} RTF={(c['rtf_mean'] if c else None)}")

with open(os.path.join(OUT_DIR, "baseline_compare.json"), "w", encoding="utf-8") as fh:
    json.dump(baseline_recs, fh, indent=2)

print("\nDone.")
