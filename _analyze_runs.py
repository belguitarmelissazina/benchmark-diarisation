import csv, os
from collections import defaultdict
import statistics as st

CSV = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_runs_summary.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

def f(x):
    try: return float(x)
    except: return None

for r in rows:
    for k in ("audio_duration","t_conversion","t_vad","t_embeddings","t_similarity","t_clustering","t_refine","total_time","rtf","der","miss","false_alarm","confusion","total_sec_ref","win_len","hop_len","vbx_Fa","vbx_Fb"):
        r[k] = f(r[k])
    r["enhance"] = (r["enhance"]=="True")
    r["refine_vbx"] = (r["refine_vbx"]=="True")

print(f"Total runs: {len(rows)}")
print(f"Runs with DER: {sum(1 for r in rows if r['der'] is not None)}")

# Files distribution
files_cnt = defaultdict(int)
for r in rows: files_cnt[r["file_id"]] += 1
print("\n# Runs per file_id:")
for k in sorted(files_cnt): print(f"  {k}: {files_cnt[k]}")

# Embed distribution
emb_cnt = defaultdict(int)
for r in rows: emb_cnt[(r["embed"], r["cluster"], r["estimate"])] += 1
print("\n# (embed, cluster, estimate) combos:")
for k in sorted(emb_cnt): print(f"  {k}: {emb_cnt[k]}")

# Detect duplicates: same file + same param tuple
dup_keys = defaultdict(list)
for r in rows:
    key = (r["file_id"], r["embed"], r["estimate"], r["cluster"], r["enhance"], r["win_len"], r["hop_len"], r["refine_vbx"], r["vbx_Fa"], r["vbx_Fb"], r["num_speakers_param"], r["min_speakers"], r["max_speakers"])
    dup_keys[key].append(r["run_id"])

print(f"\n# Unique parameter sets: {len(dup_keys)}")
dups = {k:v for k,v in dup_keys.items() if len(v)>1}
print(f"# Duplicate sets: {len(dups)}")
for k,v in list(dups.items())[:5]:
    print("  DUP", k, "->", v)

# Pipeline names
print("\n# Sample pipeline names:")
seen = set()
for r in rows:
    n = r["pipeline_name"]
    if n not in seen:
        seen.add(n);
print(f"# Unique pipeline_name: {len(seen)}")
for n in sorted(seen): print("  -", n)
