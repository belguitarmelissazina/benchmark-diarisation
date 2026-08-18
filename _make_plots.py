"""Generate the figures for the benchmark report."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_report_data"
OUT = r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_report_data/figs"
os.makedirs(OUT, exist_ok=True)

def load(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))

# ---------- 1. Method ranking (overall DER) ----------
ms = load("method_summary.json")
ms = sorted(ms, key=lambda x: x["der_mean"])
fig, ax = plt.subplots(figsize=(11, 6.5))
names = [r["method"].replace("_"," ") for r in ms]
der = [r["der_mean"]*100 for r in ms]
n_files = [r["n_files"] for r in ms]
colors = ["#2ca02c" if d < 20 else ("#ff7f0e" if d < 35 else "#d62728") for d in der]
bars = ax.barh(names, der, color=colors, edgecolor="black")
for i,(b,d,n) in enumerate(zip(bars, der, n_files)):
    ax.text(b.get_width()+0.3, b.get_y()+b.get_height()/2,
            f"{d:.1f}% (n={n})", va="center", fontsize=9)
ax.set_xlabel("DER moyen (%) - moyenne arithmétique sur les fichiers évalués")
ax.set_title("Classement des méthodes (DER moyen, sur tous les fichiers évalués)")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_method_ranking.png"), dpi=140)
plt.close()
print("01_method_ranking.png")

# ---------- 2. DER per file × method (heatmap, on common files) ----------
by_file = load("by_file.json")
# Pick the most-evaluated files
common_files = ["IS1009a", "EN2002c", "018a_EARZ", "069c_EEPL"]
methods_to_show = [
    "resnet34_gmm_bic_sc_enh",
    "resnet34_nmesc_sc_enh",
    "resnet34_gmm_bic_ahc_threshold",
    "resnet34_gmm_bic_cosine_greedy",
    "resnet34_nmesc_sc_enh_vbx(0.05,5.0)",
    "ecapa_nmesc_sc_enh",
    "campplus_nmesc_sc_enh",
    "pyannote_onnx",
]
mat = np.full((len(methods_to_show), len(common_files)), np.nan)
for j, fid in enumerate(common_files):
    for r in by_file.get(fid, []):
        m = r["method"]
        if m in methods_to_show:
            i = methods_to_show.index(m)
            mat[i,j] = r["der_mean"]*100

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(mat, cmap="RdYlGn_r", vmin=5, vmax=70, aspect="auto")
ax.set_xticks(range(len(common_files)))
ax.set_xticklabels(common_files, rotation=15, ha="right")
ax.set_yticks(range(len(methods_to_show)))
ax.set_yticklabels([m.replace("_"," ") for m in methods_to_show])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if not np.isnan(mat[i,j]):
            ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                    color="black", fontsize=10)
        else:
            ax.text(j, i, "—", ha="center", va="center", color="gray")
fig.colorbar(im, ax=ax, label="DER (%)")
ax.set_title("DER (%) par méthode et par fichier")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_heatmap_methods_files.png"), dpi=140)
plt.close()
print("02_heatmap_methods_files.png")

# ---------- 3. Time per step (stacked bar) on BEST_CONFIG files ----------
steps = load("best_per_file_steps.json")
fig, ax = plt.subplots(figsize=(10, 5))
files = [r["file_id"] for r in steps]
vad = np.array([r["t_vad"] for r in steps])
emb = np.array([r["t_emb"] for r in steps])
clu = np.array([r["t_clu"] for r in steps])
other = np.array([r["t_tot"] for r in steps]) - (vad+emb+clu)
ax.bar(files, vad, color="#1f77b4", label="VAD (Silero)")
ax.bar(files, emb, bottom=vad, color="#ff7f0e", label="Embeddings (ResNet34 ONNX)")
ax.bar(files, clu, bottom=vad+emb, color="#2ca02c", label="Clustering (SC + sim_enh)")
ax.bar(files, other, bottom=vad+emb+clu, color="#7f7f7f", label="Autre (I/O, conversion, RTTM)")
# audio length annotation
audio = [r["audio_s"] for r in steps]
for i,a in enumerate(audio):
    tot = vad[i]+emb[i]+clu[i]+other[i]
    ax.text(i, tot+5, f"audio={a/60:.1f} min\nRTF={tot/a:.2f}",
            ha="center", fontsize=9)
ax.set_ylabel("Temps (secondes)")
ax.set_title("Décomposition du temps par étape (resnet34 / sc-enh / auto-k)")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_time_per_step.png"), dpi=140)
plt.close()
print("03_time_per_step.png")

# ---------- 4. RTF vs DER scatter ----------
fig, ax = plt.subplots(figsize=(10, 6))
for r in ms:
    if r["rtf_mean"] is None: continue
    ax.scatter(r["rtf_mean"], r["der_mean"]*100, s=120, edgecolor="black",
               alpha=0.85)
    ax.annotate(r["method"].split("_")[0]+"\n"+"_".join(r["method"].split("_")[1:]),
                (r["rtf_mean"], r["der_mean"]*100),
                xytext=(6,3), textcoords="offset points", fontsize=8)
ax.set_xlabel("RTF (Real-Time Factor) — plus bas = plus rapide")
ax.set_ylabel("DER moyen (%) — plus bas = meilleur")
ax.set_title("Compromis vitesse/qualité (chaque point = une méthode)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_rtf_vs_der.png"), dpi=140)
plt.close()
print("04_rtf_vs_der.png")

# ---------- 5. Embedding compare on common (file × estimate) ----------
ec = load("emb_compare.json")
# Keep only rows where ≥2 embeddings available
keep = [r for r in ec if sum(1 for k in ("resnet34_der","ecapa_der","campplus_der") if k in r) >= 2]
fig, ax = plt.subplots(figsize=(11, 5.5))
labels = [f"{r['file_id']}/{r['estimate']}" for r in keep]
x = np.arange(len(labels))
width = 0.27
def vals(k): return [r.get(k, np.nan)*100 if r.get(k) is not None else np.nan for r in keep]
ax.bar(x-width, vals("resnet34_der"), width, label="ResNet34-LM (256-d)", color="#1f77b4")
ax.bar(x,        vals("ecapa_der"),    width, label="ECAPA-TDNN (192-d)", color="#ff7f0e")
ax.bar(x+width, vals("campplus_der"),  width, label="CAM++ LM (512-d)", color="#2ca02c")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("DER (%)")
ax.set_title("Comparaison des embeddings (sc + sim-enh, auto-k)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_emb_compare.png"), dpi=140)
plt.close()
print("05_emb_compare.png")

# ---------- 6. Cluster method compare ----------
cc = load("cluster_compare.json")
fig, ax = plt.subplots(figsize=(10, 5))
labels = [r["file_id"] for r in cc]
x = np.arange(len(labels))
keys = ["sc_enh", "ahc_threshold", "cosine_greedy"]
colors = ["#1f77b4","#ff7f0e","#2ca02c"]
width = 0.26
for i, (k, c) in enumerate(zip(keys, colors)):
    vals = [(r.get(k) or np.nan)*100 for r in cc]
    ax.bar(x + (i-1)*width, vals, width, label=k, color=c)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("DER (%)")
ax.set_title("Algorithmes de clustering (resnet34, gmm_bic, auto-k)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_cluster_compare.png"), dpi=140)
plt.close()
print("06_cluster_compare.png")

# ---------- 7. VBx vs no-VBx ----------
vb = load("vbx_compare.json")
fig, ax = plt.subplots(figsize=(8, 5))
labels = [r["file_id"] for r in vb]
x = np.arange(len(labels))
no_vbx = [r.get("no_vbx",np.nan)*100 for r in vb]
yes_vbx = [r.get("vbx",np.nan)*100 for r in vb]
w = 0.38
ax.bar(x-w/2, no_vbx, w, label="sans VBx", color="#1f77b4")
ax.bar(x+w/2, yes_vbx, w, label="avec VBx (Fa=0.05, Fb=5.0)", color="#d62728")
for i,(a,b) in enumerate(zip(no_vbx, yes_vbx)):
    delta = b - a
    ax.text(i, max(a,b)+1, f"{delta:+.1f}", ha="center", fontsize=9,
            color="green" if delta<0 else "red")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("DER (%)")
ax.set_title("Effet du raffinement VBx (HMM Variational Bayes)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_vbx.png"), dpi=140)
plt.close()
print("07_vbx.png")

# ---------- 8. Baseline pyannote-onnx vs best custom ----------
bc = load("baseline_compare.json")
keep = [r for r in bc if r["pyannote_der"] is not None]
fig, ax = plt.subplots(figsize=(8, 5))
labels = [r["file_id"] for r in keep]
x = np.arange(len(labels))
w = 0.38
py = [r["pyannote_der"]*100 for r in keep]
cu = [r["best_der"]*100 for r in keep]
ax.bar(x-w/2, py, w, label="pyannote-onnx 3.1 (baseline)", color="#7f7f7f")
ax.bar(x+w/2, cu, w, label="meilleur pipeline custom", color="#2ca02c")
for i,(a,b) in enumerate(zip(py, cu)):
    delta = b - a
    ax.text(i, max(a,b)+1, f"{delta:+.1f}", ha="center", fontsize=9,
            color="green" if delta<0 else "red")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("DER (%)")
ax.set_title("Baseline pyannote-onnx 3.1 vs meilleur pipeline custom")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_baseline.png"), dpi=140)
plt.close()
print("08_baseline.png")

# ---------- 9. Auto-k vs forced-k ----------
ns = load("nspk_compare.json")
fig, ax = plt.subplots(figsize=(7,4.5))
labels = [r["file_id"] for r in ns]
x = np.arange(len(labels))
auto = [r.get("k_auto", np.nan)*100 for r in ns]
forced_keys = []
for r in ns:
    for k in r:
        if k.startswith("k_") and k != "k_auto":
            forced_keys.append((r["file_id"], k, r[k]*100))
forced_vals = []
forced_labs = []
for fid in labels:
    fk = [(k,v) for f,k,v in forced_keys if f==fid]
    if fk:
        forced_labs.append(fk[0][0])
        forced_vals.append(fk[0][1])
    else:
        forced_labs.append("-")
        forced_vals.append(np.nan)
w = 0.38
ax.bar(x-w/2, auto, w, label="k auto (estimateur)", color="#1f77b4")
ax.bar(x+w/2, forced_vals, w, label=f"k forcé ({forced_labs[0]} / {forced_labs[1] if len(forced_labs)>1 else '-'})", color="#d62728")
for i,(a,b) in enumerate(zip(auto, forced_vals)):
    if not np.isnan(b) and not np.isnan(a):
        ax.text(i, max(a,b)+1, f"{b-a:+.1f}", ha="center", fontsize=9,
                color="green" if b<a else "red")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("DER (%)")
ax.set_title("Estimation automatique de k vs k connu (oracle)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_auto_vs_forced_k.png"), dpi=140)
plt.close()
print("09_auto_vs_forced_k.png")

# ---------- 10. DER components stacked (miss / FA / confusion) on best methods ----------
import csv
csv_rows = list(csv.DictReader(open(r"c:/Users/MelissaBELGUITAR/OneDrive - YELE CONSULTING/Bureau/diarisation+transcription/_runs_summary.csv", encoding="utf-8")))
def f(x):
    try: return float(x)
    except: return None
for r in csv_rows:
    for k in ("der","miss","false_alarm","confusion"):
        r[k] = f(r[k])

# average miss/FA/conf per method on common files
from collections import defaultdict
import statistics as st
mc = defaultdict(lambda: defaultdict(list))
for r in csv_rows:
    if r["der"] is None: continue
    if r["embed"]=="" :
        m = "pyannote_onnx"
    else:
        m = r["embed"]
        if r["estimate"]: m += "/"+r["estimate"]
        if r["cluster"]: m += "/"+r["cluster"]
        if r["enhance"]=="True": m += "+enh"
        if r["refine_vbx"]=="True": m += "+vbx"
    mc[m]["miss"].append(r["miss"])
    mc[m]["fa"].append(r["false_alarm"])
    mc[m]["conf"].append(r["confusion"])

# Keep top methods (most-runs)
keep_methods = sorted(mc, key=lambda m: -len(mc[m]["miss"]))[:8]
fig, ax = plt.subplots(figsize=(11, 5.5))
y = np.arange(len(keep_methods))
miss = np.array([st.mean(mc[m]["miss"])*100 if mc[m]["miss"] else 0 for m in keep_methods])
fa = np.array([st.mean(mc[m]["fa"])*100 if mc[m]["fa"] else 0 for m in keep_methods])
conf = np.array([st.mean(mc[m]["conf"])*100 if mc[m]["conf"] else 0 for m in keep_methods])
ax.barh(y, miss, color="#1f77b4", label="Miss (silence prédit, parole de référence)")
ax.barh(y, fa, left=miss, color="#ff7f0e", label="False alarm (parole prédite, silence ref)")
ax.barh(y, conf, left=miss+fa, color="#d62728", label="Confusion (mauvais speaker)")
ax.set_yticks(y); ax.set_yticklabels(keep_methods, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("DER décomposé (%)")
ax.set_title("Décomposition de la DER : Miss + FA + Confusion")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "10_der_components.png"), dpi=140)
plt.close()
print("10_der_components.png")

print("\nAll plots saved to:", OUT)
