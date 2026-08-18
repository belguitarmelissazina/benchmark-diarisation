"""
Module de diagnostics pour le pipeline de diarisation.

Génère des visualisations interactives pour chaque étape :
  1. VAD       → forme d'onde avec zones de parole colorées
  2. Embeddings → carte 2D (t-SNE) des embeddings, colorés par speaker
  3. Clustering → matrice d'affinité + carte 2D avant/après VBx
  4. RAM        → courbe de consommation mémoire au cours du temps
  5. Timing     → bar chart du temps par étape
  6. Rapport    → page HTML récapitulative avec liens vers tout

Usage depuis diarize_improved.py :
    diag = Diagnostics(out_dir, file_id)
    diag.snapshot_ram("vad_start")
    ...
    diag.plot_vad(wav_path, speech_segments)
    diag.plot_embeddings(embeddings, subsegments, labels)
    diag.plot_affinity(embeddings, labels)
    diag.plot_timing(timings)
    diag.plot_ram()
    diag.generate_report(pipeline_name, params)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf


class Diagnostics:
    """Collecte et visualise les diagnostics de chaque étape du pipeline."""

    def __init__(self, out_dir: Path, file_id: str):
        self.out_dir = Path(out_dir) / "diagnostics"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file_id = file_id
        self.ram_snapshots: list[dict] = []
        self.t0 = time.perf_counter()
        self._process = psutil.Process()
        self.snapshot_ram("init")

    # ══════════════════════════════════════════════════════════════════════
    #   RAM tracking
    # ══════════════════════════════════════════════════════════════════════

    def snapshot_ram(self, label: str) -> None:
        """Enregistre la consommation RAM actuelle avec un label."""
        mem = self._process.memory_info()
        self.ram_snapshots.append({
            "label": label,
            "time": time.perf_counter() - self.t0,
            "rss_mb": mem.rss / (1024 * 1024),
            "vms_mb": mem.vms / (1024 * 1024),
        })

    def plot_ram(self) -> Path:
        """Génère la courbe de consommation RAM."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [s["time"] for s in self.ram_snapshots]
        rss = [s["rss_mb"] for s in self.ram_snapshots]
        labels = [s["label"] for s in self.ram_snapshots]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, rss, "b-o", markersize=4, linewidth=1.5)
        ax.fill_between(times, rss, alpha=0.15, color="blue")

        # Annoter les points clés
        for i, (t, r, lab) in enumerate(zip(times, rss, labels)):
            if lab != "ram_poll":  # ne pas annoter les polls intermédiaires
                ax.annotate(lab, (t, r), textcoords="offset points",
                            xytext=(5, 8), fontsize=7, rotation=30)

        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("RAM RSS (MB)")
        ax.set_title(f"Consommation mémoire — {self.file_id}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.out_dir / "ram_usage.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

        # Sauver aussi en JSON
        json_path = self.out_dir / "ram_usage.json"
        json_path.write_text(json.dumps(self.ram_snapshots, indent=2), encoding="utf-8")

        return path

    # ══════════════════════════════════════════════════════════════════════
    #   1. VAD — forme d'onde avec zones de parole
    # ══════════════════════════════════════════════════════════════════════

    def plot_vad(self, wav_path: str, speech_segments: list) -> Path:
        """Forme d'onde audio avec zones de parole colorées en vert."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        times = np.linspace(0, duration, len(audio))

        fig, ax = plt.subplots(figsize=(16, 3))
        ax.plot(times, audio, linewidth=0.3, color="gray", alpha=0.7)

        # Colorier les zones de parole
        for seg in speech_segments:
            ax.axvspan(seg.start, seg.end, alpha=0.25, color="green")

        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"VAD — {len(speech_segments)} segments de parole détectés"
                      f" ({sum(s.duration for s in speech_segments):.1f}s / {duration:.1f}s)")
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        path = self.out_dir / "01_vad.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════════
    #   2. Embeddings — carte 2D t-SNE interactive
    # ══════════════════════════════════════════════════════════════════════

    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        subsegments: list,
        labels: np.ndarray,
        title_suffix: str = "",
    ) -> Path:
        """Carte 2D t-SNE des embeddings, colorés par speaker. Interactif (Plotly)."""
        import plotly.graph_objects as go
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import normalize

        n = len(embeddings)
        if n < 3:
            return self.out_dir / "embeddings_too_few.txt"

        emb = normalize(embeddings, norm="l2")
        perplexity = min(30, n - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     n_iter=1000)
        coords = tsne.fit_transform(emb)

        # Couleurs par speaker
        unique_labels = sorted(set(labels))
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        ]

        fig = go.Figure()
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            hover_texts = []
            for j in np.where(mask)[0]:
                sub = subsegments[j]
                hover_texts.append(
                    f"SPEAKER_{int(lbl):02d}<br>"
                    f"Temps: {sub.start:.2f}s — {sub.end:.2f}s<br>"
                    f"Durée: {sub.end - sub.start:.2f}s<br>"
                    f"Index: {j}"
                )

            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=f"SPEAKER_{int(lbl):02d}",
                marker=dict(size=6, color=colors[i % len(colors)], opacity=0.7),
                text=hover_texts,
                hoverinfo="text",
            ))

        fig.update_layout(
            title=f"Embeddings t-SNE — {n} vecteurs, {len(unique_labels)} speakers {title_suffix}",
            xaxis_title="t-SNE dim 1",
            yaxis_title="t-SNE dim 2",
            width=900, height=600,
            template="plotly_white",
        )

        suffix = title_suffix.replace(" ", "_").replace("(", "").replace(")", "")
        path = self.out_dir / f"02_embeddings{suffix}.html"
        fig.write_html(str(path))

        # Version statique PNG aussi
        self._plot_embeddings_static(coords, labels, subsegments, title_suffix)

        return path

    def _plot_embeddings_static(self, coords, labels, subsegments, title_suffix):
        """Version PNG statique de la carte t-SNE."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        unique_labels = sorted(set(labels))
        fig, ax = plt.subplots(figsize=(10, 7))
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=15, alpha=0.6, label=f"SPEAKER_{int(lbl):02d}")
        ax.legend(fontsize=8)
        ax.set_title(f"Embeddings t-SNE {title_suffix}")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        suffix = title_suffix.replace(" ", "_").replace("(", "").replace(")", "")
        path = self.out_dir / f"02_embeddings{suffix}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    #   3. Matrice d'affinité
    # ══════════════════════════════════════════════════════════════════════

    def plot_affinity(self, embeddings: np.ndarray, labels: np.ndarray) -> Path:
        """Matrice d'affinité triée par speaker."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize

        emb = normalize(embeddings, norm="l2")

        # Trier par label pour visualiser les blocs
        order = np.argsort(labels)
        emb_sorted = emb[order]
        labels_sorted = labels[order]

        affinity = (cosine_similarity(emb_sorted) + 1) / 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Matrice brute
        im1 = axes[0].imshow(affinity, cmap="viridis", aspect="auto",
                              vmin=0, vmax=1)
        axes[0].set_title("Matrice d'affinité (triée par speaker)")
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # Marquer les frontières de speakers
        unique = sorted(set(labels_sorted))
        for lbl in unique:
            indices = np.where(labels_sorted == lbl)[0]
            if len(indices) > 0:
                boundary = indices[-1] + 0.5
                for ax in axes:
                    ax.axhline(y=boundary, color="red", linewidth=0.5, alpha=0.5)
                    ax.axvline(x=boundary, color="red", linewidth=0.5, alpha=0.5)

        # Distribution des similarités inter/intra speaker
        intra_sims = []
        inter_sims = []
        for i in range(len(labels_sorted)):
            for j in range(i + 1, len(labels_sorted)):
                if labels_sorted[i] == labels_sorted[j]:
                    intra_sims.append(affinity[i, j])
                else:
                    inter_sims.append(affinity[i, j])

        if intra_sims and inter_sims:
            axes[1].hist(intra_sims, bins=50, alpha=0.6, label="Intra-speaker",
                         color="green", density=True)
            axes[1].hist(inter_sims, bins=50, alpha=0.6, label="Inter-speaker",
                         color="red", density=True)
            axes[1].legend()
            axes[1].set_title("Distribution des similarités")
            axes[1].set_xlabel("Cosine similarity")
        else:
            axes[1].text(0.5, 0.5, "Pas assez de données",
                         ha="center", va="center")

        plt.tight_layout()
        path = self.out_dir / "03_affinity.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════════
    #   4. Timing — bar chart
    # ══════════════════════════════════════════════════════════════════════

    def plot_timing(self, timings: dict[str, float], total: float,
                    duration: float) -> Path:
        """Bar chart du temps par étape + RTF."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(timings.keys())
        values = list(timings.values())

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(steps, values, color=[
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
            "#59a14f", "#edc948", "#b07aa1",
        ][:len(steps)])

        # Annoter avec le temps
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}s", va="center", fontsize=9)

        ax.set_xlabel("Temps (s)")
        ax.set_title(f"Temps par étape — Total: {total:.1f}s"
                     f" | RTF: {total/duration:.2f}x | Audio: {duration:.1f}s")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        path = self.out_dir / "04_timing.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

        # Sauver en JSON
        json_path = self.out_dir / "timing.json"
        json_path.write_text(json.dumps({
            "steps": timings,
            "total_s": round(total, 2),
            "audio_duration_s": round(duration, 2),
            "rtf": round(total / duration, 3),
        }, indent=2), encoding="utf-8")

        return path

    # ══════════════════════════════════════════════════════════════════════
    #   5. Diarisation timeline
    # ══════════════════════════════════════════════════════════════════════

    def plot_timeline(self, segments: list, duration: float) -> Path:
        """Timeline colorée des segments de diarisation."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        speakers = sorted({seg.speaker for seg in segments})
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        spk_color = {spk: colors[i % len(colors)] for i, spk in enumerate(speakers)}
        spk_y = {spk: i for i, spk in enumerate(speakers)}

        fig, ax = plt.subplots(figsize=(16, max(2, len(speakers) * 0.8 + 1)))

        for seg in segments:
            y = spk_y[seg.speaker]
            ax.barh(y, seg.end - seg.start, left=seg.start, height=0.6,
                    color=spk_color[seg.speaker], alpha=0.7, edgecolor="white",
                    linewidth=0.3)

        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)
        ax.set_xlabel("Temps (s)")
        ax.set_xlim(0, duration)
        ax.set_title(f"Diarisation — {len(speakers)} speakers, {len(segments)} segments")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()
        plt.tight_layout()

        path = self.out_dir / "05_timeline.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        return path

    # ══════════════════════════════════════════════════════════════════════
    #   6. Rapport HTML
    # ══════════════════════════════════════════════════════════════════════

    def generate_report(self, pipeline_name: str, params: dict) -> Path:
        """Génère un rapport HTML récapitulatif avec toutes les visualisations."""
        # Lister les fichiers générés
        images = sorted(self.out_dir.glob("*.png"))
        htmls = sorted(self.out_dir.glob("*.html"))

        params_rows = "\n".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in params.items()
        )

        ram_peak = max((s["rss_mb"] for s in self.ram_snapshots), default=0)

        image_sections = ""
        for img in images:
            name = img.stem.replace("_", " ").title()
            image_sections += f"""
            <div class="section">
                <h2>{name}</h2>
                <img src="{img.name}" style="max-width:100%;">
            </div>
            """

        interactive_sections = ""
        for html_file in htmls:
            if html_file.name == "report.html":
                continue
            name = html_file.stem.replace("_", " ").title()
            interactive_sections += f"""
            <div class="section">
                <h2>{name} (interactif)</h2>
                <p><a href="{html_file.name}" target="_blank">
                   Ouvrir la visualisation interactive →</a></p>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Diagnostics — {self.file_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
               sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;
               background: #f8f9fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db;
              padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .params {{ background: #fff; border-radius: 8px; padding: 15px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .params table {{ border-collapse: collapse; width: 100%; }}
        .params td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
        .params td:first-child {{ width: 200px; color: #7f8c8d; }}
        .section {{ background: #fff; border-radius: 8px; padding: 15px;
                    margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section img {{ border-radius: 4px; }}
        .metric {{ display: inline-block; background: #3498db; color: white;
                   padding: 8px 16px; border-radius: 20px; margin: 5px;
                   font-weight: bold; }}
        a {{ color: #3498db; }}
    </style>
</head>
<body>
    <h1>Diagnostics Pipeline — {self.file_id}</h1>

    <div class="params">
        <h2>Paramètres</h2>
        <span class="metric">Pipeline: {pipeline_name}</span>
        <span class="metric">RAM pic: {ram_peak:.0f} MB</span>
        <table>{params_rows}</table>
    </div>

    {image_sections}
    {interactive_sections}

    <div class="section">
        <h2>Fichiers JSON</h2>
        <ul>
            <li><a href="ram_usage.json">ram_usage.json</a> — snapshots mémoire</li>
            <li><a href="timing.json">timing.json</a> — temps par étape</li>
        </ul>
    </div>
</body>
</html>"""

        path = self.out_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        return path
