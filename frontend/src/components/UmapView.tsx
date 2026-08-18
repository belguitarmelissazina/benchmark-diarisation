import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, type UmapData } from "../api";
import { speakerColor, PLOT_LAYOUT_BASE } from "../utils";
import { useStore } from "../store";

export default function UmapView({ runId }: { runId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<UmapData | null>(null);
  const play = useStore((s) => s.play);

  useEffect(() => {
    api<UmapData>(`/runs/${runId}/umap`).then(setData).catch(() => setData(null));
  }, [runId]);

  useEffect(() => {
    if (!data?.points || !ref.current) return;
    const labels = data.labels || data.points.map(() => 0);
    const speakers = [...new Set(labels)].sort((a, b) => a - b);

    const traces = speakers.map((spk) => {
      const idxs = labels
        .map((l, i) => [l, i] as const)
        .filter(([l]) => l === spk)
        .map(([, i]) => i);
      return {
        x: idxs.map((i) => data.points[i][0]),
        y: idxs.map((i) => data.points[i][1]),
        mode: "markers" as const,
        name: `SPEAKER_${String(spk).padStart(2, "0")}`,
        marker: { color: speakerColor(`SPEAKER_${spk}`), size: 5, opacity: 0.7 },
        text: idxs.map((i) => {
          const c = data.chunks?.[i];
          return c ? `${c.start.toFixed(1)}s - ${c.end.toFixed(1)}s` : `#${i}`;
        }),
        hoverinfo: "text+name" as const,
        customdata: idxs.map((i) => data.chunks?.[i] || null),
        type: "scatter" as const,
      };
    });

    Plotly.newPlot(
      ref.current,
      traces,
      {
        ...PLOT_LAYOUT_BASE,
        margin: { t: 30, b: 40, l: 40, r: 20 },
        xaxis: { title: "UMAP 1", gridcolor: "#30363d" },
        yaxis: { title: "UMAP 2", gridcolor: "#30363d" },
        legend: { orientation: "h" as const, y: -0.15 },
        height: 450,
      },
      { responsive: true }
    );

    ref.current.on("plotly_click", (ev: any) => {
      const pt = ev.points?.[0];
      if (pt?.customdata) play(pt.customdata.start, pt.customdata.end);
    });
  }, [data]);

  if (!data) return <div className="text-center p-10 text-fg-2">Loading UMAP...</div>;

  return (
    <div className="bg-bg-2 rounded-lg p-4 mb-5">
      <h3 className="text-sm text-fg-2 mb-3">UMAP Embeddings</h3>
      <div ref={ref} />
    </div>
  );
}
