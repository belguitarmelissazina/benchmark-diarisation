import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, type SimData } from "../api";
import { PLOT_LAYOUT_BASE } from "../utils";

export default function SimView({ runId }: { runId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [variant, setVariant] = useState<"raw" | "enhanced">("raw");
  const [data, setData] = useState<SimData | null>(null);

  useEffect(() => {
    api<SimData>(`/runs/${runId}/sim?variant=${variant}&max_size=256`)
      .then(setData)
      .catch(() => {});
  }, [runId, variant]);

  useEffect(() => {
    if (!data || !ref.current) return;
    Plotly.newPlot(
      ref.current,
      [{ z: data.values, type: "heatmap", colorscale: "Viridis", showscale: true }],
      {
        ...PLOT_LAYOUT_BASE,
        margin: { t: 20, b: 40, l: 40, r: 20 },
        height: 450,
        title: data.stride > 1
          ? `(downsampled ${data.stride}x from ${data.original_shape[0]})`
          : "",
      },
      { responsive: true }
    );
  }, [data]);

  return (
    <div className="bg-bg-2 rounded-lg p-4 mb-5">
      <h3 className="text-sm text-fg-2 mb-3">Similarity Matrix</h3>
      <div className="flex gap-2 mb-2">
        {(["raw", "enhanced"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setVariant(v)}
            className={`px-3 py-1 rounded text-xs border transition
              ${variant === v
                ? "bg-accent text-black border-accent"
                : "bg-bg-3 text-fg border-gray-700 hover:border-fg-2"}`}
          >
            {v.charAt(0).toUpperCase() + v.slice(1)}
          </button>
        ))}
      </div>
      <div ref={ref} />
    </div>
  );
}
