import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, type RamData } from "../api";
import { PLOT_LAYOUT_BASE, SPEAKER_COLORS } from "../utils";

export default function MetricsView({
  runId,
  metrics,
}: {
  runId: string;
  metrics: Record<string, number>;
}) {
  const ramRef = useRef<HTMLDivElement>(null);
  const timingRef = useRef<HTMLDivElement>(null);
  const [ram, setRam] = useState<RamData | null>(null);

  useEffect(() => {
    api<RamData>(`/runs/${runId}/ram`).then(setRam).catch(() => {});
  }, [runId]);

  // RAM chart
  useEffect(() => {
    if (!ram?.samples?.length || !ramRef.current) return;
    const t = ram.samples.map((s) => s.t);
    const m = ram.samples.map((s) => s.rss_mb);

    const shapes = (ram.marks || []).map((mk) => ({
      type: "line" as const, x0: mk.t, x1: mk.t, y0: 0, y1: 1,
      yref: "paper" as const,
      line: { color: "#d29922", width: 1, dash: "dash" as const },
    }));
    const annotations = (ram.marks || []).map((mk) => ({
      x: mk.t, y: 1, yref: "paper" as const, text: mk.label,
      showarrow: false, font: { size: 9, color: "#d29922" },
    }));

    Plotly.newPlot(ramRef.current, [{
      x: t, y: m, type: "scatter", mode: "lines",
      line: { color: "#f85149", width: 1.5 }, fill: "tozeroy",
      fillcolor: "rgba(248,81,73,0.1)", name: "RSS",
    }], {
      ...PLOT_LAYOUT_BASE, shapes, annotations,
      margin: { t: 20, b: 40, l: 60, r: 20 },
      xaxis: { title: "Time (s)", gridcolor: "#30363d" },
      yaxis: { title: "RAM (MB)", gridcolor: "#30363d" },
      showlegend: false, height: 250,
    }, { responsive: true });
  }, [ram]);

  // Timing bar chart
  useEffect(() => {
    if (!metrics || !timingRef.current) return;
    const keys = Object.keys(metrics)
      .filter((k) => k.startsWith("time_") && k !== "time_total")
      .sort();
    const vals = keys.map((k) => metrics[k]);
    const labels = keys.map((k) => k.replace("time_", ""));

    Plotly.newPlot(timingRef.current, [{
      x: labels, y: vals, type: "bar",
      marker: { color: SPEAKER_COLORS.slice(0, labels.length) },
    }], {
      ...PLOT_LAYOUT_BASE,
      margin: { t: 20, b: 60, l: 60, r: 20 },
      xaxis: { gridcolor: "#30363d" },
      yaxis: { title: "Seconds", gridcolor: "#30363d" },
      showlegend: false, height: 250,
    }, { responsive: true });
  }, [metrics]);

  return (
    <>
      <div className="bg-bg-2 rounded-lg p-4 mb-5">
        <h3 className="text-sm text-fg-2 mb-3">RAM Usage Over Time</h3>
        <div ref={ramRef} />
      </div>
      <div className="bg-bg-2 rounded-lg p-4 mb-5">
        <h3 className="text-sm text-fg-2 mb-3">Time Per Step</h3>
        <div ref={timingRef} />
      </div>
    </>
  );
}
