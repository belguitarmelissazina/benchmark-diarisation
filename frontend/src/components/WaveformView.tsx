import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { api, type WaveformData, type VadData } from "../api";
import { PLOT_LAYOUT_BASE } from "../utils";

export default function WaveformView({ runId }: { runId: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [wf, setWf] = useState<WaveformData | null>(null);
  const [vad, setVad] = useState<VadData | null>(null);

  useEffect(() => {
    api<WaveformData>(`/runs/${runId}/waveform?max_points=2000`).then(setWf).catch(() => {});
    api<VadData>(`/runs/${runId}/vad`).then(setVad).catch(() => {});
  }, [runId]);

  useEffect(() => {
    if (!wf || !ref.current) return;

    const traces: Plotly.Data[] = [
      {
        x: wf.times, y: wf.max, type: "scatter", mode: "lines",
        line: { color: "#58a6ff", width: 0.5 }, name: "waveform",
      },
      {
        x: wf.times, y: wf.min, type: "scatter", mode: "lines",
        line: { color: "#58a6ff", width: 0.5 }, fill: "tonexty",
        fillcolor: "rgba(88,166,255,0.15)", showlegend: false,
      },
    ];

    const shapes = (vad?.segments || []).map((s) => ({
      type: "rect" as const, xref: "x" as const, yref: "paper" as const,
      x0: s.start, x1: s.end, y0: 0, y1: 1,
      fillcolor: "rgba(63,185,80,0.12)", line: { width: 0 },
    }));

    Plotly.newPlot(ref.current, traces, {
      ...PLOT_LAYOUT_BASE, shapes,
      margin: { t: 20, b: 40, l: 40, r: 20 },
      xaxis: { title: "Time (s)", gridcolor: "#30363d" },
      yaxis: { gridcolor: "#30363d", zeroline: false },
      showlegend: false, height: 200,
    }, { responsive: true });
  }, [wf, vad]);

  return (
    <div className="bg-bg-2 rounded-lg p-4 mb-5">
      <h3 className="text-sm text-fg-2 mb-3">Waveform + VAD</h3>
      <div ref={ref} />
    </div>
  );
}
