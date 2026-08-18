import { useEffect, useState } from "react";
import { api, type RttmSegment } from "../api";
import { useStore } from "../store";
import MetricCard from "./MetricCard";
import Timeline from "./Timeline";
import WaveformView from "./WaveformView";
import UmapView from "./UmapView";
import SimView from "./SimView";
import MetricsView from "./MetricsView";
import TranscriptView from "./TranscriptView";

const TABS = ["timeline", "waveform", "umap", "similarity", "metrics", "transcript"] as const;
type Tab = (typeof TABS)[number];

export default function RunDetail() {
  const { activeRunId, activeRun } = useStore();
  const [tab, setTab] = useState<Tab>("timeline");
  const [rttm, setRttm] = useState<{ segments: RttmSegment[] } | null>(null);

  useEffect(() => {
    if (!activeRunId) return;
    api<{ segments: RttmSegment[] }>(`/runs/${activeRunId}/rttm`)
      .then(setRttm)
      .catch(() => setRttm(null));
    setTab("timeline");
  }, [activeRunId]);

  if (!activeRunId)
    return <div className="text-center p-16 text-fg-2">Select a run from the sidebar</div>;
  if (!activeRun)
    return <div className="text-center p-10 text-fg-2">Loading run...</div>;

  const m = activeRun.metrics as Record<string, number>;
  const p = activeRun.params as Record<string, string>;
  const dur = m.audio_duration_sec || 0;

  return (
    <div className="pb-16">
      {/* Metrics grid */}
      <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-3 mb-5">
        <MetricCard label="Speakers" value={m.num_speakers_final?.toFixed(0) || "?"} color="#58a6ff" />
        <MetricCard label="Segments" value={m.num_segments?.toFixed(0) || "?"} />
        <MetricCard label="Duration" value={dur > 0 ? `${(dur / 60).toFixed(1)} min` : "?"} />
        <MetricCard label="Total time" value={m.time_total ? `${m.time_total.toFixed(1)}s` : "?"} />
        <MetricCard label="RTF" value={m.rtf ? `${m.rtf.toFixed(2)}x` : "?"} />
        <MetricCard label="Peak RAM" value={m.peak_ram_mb ? `${m.peak_ram_mb.toFixed(0)} MB` : "?"} />
        <MetricCard label="Speech ratio" value={m.speech_ratio ? `${(m.speech_ratio * 100).toFixed(1)}%` : "?"} />
        {m.der != null && (
          <MetricCard label="DER" value={`${(m.der * 100).toFixed(1)}%`} color="#d29922" />
        )}
        {m.num_words != null && (
          <MetricCard label="Words" value={m.num_words.toFixed(0)} color="#3fb950" />
        )}
      </div>

      {/* Params row */}
      <div className="mb-4 text-xs text-fg-2">
        embed=<b className="text-fg">{p.embed}</b> | estimate=
        <b className="text-fg">{p.estimate}</b> | cluster=
        <b className="text-fg">{p.cluster}</b> | enhance=
        <b className="text-fg">{p.enhance}</b> | vbx=
        <b className="text-fg">{p.refine_vbx}</b> | window=
        <b className="text-fg">{p.win_len}s/{p.hop_len}s</b>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-700 mb-5">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition
              ${tab === t
                ? "text-accent border-accent"
                : "text-fg-2 border-transparent hover:text-fg"}`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === "timeline" && rttm && (
        <Timeline segments={rttm.segments} duration={dur} />
      )}
      {tab === "waveform" && <WaveformView runId={activeRunId} />}
      {tab === "umap" && <UmapView runId={activeRunId} />}
      {tab === "similarity" && <SimView runId={activeRunId} />}
      {tab === "metrics" && <MetricsView runId={activeRunId} metrics={m} />}
      {tab === "transcript" && <TranscriptView runId={activeRunId} />}
    </div>
  );
}
