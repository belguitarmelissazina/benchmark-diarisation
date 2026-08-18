import { useStore } from "../store";
import { speakerColor } from "../utils";

export default function Sidebar() {
  const { runs, activeRunId, selectRun } = useStore();

  return (
    <div className="w-80 min-w-[320px] bg-bg-2 border-r border-gray-700 overflow-y-auto p-4">
      <h2 className="text-xs text-fg-2 uppercase tracking-wider mb-3">
        Experiments ({runs.length})
      </h2>
      {runs.map((r) => (
        <div
          key={r.run_id}
          className={`p-2.5 rounded-lg cursor-pointer mb-2 border transition
            ${r.run_id === activeRunId
              ? "border-accent bg-bg-3"
              : "border-transparent hover:bg-bg-3"}`}
          onClick={() => selectRun(r.run_id)}
        >
          <div className="text-sm font-semibold truncate">
            {r.tags.file_id || r.run_name}
          </div>
          <div className="flex gap-1 mt-1 flex-wrap">
            <Badge color="blue">{r.params.embed}</Badge>
            <Badge color="green">{r.params.cluster}</Badge>
            <Badge color="purple">{r.params.estimate}</Badge>
            {r.params.refine_vbx === "True" && <Badge color="purple">VBx</Badge>}
            {r.params.transcribe === "True" && <Badge color="green">ASR</Badge>}
          </div>
          <div className="text-[11px] text-fg-2 mt-1">
            {r.metrics.num_speakers_final?.toFixed(0)} spk &middot;{" "}
            {r.metrics.time_total?.toFixed(0)}s &middot; RTF{" "}
            {r.metrics.rtf?.toFixed(2)}x
            {r.metrics.der != null && (
              <> &middot; DER {(r.metrics.der * 100).toFixed(1)}%</>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function Badge({ children, color }: { children: React.ReactNode; color: string }) {
  const cls: Record<string, string> = {
    blue: "bg-blue-900/30 text-accent",
    green: "bg-green-900/30 text-green-400",
    purple: "bg-purple-900/30 text-purple-400",
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${cls[color] || ""}`}>
      {children}
    </span>
  );
}
