import { useEffect, useState } from "react";
import { api, type TranscriptData } from "../api";
import { speakerColor } from "../utils";
import { useStore } from "../store";

export default function TranscriptView({ runId }: { runId: string }) {
  const [data, setData] = useState<TranscriptData | null>(null);
  const [err, setErr] = useState(false);
  const [mode, setMode] = useState<string>("midpoint");
  const play = useStore((s) => s.play);

  useEffect(() => {
    setErr(false);
    api<TranscriptData>(`/runs/${runId}/transcript`)
      .then((d) => {
        setData(d);
        const available = Object.keys(d.modes || {});
        if (available.length && !available.includes(mode)) setMode(available[0]);
      })
      .catch(() => setErr(true));
  }, [runId]);

  if (err)
    return (
      <div className="text-center p-16 text-fg-2">
        No transcript for this run (use <code>--transcribe</code>)
      </div>
    );
  if (!data) return <div className="text-center p-10 text-fg-2">Loading transcript...</div>;

  const modes = data.modes || {};
  const availableModes = Object.keys(modes);
  const active = modes[mode] || { words: data.words, turns: data.turns };
  const turns = active.turns || [];
  const words = active.words || [];

  const MODE_DESCRIPTIONS: Record<string, string> = {
    midpoint: "Each word is assigned the speaker active at its midpoint (per-word diarization alignment).",
    boundary: "Transcribe-first: speaker changes are snapped to the nearest punctuation (. ! ?) within ±5s of each diarization speaker change.",
  };

  return (
    <div className="bg-bg-2 rounded-lg p-4 mb-5">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <h3 className="text-sm text-fg-2">
          Speaker-Attributed Transcript ({turns.length} turns, {words.length} words)
        </h3>
        {availableModes.length > 1 && (
          <div className="flex gap-1 bg-bg-3 rounded p-1">
            {availableModes.map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1 text-xs rounded transition ${
                  mode === m ? "bg-accent text-white" : "text-fg-2 hover:text-fg"
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        )}
      </div>
      {MODE_DESCRIPTIONS[mode] && (
        <div className="text-[11px] text-fg-2 mb-3 italic">{MODE_DESCRIPTIONS[mode]}</div>
      )}

      {data.words_per_speaker && (
        <div className="mb-3 text-xs text-fg-2 flex gap-4 flex-wrap">
          {Object.entries(data.words_per_speaker)
            .sort()
            .map(([spk, n]) => (
              <span key={spk}>
                <span className="font-bold" style={{ color: speakerColor(spk) }}>
                  {spk}
                </span>
                : {n} words
              </span>
            ))}
        </div>
      )}

      <div className="max-h-[500px] overflow-y-auto space-y-1">
        {turns.map((t, i) => (
          <div
            key={i}
            className="p-2 pl-3 rounded-r-md bg-bg-3 border-l-[3px] cursor-pointer
                       hover:brightness-110 transition text-sm"
            style={{ borderLeftColor: speakerColor(t.speaker) }}
            onClick={() => play(t.start, t.end, t.speaker)}
          >
            <span className="font-bold text-[11px]" style={{ color: speakerColor(t.speaker) }}>
              {t.speaker}
            </span>
            <span className="text-[10px] text-fg-2 ml-2">
              [{t.start.toFixed(1)}s - {t.end.toFixed(1)}s]
            </span>
            <div className="mt-1">{t.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
