import { speakerColor } from "../utils";
import { useStore } from "../store";
import type { RttmSegment } from "../api";

export default function Timeline({
  segments,
  duration,
}: {
  segments: RttmSegment[];
  duration: number;
}) {
  const play = useStore((s) => s.play);
  if (!segments?.length) return null;

  const speakers = [...new Set(segments.map((s) => s.speaker))].sort();

  return (
    <div className="bg-bg-2 rounded-lg p-4 mb-5">
      <h3 className="text-sm text-fg-2 mb-3">Speaker Timeline</h3>
      {speakers.map((spk) => {
        const segs = segments.filter((s) => s.speaker === spk);
        return (
          <div key={spk} className="flex items-center mb-0.5 text-[11px]">
            <div
              className="w-[90px] text-right pr-2"
              style={{ color: speakerColor(spk) }}
            >
              {spk}
            </div>
            <div className="relative flex-1 h-5 bg-bg-3 rounded">
              {segs.map((s, i) => (
                <div
                  key={i}
                  className="absolute h-5 rounded cursor-pointer hover:brightness-125 transition"
                  style={{
                    left: `${(s.start / duration) * 100}%`,
                    width: `${Math.max(((s.end - s.start) / duration) * 100, 0.2)}%`,
                    background: speakerColor(spk),
                    opacity: 0.7,
                  }}
                  title={`${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s`}
                  onClick={() => play(s.start, s.end, s.speaker)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
