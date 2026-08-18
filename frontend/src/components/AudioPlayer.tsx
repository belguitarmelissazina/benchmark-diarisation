import { useEffect, useRef, useMemo } from "react";
import { audioUrl } from "../api";
import { useStore } from "../store";

export default function AudioPlayer() {
  const { activeRunId, playRequest } = useStore();
  const audioRef = useRef<HTMLAudioElement>(null);

  const src = useMemo(() => {
    if (!activeRunId) return "";
    if (playRequest) return audioUrl(activeRunId, playRequest.start, playRequest.end);
    return audioUrl(activeRunId);
  }, [activeRunId, playRequest]);

  useEffect(() => {
    if (playRequest && audioRef.current) {
      audioRef.current.load();
      audioRef.current.play().catch(() => {});
    }
  }, [playRequest]);

  if (!activeRunId) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 h-12 bg-bg-2 border-t border-gray-700
                    flex items-center px-5 gap-3 z-50">
      <div className="text-[11px] text-fg-2 min-w-[200px]">
        {playRequest
          ? `${playRequest.speaker || ""} ${playRequest.start.toFixed(1)}s - ${playRequest.end.toFixed(1)}s`
          : "Click any segment to play"}
      </div>
      <audio ref={audioRef} controls src={src} className="h-8 flex-1" />
    </div>
  );
}
