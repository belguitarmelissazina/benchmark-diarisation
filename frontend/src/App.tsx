import { useEffect } from "react";
import { useStore } from "./store";
import Sidebar from "./components/Sidebar";
import RunDetail from "./components/RunDetail";
import AudioPlayer from "./components/AudioPlayer";

export default function App() {
  const { fetchRuns, error } = useStore();

  useEffect(() => {
    fetchRuns();
  }, []);

  if (error)
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-xl font-bold mb-2">Cannot connect to backend</h2>
          <p className="text-fg-2 text-sm">
            Start it with:{" "}
            <code className="bg-bg-3 px-2 py-1 rounded">
              python -m uvicorn backend.app:app --port 8765
            </code>
          </p>
          <p className="text-red-400 text-xs mt-2">{error}</p>
        </div>
      </div>
    );

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 overflow-y-auto p-6">
        <RunDetail />
      </div>
      <AudioPlayer />
    </div>
  );
}
