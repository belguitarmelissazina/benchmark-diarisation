import { create } from "zustand";
import { api, type RunSummary } from "./api";

interface PlayRequest {
  start: number;
  end: number;
  speaker?: string;
  _t: number; // force re-trigger
}

interface AppState {
  runs: RunSummary[];
  activeRunId: string | null;
  activeRun: any | null; // full run detail
  error: string | null;
  playRequest: PlayRequest | null;

  fetchRuns: () => Promise<void>;
  selectRun: (id: string) => Promise<void>;
  play: (start: number, end: number, speaker?: string) => void;
}

export const useStore = create<AppState>((set, get) => ({
  runs: [],
  activeRunId: null,
  activeRun: null,
  error: null,
  playRequest: null,

  fetchRuns: async () => {
    try {
      const runs = await api<RunSummary[]>("/runs?experiment=&limit=100");
      set({ runs, error: null });
      if (runs.length && !get().activeRunId) {
        get().selectRun(runs[0].run_id);
      }
    } catch (e: any) {
      set({ error: e.message });
    }
  },

  selectRun: async (id: string) => {
    set({ activeRunId: id, activeRun: null });
    try {
      const run = await api(`/runs/${id}`);
      set({ activeRun: run });
    } catch {}
  },

  play: (start, end, speaker) => {
    set({ playRequest: { start, end, speaker, _t: Date.now() } });
  },
}));
