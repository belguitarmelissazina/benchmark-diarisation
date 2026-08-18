const BASE = "/api";

export async function api<T = any>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export function audioUrl(runId: string, start?: number, end?: number): string {
  if (start != null && end != null)
    return `${BASE}/runs/${runId}/audio?start=${start}&end=${end}`;
  return `${BASE}/runs/${runId}/audio`;
}

export interface RunSummary {
  run_id: string;
  run_name: string;
  status: string;
  tags: Record<string, string>;
  params: Record<string, string>;
  metrics: Record<string, number>;
}

export interface UmapData {
  points: number[][];
  labels: number[] | null;
  chunks: { idx: number; start: number; end: number; parent_idx: number }[] | null;
}

export interface SimData {
  variant: string;
  original_shape: number[];
  shape: number[];
  stride: number;
  values: number[][];
}

export interface RttmSegment {
  start: number;
  duration: number;
  end: number;
  speaker: string;
}

export interface Turn {
  start: number;
  end: number;
  speaker: string;
  text: string;
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  speaker: string;
}

export interface TranscriptMode {
  words: TranscriptWord[] | null;
  turns: Turn[] | null;
}

export interface TranscriptData {
  modes?: Record<string, TranscriptMode>;
  words: TranscriptWord[] | null;
  turns: Turn[] | null;
  words_per_speaker: Record<string, number> | null;
}

export interface RamData {
  samples: { t: number; rss_mb: number }[];
  marks: { t: number; label: string }[];
  peak_mb: number;
  mean_mb: number;
}

export interface WaveformData {
  duration: number;
  times: number[];
  min: number[];
  max: number[];
}

export interface VadData {
  audio_duration: number;
  segments: { start: number; end: number }[];
}
