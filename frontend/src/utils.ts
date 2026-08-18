export const SPEAKER_COLORS = [
  "#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff",
  "#f778ba", "#79c0ff", "#7ee787", "#ffa657", "#ff7b72",
];

export function speakerColor(spk: string): string {
  const idx = parseInt(spk.replace(/\D/g, "")) || 0;
  return SPEAKER_COLORS[idx % SPEAKER_COLORS.length];
}

export const PLOT_LAYOUT_BASE = {
  paper_bgcolor: "#161b22",
  plot_bgcolor: "#1c2128",
  font: { color: "#e1e4e8", size: 11 },
};
