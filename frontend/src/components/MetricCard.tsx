export default function MetricCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-bg-2 rounded-lg p-3.5">
      <div className="text-[11px] text-fg-2 uppercase">{label}</div>
      <div className="text-2xl font-bold mt-1" style={{ color: color || "var(--fg)" }}>
        {value}
      </div>
    </div>
  );
}
