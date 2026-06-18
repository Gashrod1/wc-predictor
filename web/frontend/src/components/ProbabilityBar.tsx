interface Props {
  label: string;
  value: number; // 0..1
  color: string;
}

export default function ProbabilityBar({ label, value, color }: Props) {
  const pct = Math.round(value * 1000) / 10;
  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-label">
        <span>{label}</span>
        <span>{pct.toFixed(1)}%</span>
      </div>
      <div className="prob-bar-track">
        <div
          className="prob-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}
