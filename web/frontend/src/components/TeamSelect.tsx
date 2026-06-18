import type { TeamInfo } from "../types";

interface Props {
  label: string;
  teams: TeamInfo[];
  value: string;
  onChange: (value: string) => void;
}

export default function TeamSelect({ label, teams, value, onChange }: Props) {
  return (
    <div>
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">— choisir —</option>
        {teams.map((t) => (
          <option key={t.name} value={t.name}>
            {t.name} ({Math.round(t.elo)})
          </option>
        ))}
      </select>
    </div>
  );
}
