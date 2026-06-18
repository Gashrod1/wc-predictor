import type {
  TeamInfo,
  PredictResponse,
  TeamDetail,
  H2HResponse,
  FixtureItem,
  BacktestMetrics,
} from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json() as Promise<T>;
}

export function fetchTeams(): Promise<TeamInfo[]> {
  return getJSON<TeamInfo[]>("/api/teams");
}

export async function predict(
  home: string,
  away: string,
  stage: string,
): Promise<PredictResponse> {
  const resp = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ home, away, stage }),
  });
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json() as Promise<PredictResponse>;
}

export function fetchTeam(name: string): Promise<TeamDetail> {
  return getJSON<TeamDetail>(`/api/team/${encodeURIComponent(name)}`);
}

export function fetchH2H(home: string, away: string): Promise<H2HResponse> {
  return getJSON<H2HResponse>(
    `/api/h2h?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`,
  );
}

export function fetchFixtures(): Promise<FixtureItem[]> {
  return getJSON<FixtureItem[]>("/api/fixtures");
}

export function fetchBacktest(tournament: string): Promise<BacktestMetrics> {
  return getJSON<BacktestMetrics>(`/api/backtest/${tournament}`);
}
