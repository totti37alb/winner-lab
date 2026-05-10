"use client";

import { useState, useEffect, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Team {
  チーム名: string;
  カテゴリ: string;
  攻撃力: number;
  守備力: number;
  選手層: number;
  監督力: number;
  完成度: number;
}

interface ScoreProb {
  score: string;
  prob: number;
}

interface MatchResult {
  h: string;
  hScore: number;
  aScore: number;
  a: string;
}

// ─── Poisson ──────────────────────────────────────────────────────────────────

function poissonPMF(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  let logP = -lambda + k * Math.log(lambda);
  for (let i = 2; i <= k; i++) logP -= Math.log(i);
  return Math.exp(logP);
}

// ─── Prediction Logic ─────────────────────────────────────────────────────────

function predictScore(h: Team, a: Team): ScoreProb[] {
  const muH =
    (h.攻撃力 / (a.守備力 + 1)) * (1 + (h.監督力 + h.完成度) * 0.005) + 0.3;
  const muA =
    (a.攻撃力 / (h.守備力 + 1)) * (1 + (a.監督力 + a.完成度) * 0.005) + 0.3;

  const probs: ScoreProb[] = [];
  for (let hi = 0; hi < 6; hi++) {
    for (let ai = 0; ai < 6; ai++) {
      probs.push({
        score: `${hi}-${ai}`,
        prob: poissonPMF(hi, muH) * poissonPMF(ai, muA),
      });
    }
  }
  return probs.sort((a, b) => b.prob - a.prob);
}

function pickAnakuji(probs: ScoreProb[]): ScoreProb {
  const candidates = probs.filter((p) => {
    if (p.prob >= 0.05 || !p.score.includes("-")) return false;
    const [h, a] = p.score.split("-").map(Number);
    return Math.abs(h - a) >= 2;
  });
  if (candidates.length > 0) return candidates[0];
  const fallback = probs.filter((p) => p.prob < 0.05 && p.score.includes("-"));
  return fallback[0] ?? probs[3];
}

// ─── CSV Parser ───────────────────────────────────────────────────────────────

function parseCSV(text: string): Team[] {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim());
    const obj: Record<string, string | number> = {};
    headers.forEach((h, i) => {
      const n = Number(vals[i]);
      obj[h] = isNaN(n) ? vals[i] : n;
    });
    return obj as unknown as Team;
  });
}

function teamsToCSV(teams: Team[]): string {
  const header = "チーム名,カテゴリ,攻撃力,守備力,選手層,監督力,完成度";
  const rows = teams.map(
    (t) =>
      `${t.チーム名},${t.カテゴリ},${t.攻撃力},${t.守備力},${t.選手層},${t.監督力},${t.完成度}`
  );
  return [header, ...rows].join("\n");
}

// ─── Match Text Parser ────────────────────────────────────────────────────────

// Normalize full-width ASCII (Ａ→A, Ｆ→F, etc.) to half-width
function normalizeFullWidth(str: string): string {
  return str.replace(/[！-～]/g, (ch) =>
    String.fromCharCode(ch.charCodeAt(0) - 0xfee0)
  );
}

function parseMatches(text: string, teams: Team[]): [string, string][] {
  const processed = normalizeFullWidth(text)
    .replace(/　/g, " ")
    .replace(/ＶＳ/g, " ")
    .replace(/vs/gi, " ")
    .replace(/\t/g, " ");

  const teamNames = [...teams.map((t) => t.チーム名)].sort(
    (a, b) => b.length - a.length
  );
  const results: [string, string][] = [];

  for (const line of processed.split("\n")) {
    if (!line.trim()) continue;
    const hits: [number, string][] = [];
    for (const name of teamNames) {
      let start = 0;
      while (true) {
        const pos = line.indexOf(name, start);
        if (pos === -1) break;
        hits.push([pos, name]);
        start = pos + name.length;
      }
    }
    hits.sort((a, b) => a[0] - b[0]);
    const finalHits: string[] = [];
    let lastEnd = -1;
    for (const [pos, name] of hits) {
      if (pos >= lastEnd) {
        finalHits.push(name);
        lastEnd = pos + name.length;
      }
    }
    if (finalHits.length >= 2 && finalHits[0] !== finalHits[1]) {
      results.push([finalHits[0], finalHits[1]]);
    }
  }

  const seen = new Set<string>();
  return results.filter(([h, a]) => {
    const key = `${h}|${a}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// Parse raw schedule/result paste into scored match list.
// Handles format: "[home_team] [score]\n試合終了\n([PK line])\n[score] [away_team]"
function parseResults(text: string, teams: Team[]): MatchResult[] {
  const normalized = normalizeFullWidth(text)
    .replace(/　/g, " ")
    .replace(/\t/g, " ");
  const lines = normalized.split("\n").map((l) => l.trim()).filter((l) => l);

  const teamNames = [...teams.map((t) => t.チーム名)].sort(
    (a, b) => b.length - a.length
  );

  function findTeamIn(str: string): string | null {
    for (const name of teamNames) {
      if (str.includes(name)) return name;
    }
    return null;
  }

  const results: MatchResult[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < lines.length; i++) {
    const homeTeam = findTeamIn(lines[i]);
    if (!homeTeam) continue;

    // Score must appear at the end of the line (after team name)
    const homeScoreMatch = lines[i].match(/(\d+)\s*$/);
    if (!homeScoreMatch) continue;
    const hScore = Number(homeScoreMatch[1]);

    // 試合終了 must appear within the next 4 lines
    let endIdx = -1;
    for (let j = i + 1; j <= Math.min(i + 4, lines.length - 1); j++) {
      if (lines[j].includes("試合終了")) { endIdx = j; break; }
    }
    if (endIdx === -1) continue;

    // After 試合終了, find away score + team (skip PK lines starting with "(")
    for (let j = endIdx + 1; j <= Math.min(endIdx + 3, lines.length - 1); j++) {
      if (lines[j].startsWith("(")) continue;
      const awayScoreMatch = lines[j].match(/^(\d+)/);
      if (!awayScoreMatch) continue;
      const awayTeam = findTeamIn(lines[j]);
      if (!awayTeam || awayTeam === homeTeam) continue;

      const key = `${homeTeam}|${awayTeam}`;
      if (!seen.has(key)) {
        seen.add(key);
        results.push({ h: homeTeam, hScore: hScore, aScore: Number(awayScoreMatch[1]), a: awayTeam });
      }
      i = j;
      break;
    }
  }

  return results;
}

// ─── Team Colors ──────────────────────────────────────────────────────────────

const TEAM_COLORS: Record<string, [string, string]> = {
  札幌: ["#E50012", "#000000"],
  鹿島: ["#B11021", "#000000"],
  浦和: ["#E60012", "#000000"],
  柏: ["#FFF100", "#000000"],
  FC東京: ["#0000FF", "#FF0000"],
  東京V: ["#006400", "#DAA520"],
  町田: ["#000080", "#FFD700"],
  川崎F: ["#0099D9", "#000000"],
  横浜FM: ["#0000FF", "#FF0000"],
  湘南: ["#77FF00", "#0000FF"],
  新潟: ["#FF6700", "#0047AB"],
  磐田: ["#92B5D2", "#000000"],
  名古屋: ["#D51621", "#FFAD00"],
  京都: ["#8C0733", "#000000"],
  G大阪: ["#0000FF", "#000000"],
  C大阪: ["#E3007F", "#000080"],
  神戸: ["#86001E", "#000000"],
  広島: ["#502C83", "#000000"],
  福岡: ["#002E5D", "#A4B1D7"],
  鳥栖: ["#17E0FD", "#FF1493"],
  仙台: ["#FFD700", "#0000FF"],
  秋田: ["#0055FF", "#FFD700"],
  山形: ["#0000FF", "#FFFFFF"],
  いわき: ["#FF4500", "#000080"],
  水戸: ["#0000FF", "#FF0000"],
  栃木: ["#FFFF00", "#0000FF"],
  栃木C: ["#FFFF00", "#0000FF"],
  栃木SC: ["#FFFF00", "#0000FF"],
  群馬: ["#003366", "#FFFF00"],
  千葉: ["#FFF100", "#009944"],
  横浜FC: ["#00AEEF", "#000000"],
  甲府: ["#0000FF", "#FF0000"],
  清水: ["#FF8C00", "#000080"],
  藤枝: ["#800080", "#FFFFFF"],
  岡山: ["#8B0000", "#000000"],
  山口: ["#FF4500", "#000000"],
  徳島: ["#000080", "#FFFFFF"],
  愛媛: ["#FF8C00", "#006400"],
  長崎: ["#0055FF", "#FF8C00"],
  熊本: ["#FF0000", "#000000"],
  大分: ["#0000FF", "#FFFF00"],
  鹿児島: ["#000080", "#FFFFFF"],
  八戸: ["#008000", "#FFFFFF"],
  岩手: ["#FFFFFF", "#000000"],
  福島: ["#FF0000", "#FFFF00"],
  大宮: ["#FF6600", "#000080"],
  松本: ["#006400", "#FFFFFF"],
  長野: ["#FF8C00", "#000080"],
  富山: ["#0000FF", "#FF0000"],
  金沢: ["#FF0000", "#FFFF00"],
  沼津: ["#0000FF", "#FFFFFF"],
  岐阜: ["#006400", "#FFFFFF"],
  奈良: ["#000080", "#FFFFFF"],
  FC大阪: ["#ADD8E6", "#000080"],
  讃岐: ["#ADD8E6", "#000080"],
  今治: ["#007BFF", "#FFF100"],
  北九州: ["#FFFF00", "#FF0000"],
  宮崎: ["#FFFFFF", "#000000"],
  琉球: ["#8B0000", "#DAA520"],
  鳥取: ["#00FF00", "#FFFFFF"],
  滋賀: ["#0000FF", "#FFFFFF"],
  相模原: ["#006400", "#000000"],
  高知: ["#FF6700", "#000000"],
};

// Colors that require black text
const LIGHT_BG_COLORS = new Set([
  "#FFF100",
  "#FFFF33",
  "#FFFFFF",
  "#77FF00",
  "#ADD8E6",
  "#00FF00",
  "#FFD700",
  "#FFFF00",
  "#FF8C00",
]);

const TABS = [
  { id: "scan", label: "期待値", icon: "🎯" },
  { id: "learn", label: "学習", icon: "🧠" },
  { id: "power", label: "戦力表", icon: "📊" },
] as const;

type TabId = (typeof TABS)[number]["id"];

const CATEGORIES = [
  { id: "J1-EAST", label: "J1E" },
  { id: "J1-WEST", label: "J1W" },
  { id: "J23-EAST-A", label: "EA" },
  { id: "J23-EAST-B", label: "EB" },
  { id: "J23-WEST-A", label: "WA" },
  { id: "J23-WEST-B", label: "WB" },
];

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [activeTab, setActiveTab] = useState<TabId>("scan");

  // Tab 1
  const [scanText, setScanText] = useState("");
  const [scanResults, setScanResults] = useState<
    { h: string; a: string; probs: ScoreProb[] }[]
  >([]);

  // Tab 2
  const [learnRaw, setLearnRaw] = useState("");
  const [learnParsed, setLearnParsed] = useState<MatchResult[]>([]);
  const [learnRate, setLearnRate] = useState(0.05);
  const [learnMsg, setLearnMsg] = useState("");

  // Tab 3
  const [activeCat, setActiveCat] = useState("J1-EAST");

  useEffect(() => {
    fetch("/teams.csv")
      .then((r) => r.text())
      .then((text) => setTeams(parseCSV(text)));
  }, []);

  const handleScan = useCallback(() => {
    if (!scanText.trim() || teams.length === 0) return;
    const matches = parseMatches(scanText, teams);
    setScanResults(
      matches
        .map(([h, a]) => {
          const hData = teams.find((t) => t.チーム名 === h);
          const aData = teams.find((t) => t.チーム名 === a);
          if (!hData || !aData) return null;
          return { h, a, probs: predictScore(hData, aData) };
        })
        .filter(Boolean) as { h: string; a: string; probs: ScoreProb[] }[]
    );
  }, [scanText, teams]);

  const handleLearnParse = useCallback(() => {
    const results = parseResults(learnRaw, teams);
    setLearnParsed(results);
    setLearnMsg(results.length === 0 ? "試合結果を検出できませんでした。" : "");
  }, [learnRaw, teams]);

  const handleLearn = useCallback(() => {
    if (learnParsed.length === 0) return;
    const updated = teams.map((t) => ({ ...t }));
    let count = 0;

    for (const { h, hScore, a: aName, aScore } of learnParsed) {
      const hi = updated.findIndex((t) => t.チーム名 === h);
      const ai = updated.findIndex((t) => t.チーム名 === aName);
      if (hi === -1 || ai === -1) continue;

      updated[hi].攻撃力 = Math.min(25, Math.max(1, +(updated[hi].攻撃力 + (hScore - 1) * learnRate).toFixed(2)));
      updated[hi].守備力 = Math.min(25, Math.max(1, +(updated[hi].守備力 - (aScore - 1) * learnRate).toFixed(2)));
      updated[ai].攻撃力 = Math.min(25, Math.max(1, +(updated[ai].攻撃力 + (aScore - 1) * learnRate).toFixed(2)));
      updated[ai].守備力 = Math.min(25, Math.max(1, +(updated[ai].守備力 - (hScore - 1) * learnRate).toFixed(2)));
      count++;
    }

    setTeams(updated);

    const blob = new Blob([teamsToCSV(updated)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "teams.csv";
    a.click();
    URL.revokeObjectURL(url);

    setLearnMsg(`✅ ${count}試合を学習！CSVをダウンロードしました。public/teams.csv に上書きしてください。`);
  }, [learnParsed, learnRate, teams]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100dvh",
        maxWidth: "390px",
        margin: "0 auto",
        backgroundColor: "#000",
        color: "#fff",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          flexShrink: 0,
          height: "48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderBottom: "2px solid #FF6700",
          backgroundColor: "#000",
        }}
      >
        <h1
          style={{
            color: "#FF6700",
            fontWeight: 900,
            fontSize: "1.1rem",
            letterSpacing: "0.05em",
            textShadow: "0 0 10px rgba(255,103,0,0.5)",
            margin: 0,
          }}
        >
          Totti&apos;s WINNER Lab
        </h1>
      </header>

      {/* Main Scrollable Content */}
      <main
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "12px",
          paddingBottom: "80px",
        }}
      >
        {activeTab === "scan" && (
          <ScanTab
            scanText={scanText}
            setScanText={setScanText}
            scanResults={scanResults}
            onScan={handleScan}
            teamsLoaded={teams.length > 0}
          />
        )}
        {activeTab === "learn" && (
          <LearnTab
            learnRaw={learnRaw}
            setLearnRaw={setLearnRaw}
            learnParsed={learnParsed}
            learnRate={learnRate}
            setLearnRate={setLearnRate}
            learnMsg={learnMsg}
            onParse={handleLearnParse}
            onLearn={handleLearn}
          />
        )}
        {activeTab === "power" && (
          <PowerTab
            teams={teams}
            activeCat={activeCat}
            setActiveCat={setActiveCat}
          />
        )}
      </main>

      {/* Bottom Nav */}
      <nav
        style={{
          flexShrink: 0,
          height: "60px",
          display: "flex",
          backgroundColor: "#111",
          borderTop: "1px solid #333",
        }}
      >
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: "2px",
              border: "none",
              background: "none",
              cursor: "pointer",
              color: activeTab === tab.id ? "#FF6700" : "#666",
              transition: "color 0.15s",
            }}
          >
            <span style={{ fontSize: "1.3rem" }}>{tab.icon}</span>
            <span style={{ fontSize: "0.7rem", fontWeight: 700 }}>
              {tab.label}
            </span>
          </button>
        ))}
      </nav>
    </div>
  );
}

// ─── Scan Tab ─────────────────────────────────────────────────────────────────

function ScanTab({
  scanText,
  setScanText,
  scanResults,
  onScan,
  teamsLoaded,
}: {
  scanText: string;
  setScanText: (v: string) => void;
  scanResults: { h: string; a: string; probs: ScoreProb[] }[];
  onScan: () => void;
  teamsLoaded: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <textarea
        value={scanText}
        onChange={(e) => setScanText(e.target.value)}
        placeholder={"公式サイト等のコピペ、または「新潟 浦和」でOK"}
        style={{
          width: "100%",
          height: "120px",
          backgroundColor: "#111",
          color: "#fff",
          border: "1px solid #FF6700",
          borderRadius: "12px",
          padding: "12px",
          fontSize: "0.875rem",
          resize: "none",
          outline: "none",
          fontFamily: "system-ui, sans-serif",
        }}
      />
      <button
        onClick={onScan}
        disabled={!teamsLoaded}
        style={{
          width: "100%",
          height: "48px",
          borderRadius: "12px",
          fontWeight: 900,
          fontSize: "1rem",
          color: "#fff",
          border: "none",
          cursor: teamsLoaded ? "pointer" : "not-allowed",
          opacity: teamsLoaded ? 1 : 0.4,
          background: "linear-gradient(135deg, #FF6700 0%, #cc5200 100%)",
          boxShadow: "0 4px 15px rgba(255,103,0,0.4)",
        }}
      >
        ⚡ 期待値をスキャン！
      </button>

      {scanResults.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <p
            style={{
              color: "#FF6700",
              fontWeight: 700,
              fontSize: "0.875rem",
              margin: 0,
            }}
          >
            🎯 {scanResults.length}試合を検出！
          </p>
          {scanResults.map(({ h, a, probs }) => (
            <MatchCard
              key={`${h}|${a}`}
              h={h}
              a={a}
              honsen={probs[0]}
              osaeru={probs[1]}
              anakuji={pickAnakuji(probs)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function MatchCard({
  h,
  a,
  honsen,
  osaeru,
  anakuji,
}: {
  h: string;
  a: string;
  honsen: ScoreProb;
  osaeru: ScoreProb;
  anakuji: ScoreProb;
}) {
  const picks = [
    { label: "🎯本線", data: honsen },
    { label: "🔒抑え", data: osaeru },
    { label: "💥大穴", data: anakuji },
  ];

  return (
    <div
      style={{
        borderRadius: "12px",
        border: "1px solid #333",
        backgroundColor: "#0a0a0a",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          backgroundColor: "#111",
          padding: "8px 14px",
          borderBottom: "1px solid #333",
        }}
      >
        <p
          style={{
            color: "#fff",
            fontWeight: 900,
            fontSize: "0.875rem",
            margin: 0,
          }}
        >
          🏟️ {h} vs {a}
        </p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr" }}>
        {picks.map(({ label, data }, i) => (
          <div
            key={label}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              padding: "12px 4px",
              gap: "4px",
              borderLeft: i > 0 ? "1px solid #333" : undefined,
            }}
          >
            <span style={{ color: "#888", fontSize: "0.7rem" }}>{label}</span>
            <span
              style={{
                color: "#FF6700",
                fontWeight: 900,
                fontSize: "1.25rem",
                letterSpacing: "0.05em",
              }}
            >
              {data.score}
            </span>
            <span
              style={{ color: "#ccc", fontSize: "0.75rem", fontWeight: 700 }}
            >
              {(data.prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Learn Tab ────────────────────────────────────────────────────────────────

function LearnTab({
  learnRaw,
  setLearnRaw,
  learnParsed,
  learnRate,
  setLearnRate,
  learnMsg,
  onParse,
  onLearn,
}: {
  learnRaw: string;
  setLearnRaw: (v: string) => void;
  learnParsed: MatchResult[];
  learnRate: number;
  setLearnRate: (v: number) => void;
  learnMsg: string;
  onParse: () => void;
  onLearn: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <p style={{ color: "#888", fontSize: "0.8rem", margin: 0 }}>
        公式サイトのコピペをそのまま貼り付けてOK
      </p>

      {/* Raw paste area */}
      <textarea
        value={learnRaw}
        onChange={(e) => setLearnRaw(e.target.value)}
        placeholder={"Jリーグ公式サイトの試合結果ページをコピペ"}
        style={{
          width: "100%",
          height: "130px",
          backgroundColor: "#111",
          color: "#fff",
          border: "1px solid #FF6700",
          borderRadius: "12px",
          padding: "12px",
          fontSize: "0.875rem",
          resize: "none",
          outline: "none",
          fontFamily: "system-ui, sans-serif",
        }}
      />

      {/* Step 1: Parse */}
      <button
        onClick={onParse}
        style={{
          width: "100%",
          height: "44px",
          borderRadius: "12px",
          fontWeight: 900,
          fontSize: "0.95rem",
          color: "#fff",
          border: "1px solid #FF6700",
          cursor: "pointer",
          background: "transparent",
        }}
      >
        📋 結果を読み込む
      </button>

      {/* Parsed results preview */}
      {learnParsed.length > 0 && (
        <div
          style={{
            backgroundColor: "#0d0d0d",
            borderRadius: "12px",
            border: "1px solid #333",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "8px 12px",
              borderBottom: "1px solid #333",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ color: "#FF6700", fontWeight: 900, fontSize: "0.8rem" }}>
              ✅ {learnParsed.length}試合を検出
            </span>
            <span style={{ color: "#666", fontSize: "0.7rem" }}>
              PK戦は正規時間スコアで学習
            </span>
          </div>
          <div style={{ maxHeight: "200px", overflowY: "auto" }}>
            {learnParsed.map((r, idx) => {
              const hWin = r.hScore > r.aScore;
              const aWin = r.aScore > r.hScore;
              return (
                <div
                  key={`${r.h}|${r.a}|${idx}`}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "6px 12px",
                    borderBottom: idx < learnParsed.length - 1 ? "1px solid #1a1a1a" : undefined,
                    fontSize: "0.8rem",
                  }}
                >
                  <span style={{ color: hWin ? "#fff" : "#666", fontWeight: hWin ? 700 : 400, minWidth: "50px", textAlign: "right" }}>
                    {r.h}
                  </span>
                  <span
                    style={{
                      fontWeight: 900,
                      fontSize: "0.9rem",
                      padding: "2px 10px",
                      color: hWin ? "#4ade80" : aWin ? "#f87171" : "#aaa",
                      minWidth: "50px",
                      textAlign: "center",
                    }}
                  >
                    {r.hScore} - {r.aScore}
                  </span>
                  <span style={{ color: aWin ? "#fff" : "#666", fontWeight: aWin ? 700 : 400, minWidth: "50px" }}>
                    {r.a}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Learning rate slider */}
      <div
        style={{
          backgroundColor: "#111",
          borderRadius: "12px",
          padding: "12px",
          border: "1px solid #333",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <span style={{ color: "#fff", fontSize: "0.875rem", fontWeight: 700 }}>
            学習の強さ
          </span>
          <span style={{ color: "#FF6700", fontWeight: 900 }}>
            {learnRate.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0.01}
          max={0.2}
          step={0.01}
          value={learnRate}
          onChange={(e) => setLearnRate(Number(e.target.value))}
          style={{ width: "100%", accentColor: "#FF6700" }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            color: "#666",
            fontSize: "0.7rem",
            marginTop: "4px",
          }}
        >
          <span>0.01</span>
          <span>0.20</span>
        </div>
      </div>

      {/* Step 2: Execute */}
      <button
        onClick={onLearn}
        disabled={learnParsed.length === 0}
        style={{
          width: "100%",
          height: "48px",
          borderRadius: "12px",
          fontWeight: 900,
          fontSize: "1rem",
          color: "#fff",
          border: "none",
          cursor: learnParsed.length > 0 ? "pointer" : "not-allowed",
          opacity: learnParsed.length > 0 ? 1 : 0.35,
          background: "linear-gradient(135deg, #FF6700 0%, #cc5200 100%)",
          boxShadow: "0 4px 15px rgba(255,103,0,0.4)",
        }}
      >
        🧠 学習を実行
      </button>

      {learnMsg && (
        <div
          style={{
            borderRadius: "12px",
            padding: "12px",
            fontSize: "0.8rem",
            fontWeight: 700,
            backgroundColor: learnMsg.startsWith("✅")
              ? "rgba(34,197,94,0.15)"
              : "rgba(239,68,68,0.15)",
            color: learnMsg.startsWith("✅") ? "#4ade80" : "#f87171",
            border: `1px solid ${learnMsg.startsWith("✅") ? "#166534" : "#991b1b"}`,
          }}
        >
          {learnMsg}
        </div>
      )}
    </div>
  );
}

// ─── Power Tab ────────────────────────────────────────────────────────────────

function PowerTab({
  teams,
  activeCat,
  setActiveCat,
}: {
  teams: Team[];
  activeCat: string;
  setActiveCat: (v: string) => void;
}) {
  const filtered = teams
    .filter((t) => t.カテゴリ === activeCat)
    .sort((a, b) => b.攻撃力 - a.攻撃力);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      {/* Category tabs */}
      <div
        style={{
          display: "flex",
          gap: "6px",
          overflowX: "auto",
          paddingBottom: "4px",
        }}
      >
        {CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setActiveCat(cat.id)}
            style={{
              flexShrink: 0,
              padding: "6px 14px",
              borderRadius: "8px",
              fontSize: "0.75rem",
              fontWeight: 700,
              border: "none",
              cursor: "pointer",
              backgroundColor:
                activeCat === cat.id ? "#FF6700" : "#222",
              color: activeCat === cat.id ? "#fff" : "#888",
              outline:
                activeCat === cat.id ? "none" : "1px solid #444",
            }}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Team cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {filtered.map((team) => {
          const colors = TEAM_COLORS[team.チーム名] ?? ["#333333", "#555555"];
          const textColor = LIGHT_BG_COLORS.has(colors[0]) ? "#000" : "#fff";
          return (
            <div
              key={team.チーム名}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "12px 14px",
                borderRadius: "12px",
                border: "1px solid rgba(255,255,255,0.08)",
                background: `linear-gradient(135deg, ${colors[0]} 65%, ${colors[1]} 65%)`,
                color: textColor,
              }}
            >
              <span style={{ fontWeight: 900, fontSize: "1rem" }}>
                {team.チーム名}
              </span>
              <span
                style={{
                  fontSize: "0.75rem",
                  fontWeight: 900,
                  padding: "5px 10px",
                  borderRadius: "8px",
                  backgroundColor: "rgba(0,0,0,0.75)",
                  color: "#fff",
                  whiteSpace: "nowrap",
                }}
              >
                攻:{team.攻撃力.toFixed(1)} 守:{team.守備力.toFixed(1)} 監:{team.監督力} 成:{team.完成度}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
