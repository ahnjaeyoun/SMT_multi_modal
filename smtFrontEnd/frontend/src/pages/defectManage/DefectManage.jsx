import { useState, useEffect } from "react";
import { BASE_URL } from "../../config";
import { mockSummary, mockByType, mockCause, mockTrend, mockRecent } from "../../api/mock/defectManage.js";

const USE_MOCK = import.meta.env.VITE_MOCK === "true";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Label,
  LineChart, Line, XAxis, YAxis, CartesianGrid,
} from "recharts";

const API = `${BASE_URL}/api/defectmanage`;

const TOOLTIP_STYLE = {
  background: "#020c1b",
  border: "1px solid #38bdf8",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 11,
  boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
};

const RADIAN = Math.PI / 180;

const SHORT_NAMES = {
  "납좌표밀림": "좌표밀림",
  "납형성불량": "형성불량",
  "납금감/핀홀": "핀홀",
};
const shorten = (name) => SHORT_NAMES[name] || name;

function DonutSliceLabel({ cx, cy, midAngle, outerRadius, name, percent, fill }) {
  const radius = outerRadius + 18;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);
  return (
    <text
      x={x}
      y={y}
      fill={fill}
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
      fontSize={10}
      fontWeight={600}
    >
      {shorten(name)}
    </text>
  );
}



const CAUSE_COLORS = ["#ef4444", "#38bdf8", "#f59e0b", "#a78bfa", "#22c55e"];

const DONUT_COLORS = [
  "#ef4444", "#f97316", "#f59e0b", "#84cc16",
  "#22c55e", "#10b981", "#14b8a6", "#06b6d4",
  "#38bdf8", "#60a5fa", "#818cf8", "#a78bfa",
  "#c084fc", "#e879f9", "#fb7185", "#94a3b8",
];

const PROCESS_COLOR = { SD: "#38bdf8", PR: "#a78bfa" };

function SummaryCard({ label, value, sub, accent }) {
  return (
    <div style={s.summaryCard}>
      <div style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color: accent }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4, fontWeight: 500 }}>{sub}</div>}
    </div>
  );
}

export default function DefectManage() {
  const [process, setProcess] = useState("전체");
  const [trendTab, setTrendTab] = useState("일/월");

  const [summary, setSummary]     = useState({ total: "-", pr: "-", sd: "-", defect_rate: "-" });
  const [byType, setByType]       = useState([]);
  const [causeData, setCauseData] = useState([]);
  const [trendData, setTrendData] = useState({ 일별: [], 월별: [], 분기별: [], 년별: [] });
  const [recent, setRecent]       = useState([]);

  useEffect(() => {
    if (USE_MOCK) { setCauseData(mockCause.items); return; }
    const q = process === "전체" ? "" : `?process=${process}`;
    fetch(`${API}/cause${q}`)
      .then(r => r.json())
      .then(d => setCauseData(d.items || []))
      .catch(() => {});
  }, [process]);

  useEffect(() => {
    if (USE_MOCK) { setSummary(mockSummary); return; }
    fetch(`${API}/summary`)
      .then(r => r.json())
      .then(setSummary)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (USE_MOCK) { setByType(mockByType.items); return; }
    const q = process === "전체" ? "" : `?process=${process}`;
    fetch(`${API}/by-type${q}`)
      .then(r => r.json())
      .then(d => setByType(d.items || []))
      .catch(() => {});
  }, [process]);

  useEffect(() => {
    if (USE_MOCK) {
      const periods = trendTab === "일/월"
        ? [["daily", "일별"], ["monthly", "월별"]]
        : [["quarterly", "분기별"], ["yearly", "년별"]];
      periods.forEach(([period, key]) => {
        setTrendData(prev => ({ ...prev, [key]: mockTrend[period] || [] }));
      });
      return;
    }
    const periods = trendTab === "일/월"
      ? [["daily", "일별"], ["monthly", "월별"]]
      : [["quarterly", "분기별"], ["yearly", "년별"]];
    periods.forEach(([period, key]) => {
      fetch(`${API}/trend?period=${period}`)
        .then(r => r.json())
        .then(d => setTrendData(prev => ({ ...prev, [key]: d.trend || [] })))
        .catch(() => {});
    });
  }, [trendTab]);

  useEffect(() => {
    if (USE_MOCK) { setRecent(mockRecent.items); return; }
    const q = process === "전체" ? "" : `?process=${process}`;
    fetch(`${API}/recent${q}`)
      .then(r => r.json())
      .then(d => setRecent(d.items || []))
      .catch(() => {});
  }, [process]);

  const donutData = byType.map(d => ({ name: d.type, count: d.count }));

  const filteredRecent = recent;

  return (
    <div style={s.root}>
      <style>{`
        .recharts-wrapper { overflow: visible !important; }
        .recharts-wrapper svg { overflow: visible !important; }
        .recharts-surface { overflow: visible !important; }
      `}</style>
      <div style={s.summaryRow}>
        <SummaryCard label="전체 불량"    value={summary.total !== "-" ? `${summary.total}건` : "-"} sub="금일 누계"      accent="#38bdf8" />
        <SummaryCard label="PR 공정 불량" value={summary.pr    !== "-" ? `${summary.pr}건`    : "-"} sub="금일 누계"      accent="#a78bfa" />
        <SummaryCard label="SD 공정 불량" value={summary.sd    !== "-" ? `${summary.sd}건`    : "-"} sub="금일 누계"      accent="#38bdf8" />
        <SummaryCard label="불량률"       value={summary.defect_rate !== "-" ? `${summary.defect_rate}%` : "-"} sub="전체 검사 대비" accent="#f59e0b" />
      </div>

      {/* 공정 필터 */}
      <div style={s.filterRow}>
        {[
          { key: "전체", label: "전체" },
          { key: "PR",   label: "PR(사전)" },
          { key: "SD",   label: "SD(납땜)" },
        ].map(({ key: p, label }) => (
          <button
            key={p}
            style={{
              ...s.filterBtn,
              background: process === p ? "#1e293b" : "transparent",
              color: process === p
                ? (p === "SD" ? "#38bdf8" : p === "PR" ? "#a78bfa" : "#fff")
                : "#64748b",
              borderColor: process === p ? "#334155" : "transparent",
            }}
            onClick={() => setProcess(p)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* 메인 3단 레이아웃 */}
      <div style={s.mainLayout}>

        {/* 왼쪽: 도넛 2개 */}
        <div style={{ ...s.panel, flex: "0 0 43%" }}>
        <div style={s.splitGrid}>
          {/* 불량 유형별 분포 */}
          <div style={s.subSection}>
          <div style={s.panelTitle}>불량 유형별 분포</div>
          <div style={s.donutLayout}>
            {/* 왼쪽: 도넛 차트 */}
            <div style={s.donutChartWrap}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart style={{ overflow: "visible" }} margin={{ top: 24, right: 48, bottom: 24, left: 48 }}>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius="34%"
                    outerRadius="55%"
                    dataKey="count"
                    paddingAngle={2}
                    label={(props) => <DonutSliceLabel {...props} />}
                    labelLine={{ stroke: "#475569", strokeWidth: 1 }}
                  >
                    {donutData.map((_, i) => (
                      <Cell key={i} fill={DONUT_COLORS[i % DONUT_COLORS.length]} />
                    ))}
                    <Label content={({ viewBox }) => {
                      const { cx, cy } = viewBox;
                      const total = donutData.reduce((s, d) => s + d.count, 0);
                      const label = process === "전체" ? "전체 불량" : `${process} 불량`;
                      return (
                        <>
                          <text x={cx} y={cy - 7} textAnchor="middle" fill="#f1f5f9" fontSize={15} fontWeight={800}>{total}건</text>
                          <text x={cx} y={cy + 11} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={600}>{label}</text>
                        </>
                      );
                    }} />
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }}
                    formatter={(v, name) => [v + "건", name]}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* 오른쪽: 2열 그리드 표 */}
            <div style={s.tableWrap}>
              {(() => {
                const total = donutData.reduce((s, x) => s + x.count, 0);
                const half = Math.ceil(donutData.length / 2);
                const left  = donutData.slice(0, half);
                const right = donutData.slice(half);
                const TableCol = ({ items, offset }) => (
                  <table style={s.table}>
                    <tbody>
                      {items.map((d, i) => {
                        const idx = offset + i;
                        const pct = ((d.count / total) * 100).toFixed(1);
                        return (
                          <tr key={d.name} style={s.tr}>
                            <td style={s.tdRank}>
                              <span style={{ ...s.legendDot, background: DONUT_COLORS[idx % DONUT_COLORS.length] }} />
                            </td>
                            <td style={s.tdName}>{d.name}</td>
                            <td style={s.tdNum}>{d.count}건</td>
                            <td style={{ ...s.tdPct, color: DONUT_COLORS[idx % DONUT_COLORS.length] }}>{pct}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                );
                return (
                  <div style={s.twoColGrid}>
                    <TableCol items={left}  offset={0} />
                    <div style={s.colDivider} />
                    <TableCol items={right} offset={half} />
                  </div>
                );
              })()}
            </div>
          </div>
          </div>{/* /subSection 첫번째 */}


          {/* 불량 원인별 분포 */}
          <div style={s.subSection}>
            <div style={s.panelTitle}>불량 원인별 분포</div>
            <div style={s.donutLayout}>
              <div style={s.donutChartWrap}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart style={{ overflow: "visible" }} margin={{ top: 28, right: 56, bottom: 28, left: 56 }}>
                    <Pie
                      data={causeData}
                      cx="50%"
                      cy="50%"
                      innerRadius="34%"
                      outerRadius="60%"
                      dataKey="value"
                      paddingAngle={3}
                      label={(props) => <DonutSliceLabel {...props} />}
                      labelLine={{ stroke: "#475569", strokeWidth: 1 }}
                    >
                      {causeData.map((_, i) => (
                        <Cell key={i} fill={CAUSE_COLORS[i]} />
                      ))}
                      <Label content={({ viewBox }) => {
                        const { cx, cy } = viewBox;
                        return (
                          <>
                            <text x={cx} y={cy - 7} textAnchor="middle" fill="#f1f5f9" fontSize={15} fontWeight={800}>센서</text>
                            <text x={cx} y={cy + 11} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={600}>기여도</text>
                          </>
                        );
                      }} />
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }}
                      formatter={(v, name) => [`${v}%`, name]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div style={s.causeLegend}>
                {causeData.map((d, i) => (
                  <div key={d.name} style={s.legendRow}>
                    <span style={{ ...s.legendDot, background: CAUSE_COLORS[i] }} />
                    <span style={s.legendName}>{d.name}</span>
                    <span style={{ ...s.legendPct, color: CAUSE_COLORS[i] }}>{d.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>{/* /splitGrid */}
        </div>

        {/* 가운데: 불량 추이 */}
        <div style={s.midCol}>
          <div style={s.panel}>
          <div style={s.splitGrid}>
            {/* 위 섹션: 타이틀 + 탭 + 범례 + 차트 */}
            <div style={s.subSection}>
              <div style={s.panelTitle}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span>불량 추이</span>
                  {["일/월", "분기/년"].map(tab => (
                    <button key={tab} style={{ ...s.trendTab, background: trendTab === tab ? "#0f172a" : "transparent", color: trendTab === tab ? "#38bdf8" : "#64748b", borderColor: trendTab === tab ? "#334155" : "transparent" }} onClick={() => setTrendTab(tab)}>{tab}</button>
                  ))}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "#a78bfa" }}><span style={{ width: 16, height: 2, background: "#a78bfa", display: "inline-block", borderRadius: 2 }} />PR</span>
                  <span style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "#38bdf8" }}><span style={{ width: 16, height: 2, background: "#38bdf8", display: "inline-block", borderRadius: 2 }} />SD</span>
                </div>
              </div>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#ffffff", marginBottom: 4, flexShrink: 0 }}>{trendTab === "일/월" ? "일별" : "분기별"}</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendTab === "일/월" ? trendData["일별"] : trendData["분기별"]} margin={{ top: 4, right: 16, left: -10, bottom: 4 }}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }} />
                  {process !== "SD" && <Line type="monotone" dataKey="PR" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />}
                  {process !== "PR" && <Line type="monotone" dataKey="SD" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />}
                </LineChart>
              </ResponsiveContainer>
            </div>

  
            {/* 아래 섹션: 타이틀 + 차트 */}
            <div style={s.subSection}>
              <div style={s.panelTitle}>
                <span>{trendTab === "일/월" ? "월별" : "년별"}</span>
              </div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendTab === "일/월" ? trendData["월별"] : trendData["년별"]} margin={{ top: 4, right: 16, left: -10, bottom: 4 }}>
                  <CartesianGrid stroke="#1e293b" />
                  <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }} />
                  {process !== "SD" && <Line type="monotone" dataKey="PR" stroke="#a78bfa" strokeWidth={2} dot={{ r: 3 }} />}
                  {process !== "PR" && <Line type="monotone" dataKey="SD" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>{/* /splitGrid */}
          </div>
        </div>

        {/* 오른쪽: 최근 불량 목록 */}
        <div style={{ ...s.panel, flex: "0 0 13%" }}>
          <div style={s.panelTitle}>최근 불량 발생</div>
          <div style={s.recentList}>
            {filteredRecent.map((item) => (
              <div key={item.id} style={s.recentItem}>
                <div style={s.recentTop}>
                  <span style={{ fontSize: 11, color: "#cbd5e1", fontWeight: 600 }}>{item.time}</span>
                  <span style={{
                    ...s.processBadge,
                    background: item.process === "SD" ? "#1e3a5f" : "#2e1065",
                    color: PROCESS_COLOR[item.process],
                  }}>{item.process}</span>
                </div>
                <div style={{ fontSize: 12, color: "#f1f5f9", fontWeight: 700, marginTop: 4 }}>
                  {item.type || "-"}
                </div>
                <div style={s.recentBottom}>
                  <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600 }}>정확도</span>
                  <span style={{ fontSize: 12, fontWeight: 800, color: item.confidence >= 90 ? "#22c55e" : item.confidence >= 70 ? "#f59e0b" : "#ef4444" }}>
                    {item.confidence}%
                  </span>
                </div>
              </div>
            ))}
            {filteredRecent.length === 0 && (
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 12 }}>
                해당 공정 불량 없음
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}

const s = {
  root: {
    background: "#0f172a",
    height: "100%",
    fontFamily: "sans-serif",
    color: "#fff",
    boxSizing: "border-box",
    display: "flex",
    flexDirection: "column",
    padding: "10px 14px",
    gap: 10,
  },

  summaryRow: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 8,
    flexShrink: 0,
  },
  summaryCard: {
    background: "#1e293b",
    borderRadius: 8,
    padding: "10px 14px",
  },

  filterRow: {
    display: "flex",
    gap: 4,
    flexShrink: 0,
  },
  filterBtn: {
    border: "1px solid transparent",
    fontSize: 12,
    fontWeight: 700,
    padding: "5px 14px",
    borderRadius: 6,
    cursor: "pointer",
    transition: "all 0.15s",
  },

  mainLayout: {
    display: "flex",
    gap: 10,
    flex: 1,
    minHeight: 0,
  },

  panel: {
    flex: 1,
    background: "#1e293b",
    borderRadius: 10,
    padding: "12px 14px",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "visible",
  },
  panelTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: "#ffffff",
    paddingBottom: 10,
    marginBottom: 10,
    borderBottom: "1px solid #334155",
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },

  midCol: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 10,
    minHeight: 0,
  },

  trendTab: {
    fontSize: 11,
    fontWeight: 600,
    padding: "3px 10px",
    borderRadius: 5,
    border: "1px solid transparent",
    cursor: "pointer",
    transition: "all 0.15s",
  },


  recentList: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
    overflowY: "auto",
    flex: 1,
  },
  recentItem: {
    background: "#0f172a",
    borderRadius: 8,
    padding: "8px 10px",
    flexShrink: 0,
  },
  recentTop: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  processBadge: {
    fontSize: 10,
    fontWeight: 700,
    padding: "1px 6px",
    borderRadius: 4,
  },
  recentBottom: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 4,
  },

  splitGrid: {
    display: "grid",
    gridTemplateRows: "1fr 1fr",
    flex: 1,
    minHeight: 0,
    overflow: "visible",
  },

  subSection: {
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "visible",
  },

  donutLayout: {
    flex: 1,
    display: "flex",
    flexDirection: "row",
    gap: 8,
    minHeight: 0,
    overflow: "visible",
  },
  donutChartWrap: {
    position: "relative",
    flex: "0 0 50%",
    overflow: "visible",
  },

  tableWrap: {
    flex: 1,
    minWidth: 0,
    overflow: "hidden",
  },
  twoColGrid: {
    display: "grid",
    gridTemplateColumns: "1fr auto 1fr",
    height: "100%",
    gap: 0,
  },
  colDivider: {
    width: 0,
    margin: "0 6px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 10,
  },
  th: {
    color: "#64748b",
    fontWeight: 700,
    padding: "4px 4px",
    textAlign: "right",
    borderBottom: "1px solid #334155",
    whiteSpace: "nowrap",
  },
  tr: {
    height: 28,
  },
  tdRank: {
    padding: "3px 4px",
    textAlign: "center",
    verticalAlign: "middle",
  },
  tdName: {
    padding: "3px 4px",
    color: "#cbd5e1",
    textAlign: "left",
    whiteSpace: "nowrap",
  },
  tdNum: {
    padding: "3px 4px",
    color: "#f1f5f9",
    fontWeight: 700,
    textAlign: "right",
    whiteSpace: "nowrap",
  },
  tdPct: {
    padding: "3px 4px",
    fontWeight: 700,
    textAlign: "right",
    whiteSpace: "nowrap",
  },
  donutLegend: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    alignContent: "space-evenly",
    gap: "2px 6px",
    minWidth: 0,
    overflow: "hidden",
  },
  causeLegend: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "space-evenly",
    minWidth: 0,
    overflow: "hidden",
  },
  legendRow: {
    display: "flex",
    alignItems: "center",
    gap: 4,
    minWidth: 0,
  },
  legendDot: {
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: "50%",
    flexShrink: 0,
  },
  legendName: {
    flex: 1,
    fontSize: 10,
    color: "#94a3b8",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  legendCount: {
    fontSize: 10,
    color: "#cbd5e1",
    fontWeight: 700,
    whiteSpace: "nowrap",
  },
  legendPct: {
    fontSize: 10,
    fontWeight: 700,
    whiteSpace: "nowrap",
    minWidth: 34,
    textAlign: "right",
  },
};
