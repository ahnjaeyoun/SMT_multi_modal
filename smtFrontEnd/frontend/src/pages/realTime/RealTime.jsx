import { useState, useEffect, useRef } from "react";
import { BASE_URL } from "../../config";
import { fetchDashboardStats, fetchProducts, subscribeRealtimeStream, fetchThresholds } from "../../api/realTime";
import RealTimePopUp from "./RealTimePopUp";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, CartesianGrid, ReferenceLine
} from "recharts";

const TOOLTIP_STYLE = {
  background: "#020c1b",
  border: "1px solid #38bdf8",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 13,
  boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
};


const SENSOR_KO_RULES = [
  { keywords: ["temperature", "temp", "온도"], ko: "온도" },
  { keywords: ["humidity", "humid", "습도"], ko: "습도" },
  { keywords: ["vibration", "vibr", "진동"], ko: "진동" },
  { keywords: ["acceleration", "accel", "가속"], ko: "가속도" },
  { keywords: ["noise", "소음"], ko: "소음" },
];
const sensorLabel = (key) => {
  const lower = key.toLowerCase();
  const rule = SENSOR_KO_RULES.find(r => r.keywords.some(k => lower.includes(k)));
  return rule ? `${key} [${rule.ko}]` : key;
};

// Kafka payload의 base_name에서 process(PR/SD)와 status(NOR/DEF)를 파싱하는 함수
// base_name 예시: "PR_DEF_MF_A_20250808-091228_00001"
function parseKafkaItem(raw, index) {
  const name = raw.file_name || raw.base_name || "";
  const parts = name.toUpperCase().split("_");
  return {
    id:          name || index,
    name,
    process:     (raw.category || parts[0] || "??").toUpperCase(),
    status:      parts[1] || "??",
    predictions:  raw.predictions  || null,
    cause:        raw.cause        ?? null,
    confidence:   raw.confidence   ?? null,
    sensor_data:  raw.sensor_data  || [],
    img_path:   raw.img_path   || null,
  };
}


// DB 연동 전 기본값 (백엔드 미연결 시 표시)
const DEFAULT_STATS = {
  total: 0, defect: 0, normal: 0,
  avg_accuracy: 0, defect_rate: 0, per_minute: 0,
};

export default function RealTime() {
  const [sensorDataPR, setSensorDataPR] = useState([]);
  const [sensorDataSD, setSensorDataSD] = useState([]);
  const [sensorTab, setSensorTab] = useState("PR");
  const [thresholds, setThresholds] = useState({ pr: {}, sd: {} });
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState("asc");
  const [stats, setStats] = useState(DEFAULT_STATS);
  const [filterProcess, setFilterProcess] = useState("");
  const [filterStatus, setFilterStatus] = useState("");
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const esRef = useRef(null);

  // 센서 임계값 — 최초 1회 로드
  useEffect(() => {
    fetchThresholds().then(setThresholds).catch(() => {});
  }, []);

  // Kafka products에서 공정별 가장 최근 항목 (카메라 뷰용)
  const lastPR = products.find(p => p.process === "PR") || null;
  const lastSD = products.find(p => p.process === "SD") || null;

  // products 기반 실시간 통계 계산
  const uniqueProducts = [...new Map(products.map(p => [p.name, p])).values()];
  const liveDefect = uniqueProducts.filter(p => p.status === "DEF").length;

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  };

  const sortIcon = (key) => {
    if (sortKey !== key) return " ↕";
    return sortDir === "asc" ? " ↑" : " ↓";
  };

  // 센서 더미 데이터 + DB 통계 3초 폴링
  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchDashboardStats();
        setStats(data);
      } catch {
        // 백엔드 미연결 시 기본값 유지
      }
    };

    loadStats();
    const id = setInterval(loadStats, 3000);
    return () => clearInterval(id);
  }, []);


  // 최초 진입 시 DB에서 최근 20개 로드
  useEffect(() => {
    fetchProducts(20)
      .then(({ items }) => {
        const parsed = items.map(item => ({
          id:          item.file_name,
          name:        item.file_name,
          process:     (item.category || "").toUpperCase(),
          status:      item.status,
          cause:       item.cause ?? null,
          confidence:  item.confidence ?? null,
          predictions: item.predictions ?? null,
          sensor_data: item.sensor_data || [],
          img_path:  item.img_path ?? null,
        }));
        setProducts([...new Map(parsed.map(p => [p.name, p])).values()]);
      })
      .catch(() => {});
  }, []);

  // Kafka SSE 실시간 수신 — 제품 리스트 & 센서 그래프 업데이트
  useEffect(() => {
    esRef.current = subscribeRealtimeStream((raw) => {
      // 신규 항목 맨 앞에 추가 (중복 제거 후 최대 500건 유지)
      setProducts(prev => {
        const next = [parseKafkaItem(raw, Date.now()), ...prev];
        return [...new Map(next.map(p => [p.name, p])).values()].slice(0, 500);
      });

      // sensor_data → 공정별 그래프 누적 (최근 100포인트 유지)
      if (raw.sensor_data?.length) {
        const now = new Date();
        const timeLabel = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}`;
        const tagged = raw.sensor_data.map(row => ({ ...row, time: timeLabel }));
        const process = (raw.category || "").toUpperCase();
        if (process === "SD") {
          setSensorDataSD(prev => [...prev, ...tagged].slice(-100));
        } else {
          setSensorDataPR(prev => [...prev, ...tagged].slice(-100));
        }
      }
    });

    return () => { esRef.current?.close(); };
  }, []);

  return (
    <div style={s.root}>
      <RealTimePopUp key={selectedProduct?.name} product={selectedProduct} onClose={() => setSelectedProduct(null)} />
      <div style={s.content}>
      {/* 상단 카드 한 줄 (대형 4 + 소형 5) */}
      <div style={{ ...s.grid9, gridTemplateColumns: "repeat(5,1fr)" }}>
        <StatCard label="총 처리량"  value={`${stats.total.toLocaleString()}건`}   sub={`정상 처리 ${stats.normal.toLocaleString()}건`} valueColor="#3b82f6" />
        <StatCard label="정상 처리"  value={`${stats.normal.toLocaleString()}건`}  sub="Normal Processing" valueColor="#22c55e" />
        <StatCard label="불량 감지"  value={`${stats.defect.toLocaleString()}건`}  sub={`최근 불량률 ${stats.defect_rate}%`} valueColor="#ef4444" />
        <StatCard label="불량률"     value={`${stats.defect_rate}%`}              sub="Defect Rate" valueColor="#ef4444" />
        <MiniCard label="모델 가동" sub="AI Model Status" modelStatuses={[1,1,1,1]} />
      </div>

      {/* 메인 2단 레이아웃 */}
      <div style={s.mainLayout}>

        {/* 왼쪽: 사이드바 + 카메라 */}
        <div style={s.leftCol}>
          <div style={s.middle}>
            {/* 사이드바 */}
            <div style={s.sidebar}>
              <div style={s.sideTitle}>실시간 제품 리스트</div>
              <div style={{ display: "flex", gap: 6 }}>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontSize: 11, color: "#e2e8f0", letterSpacing: 0.5 }}>공정</span>
                  <select style={s.select} value={filterProcess} onChange={e => setFilterProcess(e.target.value)}>
                    <option value="">전체</option>
                    <option value="PR">PR (사전공정)</option>
                    <option value="SD">SD (납땜공정)</option>
                  </select>
                </div>
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontSize: 11, color: "#e2e8f0", letterSpacing: 0.5 }}>정상/불량</span>
                  <select style={s.select} value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
                    <option value="">전체</option>
                    <option value="NOR">정상</option>
                    <option value="DEF">불량</option>
                  </select>
                </div>
              </div>
              <div style={s.dataBox}>
                {/* 테이블 헤더 */}
                <div style={s.tableHeader}>
                  <button style={{ ...s.thBtn, flex: "0 0 70px" }} onClick={() => handleSort("process")}>공정</button>
                  <div style={{ ...s.thBtn, flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>제품명</div>
                  <button style={{ ...s.thBtn, flex: "0 0 70px" }} onClick={() => handleSort("status")}>정상/불량</button>
                </div>
                {/* 데이터 영역 */}
                <div style={s.tableBody}>
                  {(() => {
                    const filtered = [...new Map(products.map(p => [p.name, p])).values()]
                      .filter(p => !filterProcess || p.process === filterProcess)
                      .filter(p => !filterStatus  || p.status  === filterStatus)
                      .sort((a, b) => {
                        if (!sortKey) return 0;
                        const va = a[sortKey], vb = b[sortKey];
                        return sortDir === "asc" ? va.localeCompare(vb) : vb.localeCompare(va);
                      });

                    if (filtered.length === 0) {
                      return <div style={{ ...s.dataEmpty, width: "100%", paddingTop: 40 }}>데이터를 불러오면<br />여기에 표시됩니다</div>;
                    }

                    return filtered.map(p => (
                      <div key={p.id} style={{ ...s.tableRow, cursor: "pointer", width: "100%" }} onClick={() => setSelectedProduct(p)}>
                        <span style={{ ...s.tableCell, flex: "0 0 70px", color: "#38bdf8", fontWeight: 700 }}>{p.process}</span>
                        <span style={{ ...s.tableCell, flex: 1, color: "#cbd5e1", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", justifyContent: "flex-start" }} title={p.name}>{p.name}</span>
                        <span style={{ ...s.tableCell, flex: "0 0 70px", color: p.status === "DEF" ? "#ef4444" : "#22c55e", fontWeight: 700 }}>
                          {p.status === "DEF" ? "불량" : "정상"}
                        </span>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>

            {/* 카메라 뷰 */}
            <div style={s.camWrap}>
              <div style={s.camHeader}>
                <span>실시간 AI 객체 탐지</span>
              </div>
              <div style={{ display: "flex", flexDirection: "row", gap: 6, padding: 8, flex: 1, minHeight: 0, overflow: "hidden" }}>
                <div style={s.camSlot}>
                  <div style={s.camLabel}>
                    <span>PR (사전공정)</span>
                    {lastPR && (
                      <>
                        <span style={s.camLabelSep}>·</span>
                        <span style={s.camLabelName} title={lastPR.name}>{lastPR.name}</span>
                        <span style={{ ...s.camLabelBadge, color: lastPR.status === "DEF" ? "#fca5a5" : "#86efac" }}>
                          {lastPR.status === "DEF" ? "불량" : "정상"}
                        </span>
                      </>
                    )}
                  </div>
                  <div style={s.camPlaceholder}>
                    {lastPR?.name
                      ? <CamImage src={`${BASE_URL}/api/realtime/image/${lastPR.name}`} alt={lastPR.name} />
                      : null}
                  </div>
                </div>
                <div style={s.camSlot}>
                  <div style={s.camLabel}>
                    <span>SD (납땜공정)</span>
                    {lastSD && (
                      <>
                        <span style={s.camLabelSep}>·</span>
                        <span style={s.camLabelName} title={lastSD.name}>{lastSD.name}</span>
                        <span style={{ ...s.camLabelBadge, color: lastSD.status === "DEF" ? "#fca5a5" : "#86efac" }}>
                          {lastSD.status === "DEF" ? "불량" : "정상"}
                        </span>
                      </>
                    )}
                  </div>
                  <div style={s.camPlaceholder}>
                    {lastSD?.name
                      ? <CamImage src={`${BASE_URL}/api/realtime/image/${lastSD.name}`} alt={lastSD.name} />
                      : null}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 오른쪽: 센서 그래프 + 불량 분석 */}
        <div style={s.rightCol}>
          {/* 센서 그래프 */}
          <div style={{ ...s.panel, flex: 3 }}>
            <div style={{ display: "flex", alignItems: "center", marginBottom: 8 }}>
              <span style={{ ...s.panelTitle, marginBottom: 0 }}>실시간 센서 데이터 그래프</span>
              <div style={{ marginLeft: "auto", display: "flex" }}>
              {["PR", "SD"].map(tab => (
                <button key={tab} onClick={() => setSensorTab(tab)} style={{
                  padding: "3px 14px", fontSize: 11, fontWeight: 700, cursor: "pointer",
                  borderRadius: tab === "PR" ? "6px 0 0 6px" : "0 6px 6px 0",
                  border: "1px solid #334155",
                  background: sensorTab === tab ? "#38bdf8" : "#1e293b",
                  color: sensorTab === tab ? "#0f172a" : "#94a3b8",
                }}>
                  {tab}
                </button>
              ))}
              </div>
            </div>
            {(() => {
              const data = sensorTab === "SD" ? sensorDataSD : sensorDataPR;
              if (data.length === 0)
                return <div style={{ color: "#475569", fontSize: 12, textAlign: "center", paddingTop: 20 }}>센서 데이터 대기 중...</div>;
              const ORDER  = ["소음", "습도", "온도", "진동", "가속도"];
              const COLORS = { 소음: "#ef4444", 습도: "#3b82f6", 온도: "#f59e0b", 진동: "#22c55e", 가속도: "#a855f7" };
              const FALLBACK = ["#3b82f6","#22c55e","#f59e0b","#a855f7","#ef4444"];
              const allKeys = Object.keys(data[0]).filter(k => k !== "time" && k !== "timestamp");
              const keys = [...ORDER.filter(k => allKeys.includes(k)), ...allKeys.filter(k => !ORDER.includes(k))];
              const tabThresholds = thresholds[sensorTab] || thresholds[sensorTab.toLowerCase()] || {};
              return keys.map((key, i) => {
                // 센서 컬럼명에서 임계값 매핑 (includes로 매칭)
                const thKey = Object.keys(tabThresholds).find(k => key.toLowerCase().includes(k.toLowerCase()) || k.toLowerCase().includes(key.toLowerCase().split("(")[0]));
                const mean = thKey ? tabThresholds[thKey]?.mean : undefined;
                return <SensorChart key={key} dataKey={key} color={COLORS[key] || FALLBACK[i % FALLBACK.length]} data={data} mean={mean} />;
              });
            })()}
          </div>

          {/* 불량 판정 */}
          <div style={{ ...s.panel, flex: 1 }}>
          <div style={s.panelTitle}>불량 판정</div>
          <div style={s.defectSummary}>
            <table style={{ borderCollapse: "collapse", width: "100%" }}>
              <thead>
                <tr>
                  {["항목", "값"].map(h => (
                    <th key={h} style={{ color: "#64748b", fontWeight: 600, padding: "4px 10px", borderBottom: "1px solid #334155", textAlign: "center", fontSize: 11, letterSpacing: 0.5 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { label: "불량 감지", value: `${stats.defect}건`,      color: "#ef4444" },
                  { label: "불량률",    value: `${stats.defect_rate}%`, color: "#ef4444" },
                  { label: "총 처리량", value: `${stats.total.toLocaleString()}건`, color: "#ffffff" },
                ].map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #1e3a5f" }}>
                    <td style={{ color: "#94a3b8", padding: "5px 10px", textAlign: "left", fontSize: 11 }}>{row.label}</td>
                    <td style={{ color: row.color, fontWeight: 700, padding: "5px 10px", textAlign: "right", fontSize: 13 }}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          </div>
        </div>
      </div>
      </div>
    </div>
  );
}

// 대형 통계 카드
function StatCard({ label, value, sub, valueColor = "#ffffff" }) {
  return (
    <div style={{ ...s.statCard, minWidth: 0 }}>
      <div style={{ color: "#94a3b8", fontSize: 10, marginBottom: 2 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 15, fontWeight: 800, color: valueColor }}>{value}</span>
      </div>
      <div style={{ fontSize: 10, color: "#94a3b8" }}>{sub}</div>
    </div>
  );
}

// 소형 카드 — modelStatuses: 모델별 상태 배열 (1=정상, 0=오류), 추후 실제 데이터 연결
function MiniCard({ label, value, sub, modelStatuses }) {
  const total   = modelStatuses ? modelStatuses.length : null;
  const running = modelStatuses ? modelStatuses.filter(s => s === 1).length : null;
  const color   = modelStatuses
    ? (running === total ? "#22c55e" : "#ef4444")
    : "#ffffff";
  const display = modelStatuses ? `${running}/${total}` : value;
  return (
    <div style={s.miniCard}>
      <div style={{ color: "#94a3b8", fontSize: 10 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ fontSize: 15, fontWeight: 800, color }}>{display}</span>
      </div>
      <div style={{ fontSize: 10, color: "#94a3b8" }}>{sub}</div>
    </div>
  );
}

function CamImage({ src, alt }) {
  const [err, setErr] = useState(false);
  useEffect(() => { setErr(false); }, [src]);
  return err
    ? <span style={{ color: "#475569", fontSize: 12 }}>데이터를 불러올 수 없습니다</span>
    : <img src={src} alt={alt} style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 6 }} onError={() => setErr(true)} />;
}

function SensorChart({ dataKey, color, data, mean }) {
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ fontSize: 10, color: "#fff", fontWeight: 700, marginBottom: 2, textAlign: "left", display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: color, verticalAlign: "middle" }} />
        {sensorLabel(dataKey)}
        {mean !== undefined && (
          <span style={{ fontSize: 9, color: "#94a3b8", fontWeight: 500 }}>임계 {mean.toFixed(2)}</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={60}>
        <LineChart data={data} margin={{ top: 2, right: 4, left: -20, bottom: 0 }}>
          <XAxis dataKey="time" tick={{ fontSize: 6, fill: "#94a3b8" }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 6, fill: "#94a3b8" }} width={30} />
          <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: "#e2e8f0" }} labelStyle={{ color: "#94a3b8" }} />
          {mean !== undefined && (
            <ReferenceLine y={mean} stroke="#ffffff" strokeDasharray="4 3" strokeWidth={1} strokeOpacity={0.4} />
          )}
          <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// 스타일
const s = {
  root: { background: "#0f172a", height: "100vh", fontFamily: "sans-serif", color: "#fff", boxSizing: "border-box", display: "flex", flexDirection: "column", overflow: "hidden" },
  content: { padding: "4px 10px 6px 10px", flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" },
  grid9: { display: "grid", gridTemplateColumns: "repeat(9,1fr)", gap: 8, marginBottom: 8, flexShrink: 0 },
  statCard: { background: "#1e293b", borderRadius: 8, padding: "3px 8px", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" },
  miniCard: { background: "#1e293b", borderRadius: 8, padding: "3px 8px" },
  mainLayout: { display: "flex", gap: 8, alignItems: "stretch", flex: 1, minHeight: 0, overflow: "hidden" },
  leftCol: { flex: "0 0 82%", minWidth: 0, minHeight: 0, display: "flex", flexDirection: "column" },
  rightCol: { flex: "0 0 18%", minWidth: 0, display: "flex", flexDirection: "column", gap: 6 },
  middle: { display: "flex", gap: 8, flex: 1, minHeight: 0, overflow: "hidden" },
  sidebar: { background: "#1e293b", borderRadius: 10, padding: "10px 10px 10px 10px", width: 360, flexShrink: 0, display: "flex", flexDirection: "column" },
  sideTitle: { fontSize: 13, fontWeight: 700, color: "#ffffff", paddingBottom: 8, marginBottom: 6, borderBottom: "1px solid #334155" },
  select: { width: "100%", background: "#334155", border: "none", color: "#fff", borderRadius: 6, padding: "3px 6px", marginBottom: 6, fontSize: 11 },
  lineItem: { fontSize: 12, color: "#cbd5e1", padding: "6px 8px", borderRadius: 6, cursor: "pointer", marginBottom: 2 },
  camWrap: { flex: 1, minHeight: 0, background: "#1e293b", borderRadius: 10, overflow: "hidden", display: "flex", flexDirection: "column" },
  camHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 12px", borderBottom: "1px solid #334155", fontSize: 13, fontWeight: 700, color: "#ffffff" },
  liveBadge: { background: "#f59e0b", color: "#000", fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 4 },
  dataBox: { flex: 1, background: "#0f172a", borderRadius: 8, marginTop: 8, display: "flex", flexDirection: "column", border: "1px solid #1e293b", minHeight: 0, overflow: "hidden" },
  tableHeader: { display: "flex", borderBottom: "1px solid #1e293b", flexShrink: 0 },
  thBtn: { flex: 1, background: "#0f172a", border: "none", borderRight: "1px solid #1e293b", color: "#94a3b8", fontSize: 10, fontWeight: 700, padding: "5px 0", cursor: "pointer", textAlign: "center", letterSpacing: 0.5 },
  tableBody: { flex: 1, display: "flex", flexDirection: "column", overflowY: "auto" },
  dataEmpty: { color: "#334155", fontSize: 12, textAlign: "center", lineHeight: 1.8 },
  tableRow: { display: "flex", borderBottom: "1px solid #1e293b", padding: "5px 4px", alignItems: "center" },
  tableCell: { fontSize: 10, textAlign: "center", padding: "0 0", display: "flex", alignItems: "center", justifyContent: "center" },
  camSlot: { flex: 1, minWidth: 0, minHeight: 0, background: "#0f172a", borderRadius: 8, display: "flex", flexDirection: "column", overflow: "hidden" },
  camLabel: { fontSize: 13, fontWeight: 700, color: "#e2e8f0", padding: "6px 12px", borderBottom: "1px solid #1e293b", display: "flex", alignItems: "center", gap: 8, overflow: "hidden", letterSpacing: 0.3 },
  camLabelSep: { color: "#475569" },
  camLabelName: { color: "#94a3b8", fontWeight: 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, fontSize: 11 },
  camLabelBadge: { fontSize: 11, fontWeight: 700, flexShrink: 0 },
  camPlaceholder: { flex: 1, minHeight: 0, overflow: "hidden", display: "flex", alignItems: "center", justifyContent: "center", border: "2px dashed #334155", margin: 6, borderRadius: 6 },
  panel: { background: "#1e293b", borderRadius: 10, padding: "10px 12px 12px 12px", flex: 1, display: "flex", flexDirection: "column" },
  panelTitle: { fontSize: 13, fontWeight: 700, paddingBottom: 8, marginBottom: 8, color: "#ffffff", borderBottom: "1px solid #334155" },
  chartLabel: { fontSize: 12, color: "#94a3b8", marginBottom: 4 },
  defectSummary: { display: "flex", alignItems: "center", gap: 12, background: "#0f172a", borderRadius: 8, padding: 8, marginBottom: 8 },
  defectRow: { display: "flex", justifyContent: "space-between", gap: 16, fontSize: 12, marginBottom: 4 },
  defectItem: { display: "flex", gap: 8, alignItems: "center", background: "#0f172a", borderRadius: 6, padding: 7, marginBottom: 6 },
  defectThumb: { width: 44, height: 36, background: "#334155", borderRadius: 4, flexShrink: 0 },
};
