import { useState, useRef, useCallback, useEffect } from "react";
import { fetchSensorImportance } from "../../api/realTime";
import { BASE_URL } from "../../config";

// cause ID → 불량 유형 한글명 매핑
const CAUSE_MAP = {
  17: "미납",
  18: "납부족",
  19: "납쇼트",
  20: "납볼",
  21: "납좌표밀림",
  22: "납형성불량",
  23: "냉납",
  24: "밀림",
  25: "쇼트",
  26: "오삽",
  27: "미삽",
  28: "역삽",
  29: "뒤집힘",
  30: "일어섬",
  31: "납금감/핀홀",
  32: "납고드름",
};
const causeLabel = (id) => CAUSE_MAP[id] ?? `원인 ${id}`;

// 센서 컬럼명 → 한글 매핑
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
const SENSOR_UNIT = { 온도: "°C", 습도: "%", 진동: "m/s²", 가속도: "g", 소음: "dB" };
// 시계열 테이블 헤더용 — 한글 위, 단위 아래 2줄
const sensorShort = (key) => {
  const lower = key.toLowerCase();
  const rule = SENSOR_KO_RULES.find(r => r.keywords.some(k => lower.includes(k)));
  if (!rule) return <>{key}</>;
  const unit = SENSOR_UNIT[rule.ko] || "";
  return <><span style={{ color: "#e2e8f0" }}>{rule.ko}</span><br /><span style={{ color: "#94a3b8", fontWeight: 500, fontSize: 11 }}>({unit})</span></>;
};

// 제품명 클릭 시 표시되는 상세 팝업 (이미지 + 센서 데이터 + 불량 유형/원인)
export default function RealTimePopUp({ product, onClose }) {
  // 훅은 조건 분기 없이 항상 최상단에서 호출 (Rules of Hooks)
  const [imgError, setImgError] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [sensorTop3, setSensorTop3] = useState([]);
  const dragging = useRef(false);
  const startRef = useRef({ mx: 0, my: 0, px: 0, py: 0 });

  useEffect(() => {
    setImgError(false);
  }, [product?.img_path]);

  useEffect(() => {
    if (!product?.name) return;
    setSensorTop3([]);
    fetchSensorImportance(product.name)
      .then(({ sensor_importance }) => {
        const top3 = Object.entries(sensor_importance || {})
          .sort(([, a], [, b]) => b - a)
          .slice(0, 3)
          .map(([key, value]) => ({ sensor: sensorLabel(key), pct: (value * 100).toFixed(1) }));
        setSensorTop3(top3);
      })
      .catch(() => {});
  }, [product?.name]);

  const onMouseDown = useCallback((e) => {
    dragging.current = true;
    startRef.current = { mx: e.clientX, my: e.clientY, px: pos.x, py: pos.y };
    e.preventDefault();
  }, [pos]);

  const onMouseMove = useCallback((e) => {
    if (!dragging.current) return;
    setPos({
      x: startRef.current.px + (e.clientX - startRef.current.mx),
      y: startRef.current.py + (e.clientY - startRef.current.my),
    });
  }, []);

  const onMouseUp = useCallback(() => { dragging.current = false; }, []);

  if (!product) return null;

  const isDefect = product.status === "DEF";

  // 센서 데이터 — 가장 최신 시간 기준 마지막 행 사용
  const rawSensor = product.sensor_data || [];
  const latestRow = rawSensor.length > 0 ? rawSensor[rawSensor.length - 1] : null;
  const sensorRows = latestRow
    ? Object.entries(latestRow)
        .filter(([key]) => key !== "time" && key !== "timestamp")
        .map(([key, value]) => ({
          label: sensorLabel(key),
          value: typeof value === "number" ? value.toFixed(2) : String(value),
        }))
    : [];

  // cause(숫자 ID) → 한글 불량 유형명, confidence → 신뢰도
  const defectName = isDefect && product.cause != null ? causeLabel(product.cause) : null;
  const confidence = product.confidence != null ? (product.confidence * 100).toFixed(1) : null;

  // 센서 시계열 — timestamp 맨 앞, 나머지 센서 컬럼
  const tsKey = rawSensor.length > 0
    ? Object.keys(rawSensor[0]).find(k => k.includes("timestamp") || k === "time") || null
    : null;
  const sensorCols = rawSensor.length > 0
    ? Object.keys(rawSensor[0]).filter(k => !k.includes("timestamp") && k !== "time")
    : [];
  const imageUrl = product.name
    ? `${BASE_URL}/api/realtime/image/${product.name}`
    : null;

  return (
    <div style={s.overlay} onClick={onClose} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
      <div
        style={{ ...s.modal, transform: `translate(${pos.x}px, ${pos.y}px)` }}
        onClick={e => e.stopPropagation()}
      >

        {/* 헤더 — 드래그 핸들 */}
        <div style={{ ...s.header, cursor: "grab" }} onMouseDown={onMouseDown}>
          <div style={s.headerTitleRow}>
            <div style={s.headerAccent} />
            <span style={s.title}>제품 상세 정보</span>
            <button style={{ ...s.closeBtn, justifySelf: "end" }} onClick={onClose}>✕ 닫기</button>
          </div>
        </div>

        {/* 바디 — Grid A안 */}
        <div style={s.body}>

          {/* 왼쪽: 공정 이미지 */}
          <div style={s.imageSection}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, borderBottom: "1px solid #1e3a5f", paddingBottom: 10, marginBottom: 4, flexWrap: "wrap" }}>
              <span style={{ color: "#38bdf8", fontSize: 15, fontWeight: 700, whiteSpace: "nowrap" }}>공정 이미지</span>
              <span style={{ color: "#334155" }}>·</span>
              <span style={{ color: isDefect ? "#fca5a5" : "#86efac", fontSize: 14, fontWeight: 600, whiteSpace: "nowrap" }}>{isDefect ? "불량" : "정상"}</span>
              <span style={{ color: "#334155" }}>·</span>
              <span style={{ color: "#7dd3fc", fontSize: 14, fontWeight: 600, whiteSpace: "nowrap" }}>{product.process}</span>
              <span style={{ color: "#334155" }}>·</span>
              <span style={{ color: "#94a3b8", fontSize: 14, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={product.name}>{product.name}</span>
              {defectName && (
                <div style={{ marginLeft: "auto" }}>
                  <span style={s.defectTypeBadge}>{defectName}</span>
                </div>
              )}
            </div>
            <div style={s.imagePlaceholder}>
              {imgError
                ? <span style={{ color: "#475569", fontSize: 13 }}>데이터를 불러올 수 없습니다</span>
                : <img
                    src={imageUrl}
                    alt={product.name}
                    onError={() => setImgError(true)}
                    style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 8 }}
                  />
              }
            </div>
          </div>

          {/* 오른쪽: 센서 / 불량유형 / 불량원인 */}
          <div style={s.rightCol}>



            {/* 센서 시계열 테이블 */}
            {rawSensor.length > 0 && (
              <div style={{ ...s.panel, flex: 1, overflow: "auto" }}>
                <div style={{ ...s.sectionTitle, borderBottom: "1px solid #1e3a5f", paddingBottom: 6, marginBottom: 8 }}>센서 시계열</div>
                <table style={{ borderCollapse: "collapse", width: "100%", tableLayout: "fixed" }}>
                  <thead>
                    <tr>
                      {tsKey && <th style={{ ...s.th, fontSize: 11, padding: "4px 6px", whiteSpace: "nowrap" }}>시간</th>}
                      {sensorCols.map(col => (
                        <th key={col} style={{ ...s.th, fontSize: 11, padding: "4px 6px", whiteSpace: "nowrap" }}>{sensorShort(col)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rawSensor.map((row, i) => (
                      <tr key={i}>
                        {tsKey && (() => {
                          const raw = String(row[tsKey] ?? "-");
                          const [date, time] = raw.includes(" ") ? raw.split(" ") : [raw, ""];
                          return (
                            <td style={{ ...s.tdLabel, padding: "5px 8px", textAlign: "center", lineHeight: 1.5, whiteSpace: "nowrap" }}>
                              <span style={{ display: "block", color: "#cbd5e1", fontSize: 11, fontWeight: 500 }}>{date}</span>
                              <span style={{ display: "block", color: "#38bdf8", fontWeight: 700, fontSize: 11 }}>{time}</span>
                            </td>
                          );
                        })()}
                        {sensorCols.map(col => (
                          <td key={col} style={{ ...s.td, fontSize: 13, padding: "5px 8px" }}>
                            {typeof row[col] === "number" ? row[col].toFixed(2) : String(row[col] ?? "-")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* 불량 원인 TOP3 */}
            <div style={{ ...s.panel, flex: "none", padding: "8px 12px" }}>
              <div style={{ ...s.sectionTitle, fontSize: 12, borderBottom: "1px solid #1e3a5f", paddingBottom: 4, marginBottom: 6 }}>불량 원인 TOP3</div>
              {sensorTop3.length > 0
                ? sensorTop3.map((item, i) => (
                    <div key={i} style={{ ...s.causeRow, padding: "6px 10px", marginBottom: 4 }}>
                      <span style={{ ...s.causeSensor, minWidth: 40, textAlign: "center", fontSize: 11, padding: "2px 8px" }}>
                        {["1st","2nd","3rd"][i]}
                      </span>
                      <span style={{ ...s.causeSensor, background: "#1e3a5f", color: "#7dd3fc", border: "1px solid #38bdf8", fontSize: 11, padding: "2px 8px" }}>
                        {item.sensor}
                      </span>
                      <span style={{ ...s.causeDesc, fontSize: 13 }}>기여도 {item.pct}%</span>
                    </div>
                  ))
                : <span style={s.emptyText}>분석 데이터 대기 중</span>
              }
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

const s = {
  overlay: {
    position: "fixed", inset: 0,
    background: "rgba(0,0,0,0.65)",
    display: "flex", alignItems: "center", justifyContent: "center",
    zIndex: 1000,
  },
  modal: {
    background: "#0f172a",
    border: "1px solid #1e3a5f",
    borderRadius: 14,
    width: 1400,
    maxHeight: "95vh",
    display: "flex",
    flexDirection: "column",
    boxShadow: "0 8px 40px rgba(0,0,0,0.7)",
    overflow: "hidden",
  },
  header: {
    display: "grid",
    gridTemplateRows: "auto auto",
    gap: 6,
    padding: "16px 24px",
    borderBottom: "2px solid #1e3a5f",
    background: "#0d1f3c",
  },
  headerTitleRow: {
    display: "grid",
    gridTemplateColumns: "auto auto 1fr",
    alignItems: "center",
    gap: 12,
  },
  headerAccent: { width: 4, height: 28, background: "#38bdf8", borderRadius: 4 },
  title:    { fontSize: 20, fontWeight: 800, color: "#ffffff", letterSpacing: 0.5, whiteSpace: "nowrap" },
  subtitle: { fontSize: 13, color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  badge:    { fontSize: 12, fontWeight: 600, borderRadius: 6, padding: "3px 12px" },
  closeBtn: {
    background: "#ef4444", border: "none", color: "#fff",
    fontSize: 14, fontWeight: 700, cursor: "pointer",
    borderRadius: 6, padding: "6px 16px", letterSpacing: 0.5,
  },

  // Grid A안: 왼쪽 이미지 2fr / 오른쪽 패널 1fr
  body: {
    display: "grid",
    gridTemplateColumns: "2fr 1fr",
    gap: 16,
    padding: "16px 24px 20px 24px",
    flex: 1,
    minHeight: 0,
    overflow: "hidden",
  },
  imageSection: { display: "flex", flexDirection: "column", gap: 8, minHeight: 0, overflow: "hidden" },
  imagePlaceholder: {
    flex: 1,
    background: "#1e293b",
    borderRadius: 10,
    border: "2px dashed #334155",
    display: "flex", alignItems: "center", justifyContent: "center",
    overflow: "hidden",
    minHeight: 0,
  },
  rightCol: { display: "flex", flexDirection: "column", gap: 14, minHeight: 0, overflow: "hidden" },
  panel: {
    background: "#1e293b", borderRadius: 10, padding: "10px 14px",
    flex: 1, minHeight: 0, overflow: "hidden",
    width: "100%", boxSizing: "border-box",
  },

  sectionTitle: {
    fontSize: 14, fontWeight: 700, color: "#38bdf8",
    letterSpacing: 0.5, whiteSpace: "nowrap",
  },
  th: { color: "#cbd5e1", fontSize: 12, fontWeight: 700, padding: "5px 10px", borderBottom: "2px solid #334155", borderRight: "1px solid #334155", textAlign: "center", letterSpacing: 0.5 },
  td: { color: "#e2e8f0", fontSize: 12, padding: "5px 10px", textAlign: "center", borderBottom: "1px solid #253554", borderRight: "1px solid #253554" },
  tdLabel: { color: "#94a3b8", fontSize: 14, padding: "7px 10px", textAlign: "left", borderBottom: "1px solid #253554", borderRight: "1px solid #253554" },

  defectTypeBadge: {
    background: "#3b1f1f", border: "1px solid #ef4444",
    color: "#ef4444", fontSize: 11, fontWeight: 600,
    borderRadius: 6, padding: "3px 10px",
  },
  causeRow: {
    display: "flex", alignItems: "center", gap: 10,
    background: "#0f172a", borderRadius: 8,
    padding: "9px 12px", marginBottom: 6,
  },
  causeSensor: {
    background: "#3b1f1f", color: "#fca5a5",
    fontSize: 12, fontWeight: 700, borderRadius: 4,
    padding: "2px 10px", flexShrink: 0, border: "1px solid #ef4444",
  },
  causeDesc: { color: "#cbd5e1", fontSize: 13 },
  emptyText: { color: "#64748b", fontSize: 13, fontWeight: 600, padding: "6px 0" },
};
