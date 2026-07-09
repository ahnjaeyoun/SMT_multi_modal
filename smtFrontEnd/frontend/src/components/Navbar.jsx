import { useState, useEffect, useRef } from "react";
import { BASE_URL } from "../config";

export default function Navbar() {
  const [now, setNow] = useState(new Date());
  const [alertCount, setAlertCount] = useState(0);
  const [kafkaLive, setKafkaLive] = useState(false);
  const lastReceivedRef = useRef(null);

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const fetchAlerts = () => {
      fetch(`${BASE_URL}/api/errorcheck/stats`)
        .then(r => r.json())
        .then(data => setAlertCount(data.pending ?? 0))
        .catch(() => {});
    };
    fetchAlerts();
    const id = setInterval(fetchAlerts, 5000);
    return () => clearInterval(id);
  }, []);

  // Kafka SSE 연결 감지 — 10초 안에 메시지 수신 시 파란색, 아니면 빨간색
  useEffect(() => {
    const es = new EventSource(`${BASE_URL}/api/realtime/stream`);
    es.onmessage = () => { lastReceivedRef.current = Date.now(); };
    es.onerror = () => { lastReceivedRef.current = null; };

    const checkId = setInterval(() => {
      const last = lastReceivedRef.current;
      setKafkaLive(last !== null && Date.now() - last < 20000);
    }, 1000);

    return () => { es.close(); clearInterval(checkId); };
  }, []);

  const timeStr = now.toTimeString().slice(0, 8);
  const dateStr = now.toISOString().slice(0, 10).replace(/-/g, "/");

  return (
    <div style={s.navbar}>
      <div style={s.left}>
        <img src="/favicon.svg" style={s.dot} alt="logo" />
        <span style={s.brand}>SMT 공정 관제 시스템</span>
        <span style={s.divider}>|</span>
        <span style={s.sub}>AI 불량 분석 플랫폼</span>
      </div>
      <div style={s.right}>
        <span style={s.status}><span style={{ ...s.circle, background: kafkaLive ? "#3b82f6" : "#ef4444" }} />LIVE STREAM</span>
        <span style={s.status}><span style={{ ...s.circle, background: "#22c55e" }} />AI MODELS</span>
        <span style={s.status}><span style={{ ...s.circle, background: "#f59e0b" }} />ALERTS: {alertCount}</span>
        <span style={s.time}>
          <span style={s.date}>{dateStr}</span>
          <br />{timeStr}
        </span>
      </div>
    </div>
  );
}

const s = {
  navbar: { display: "flex", justifyContent: "space-between", alignItems: "center", background: "#020817", borderBottom: "1px solid #1e293b", padding: "0 16px", height: 40, flexShrink: 0 },
  left:   { display: "flex", alignItems: "center", gap: 8 },
  dot:    { width: 20, height: 20 },
  brand:  { fontSize: 13, fontWeight: 700, color: "#f1f5f9", letterSpacing: 1 },
  divider:{ color: "#64748b", fontSize: 14 },
  sub:    { fontSize: 11, color: "#94a3b8", letterSpacing: 1 },
  right:  { display: "flex", alignItems: "center", gap: 16 },
  status: { display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#94a3b8", fontWeight: 600 },
  circle: { width: 6, height: 6, borderRadius: "50%", display: "inline-block" },
  time:   { fontSize: 13, fontWeight: 700, color: "#22c55e", textAlign: "right", lineHeight: 1.4 },
  date:   { fontSize: 10, color: "#94a3b8" },
};
