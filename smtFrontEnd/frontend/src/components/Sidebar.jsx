import { useNavigate, useLocation } from "react-router-dom";
import { MdMonitor, MdBarChart, MdFactCheck } from "react-icons/md";

const menus = [
  { label: "실시간 모니터링", path: "/realtime", icon: MdMonitor },
  { label: "불량 현황",       path: "/defect",   icon: MdBarChart },
  { label: "불량 검수",       path: "/error",    icon: MdFactCheck },
];

export default function Sidebar() {
  const navigate  = useNavigate();
  const location  = useLocation();

  return (
    <div style={s.sidebar}>
      <div style={s.navLabel}>NAVIGATION</div>
      {menus.map(m => {
        const active = location.pathname === m.path;
        return (
          <div
            key={m.path}
            style={{ ...s.item, background: active ? "#1e40af" : "transparent", color: active ? "#fff" : "#94a3b8" }}
            onClick={() => navigate(m.path)}
          >
            <m.icon style={s.icon} />
            {m.label}
          </div>
        );
      })}
    </div>
  );
}

const s = {
  sidebar:  { width: 160, background: "#020817", borderRight: "1px solid #1e293b", display: "flex", flexDirection: "column", flexShrink: 0 },
  navLabel: { fontSize: 10, color: "#64748b", fontWeight: 700, letterSpacing: 1.5, padding: "14px 14px 8px" },
  item:     { display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer", borderRadius: 6, margin: "2px 6px", transition: "background 0.15s" },
  icon:     { fontSize: 16, width: 16, flexShrink: 0 },
};
