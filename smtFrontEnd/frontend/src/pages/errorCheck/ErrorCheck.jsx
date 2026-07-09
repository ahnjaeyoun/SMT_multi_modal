import { useState, useEffect } from "react";
import { BASE_URL } from "../../config";

const hideDateClearStyle = document.createElement("style");
hideDateClearStyle.textContent = `
  input[type="date"]::-webkit-clear-button { display: none; }
  input[type="date"]::-webkit-inner-spin-button { display: none; }
`;
document.head.appendChild(hideDateClearStyle);


const PRIORITY_COLOR = { HIGH: "#ef4444", MEDIUM: "#f59e0b", LOW: "#22c55e" };
const STATUS_COLOR   = { 대기: "#94a3b8", 승인: "#22c55e", 반려: "#ef4444" };

function confidenceColor(v) {
  if (v >= 90) return "#22c55e";
  if (v >= 80) return "#f59e0b";
  return "#ef4444";
}

const TABS = ["전체", "검수 대기", "승인", "반려"];


export default function ErrorCheck() {
  const [items, setItems]         = useState([]);
  const [total, setTotal]         = useState(0);
  const [stats, setStats]         = useState({ total: 0, pending: 0, approved: 0, rejected: 0 });
  const [loading, setLoading]     = useState(false);
  const [activeTab, setActiveTab] = useState("전체");
  const [selected, setSelected]   = useState(null);
  const [pageSize, setPageSize]   = useState(25);
  const [offset, setOffset]       = useState(0);

  // 정렬
  const [sortKey, setSortKey]   = useState(null);
  const [sortDir, setSortDir]   = useState("asc");
  // 필터
  const [filterProcess, setFilterProcess] = useState("전체");
  const [filterType, setFilterType]       = useState("전체");
  const [filterStatus, setFilterStatus]   = useState("전체");
  const [filterDateFrom, setFilterDateFrom] = useState("");
  const [filterDateTo, setFilterDateTo]     = useState("");

  // 확인 팝업 상태
  const [confirm, setConfirm] = useState(null); // { id, action: "approve"|"reject", noShow: false }

  function requestAction(id, action) {
    const skipKey = `errorcheck_skip_confirm_${action}`;
    if (sessionStorage.getItem(skipKey) === "1") {
      action === "approve" ? doApprove(id) : doReject(id);
    } else {
      setConfirm({ id, action, noShow: false });
    }
  }

  function handleConfirm() {
    if (!confirm) return;
    const { id, action, noShow } = confirm;
    if (noShow) sessionStorage.setItem(`errorcheck_skip_confirm_${action}`, "1");
    action === "approve" ? doApprove(id) : doReject(id);
    setConfirm(null);
  }

  function doApprove(id) {
    fetch(`${BASE_URL}/api/errorcheck/items/${id}/approve`, { method: "PATCH" })
      .then(() => fetchStats())
      .catch(() => {});
    setItems(prev => prev.map(item => item.id === id ? { ...item, status: "승인", status_num: 1 } : item));
    if (selected?.id === id) setSelected(s => ({ ...s, status: "승인", status_num: 1 }));
  }

  function doReject(id) {
    fetch(`${BASE_URL}/api/errorcheck/items/${id}/reject`, { method: "PATCH" })
      .then(() => fetchStats())
      .catch(() => {});
    setItems(prev => prev.map(item => item.id === id ? { ...item, status: "반려", status_num: 2 } : item));
    if (selected?.id === id) setSelected(s => ({ ...s, status: "반려", status_num: 2 }));
  }

  function toggleSort(key) {
    if (sortKey === key) {
      if (sortDir === "asc")  setSortDir("desc");
      else if (sortDir === "desc") { setSortKey(null); setSortDir("asc"); }
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function sortIcon(key) {
    if (sortKey !== key) return " ↕";
    return sortDir === "asc" ? " ↑" : " ↓";
  }

  function fetchStats() {
    fetch(`${BASE_URL}/api/errorcheck/stats`)
      .then(r => r.json())
      .then(data => setStats(data))
      .catch(() => {});
  }

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    const limit = pageSize >= 9999 ? 9999 : pageSize;

    // 탭 status 우선, 탭이 전체일 때만 컬럼 드롭다운 status 적용
    const tabStatus = activeTab === "검수 대기" ? "대기"
                    : activeTab === "승인"     ? "승인"
                    : activeTab === "반려"     ? "반려"
                    : "";
    const effectiveStatus = tabStatus || (filterStatus !== "전체" ? filterStatus : "");

    const params = new URLSearchParams({ limit, offset });
    if (effectiveStatus) params.append("status", effectiveStatus);
    if (filterProcess !== "전체") params.append("process", filterProcess);

    setLoading(true);
    fetch(`${BASE_URL}/api/errorcheck/items?${params}`)
      .then(r => r.json())
      .then(data => {
        setItems(data.items);
        setTotal(data.total);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [pageSize, offset, activeTab, filterStatus, filterProcess]);

  const allTypes = ["전체", ...Array.from(new Set(items.map(i => i.type))).sort()];

  let displayed = items;
  // 유형·날짜는 DB에 필드 없어 클라이언트 필터 유지, 공정·상태는 백엔드에서 처리
  if (filterType    !== "전체") displayed = displayed.filter(i => i.type === filterType);
  if (filterDateFrom) displayed = displayed.filter(i => (i.date || "") >= filterDateFrom);
  if (filterDateTo)   displayed = displayed.filter(i => (i.date || "") <= filterDateTo);
  if (sortKey) {
    displayed = [...displayed].sort((a, b) => {
      const va = sortKey === "time" ? a.time : a.confidence;
      const vb = sortKey === "time" ? b.time : b.confidence;
      return sortDir === "asc" ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
    });
  }

  function approve(id) { requestAction(id, "approve"); }
  function reject(id)  { requestAction(id, "reject");  }

  return (
    <div style={s.root}>
      {/* 확인 팝업 */}
      {confirm && (
        <div style={s.overlay}>
          <div style={s.popup}>
            <div style={s.popupTitle}>
              {confirm.action === "approve" ? "✓ 승인" : "✗ 반려"} 확인
            </div>
            <div style={s.popupMsg}>
              정말 <span style={{ color: confirm.action === "approve" ? "#22c55e" : "#ef4444", fontWeight: 700 }}>
                {confirm.action === "approve" ? "승인" : "반려"}
              </span> 하시겠습니까?
            </div>
            <label style={s.popupCheck}>
              <input
                type="checkbox"
                checked={confirm.noShow}
                onChange={e => setConfirm(c => ({ ...c, noShow: e.target.checked }))}
                style={{ marginRight: 6, accentColor: "#38bdf8" }}
              />
              <span style={{ fontSize: 11, color: "#64748b" }}>다시 표시하지 않음</span>
            </label>
            <div style={s.popupBtns}>
              <button style={s.popupCancel} onClick={() => setConfirm(null)}>취소</button>
              <button
                style={confirm.action === "approve" ? s.popupApprove : s.popupReject}
                onClick={handleConfirm}
              >
                {confirm.action === "approve" ? "✓ 승인" : "✗ 반려"}
              </button>
            </div>
          </div>
        </div>
      )}
      {/* 상단 탭 + 버튼 */}
      <div style={s.topBar}>
        <div style={s.tabs}>
          {TABS.map(tab => {
            const count =
              tab === "전체"    ? stats.total    :
              tab === "검수 대기" ? stats.pending  :
              tab === "승인"    ? stats.approved :
              tab === "반려"    ? stats.rejected : 0;
            const active = tab === activeTab;
            const tabColor =
              tab === "반려"   ? "#ef4444" :
              tab === "승인"   ? "#22c55e" :
              tab === "검수 대기" ? "#f59e0b" :
              "#38bdf8";
            return (
              <button
                key={tab}
                style={{
                  ...s.tab,
                  ...(active ? s.tabActive : {}),
                  color: active ? tabColor : "#94a3b8",
                }}
                onClick={() => { setActiveTab(tab); setOffset(0); setFilterProcess("전체"); setFilterType("전체"); setFilterStatus("전체"); }}
              >
                {tab} <span style={{
                  ...s.tabBadge,
                  background: active ? tabColor + "22" : "#1e293b",
                  color: active ? tabColor : "#64748b",
                }}>{count}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 메인 레이아웃: 테이블 + 미리보기 */}
      <div style={s.mainLayout}>
        {/* 왼쪽: 검수 대기열 테이블 */}
        <div style={s.tablePanel}>
          <div style={s.tablePanelHeader}>
            <span style={s.panelTitle}>검수 대기열</span>
          </div>

          <div style={{ flex: 1, overflowY: "auto", minHeight: 0, overflowX: "hidden", paddingBottom: 4 }}>
          <table style={s.table}>
            <thead>
              <tr>
                <th style={{ ...s.th, width: 180 }}>제품명</th>

                {/* 공정: PR/SD/전체 드롭다운 */}
                <th style={{ ...s.th }}>
                  <select
                    style={s.thSelect}
                    value={filterProcess}
                    onChange={e => { setFilterProcess(e.target.value); setOffset(0); }}
                  >
                    {["전체", "PR", "SD"].map(v => (
                      <option key={v} value={v}>{v === "전체" ? (filterProcess === "전체" ? "공정 ▾" : "전체") : v}</option>
                    ))}
                  </select>
                </th>

                {/* 유형: 유형별 드롭다운 */}
                <th style={{ ...s.th }}>
                  <select
                    style={s.thSelect}
                    value={filterType}
                    onChange={e => setFilterType(e.target.value)}
                  >
                    {allTypes.map(v => (
                      <option key={v} value={v}>{v === "전체" ? (filterType === "전체" ? "유형 ▾" : "전체") : v}</option>
                    ))}
                  </select>
                </th>

                <th style={{ ...s.th, cursor: "pointer" }} onClick={() => toggleSort("confidence")}>
                  신뢰도{sortIcon("confidence")}
                </th>

                <th style={{ ...s.th }}>
                  {activeTab === "전체" ? (
                    <select
                      style={s.thSelect}
                      value={filterStatus}
                      onChange={e => { setFilterStatus(e.target.value); setOffset(0); }}
                    >
                      {["전체", "대기", "승인", "반려"].map(v => (
                        <option key={v} value={v}>{v === "전체" ? (filterStatus === "전체" ? "상태 ▾" : "전체") : v}</option>
                      ))}
                    </select>
                  ) : "상태"}
                </th>
                <th style={{ ...s.th, borderRight: "none" }}>승인/반려</th>
              </tr>
            </thead>
            <tbody>
              {loading && (
                <tr><td colSpan={6} style={{ textAlign: "center", padding: 24, color: "#475569", fontSize: 12 }}>불러오는 중...</td></tr>
              )}
              {!loading && displayed.map((item, idx) => {
                const isSelected = selected?.id === item.id;
                return (
                  <tr
                    key={item.id}
                    style={{
                      ...s.tr,
                      background: isSelected ? "#1e3a5f" : idx % 2 === 0 ? "#0f172a" : "#111827",
                      cursor: "pointer",
                    }}
                    onClick={() => setSelected(item)}
                  >
                    <td style={{ ...s.td, color: "#38bdf8", fontWeight: 700, fontSize: 10, textAlign: "left", maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.file_name}</td>
                    <td style={s.td}>
                      <span style={{
                        ...s.badge,
                        background: item.process === "SD" ? "#1e3a5f" : "#1e293b",
                        color: item.process === "SD" ? "#38bdf8" : "#a78bfa",
                      }}>{item.process}</span>
                    </td>
                    <td style={{ ...s.td, color: "#e2e8f0" }}>{item.type}</td>
                    <td style={{ ...s.td, color: confidenceColor(item.confidence), fontWeight: 700 }}>
                      {item.confidence.toFixed(1)}%
                    </td>
                    <td style={{ ...s.td, color: STATUS_COLOR[item.status] }}>{item.status}</td>
                    <td style={{ ...s.td, borderRight: "none", textAlign: "center", verticalAlign: "middle" }}>
                      <div style={{ display: "inline-flex", gap: 6 }}>
                        {item.status_num !== 2 && (
                          <button
                            style={item.status_num === 1 ? s.approveDoneBtn : s.approveBtn}
                            onClick={e => { e.stopPropagation(); if (item.status_num === 0) approve(item.id); }}
                            title="승인"
                          >✓</button>
                        )}
                        {item.status_num !== 1 && (
                          <button
                            style={item.status_num === 2 ? s.rejectDoneBtn : s.rejectBtn}
                            onClick={e => { e.stopPropagation(); if (item.status_num === 0) reject(item.id); }}
                            title="반려"
                          >✗</button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0 10px", flexShrink: 0, borderTop: "1px solid #334155", position: "relative" }}>
            <div />
            {/* 페이지네이션 — 가운데 absolute 배치 */}
            {pageSize < 9999 && (() => {
              const totalPages = Math.max(1, Math.ceil(total / pageSize));
              const currentPage = Math.floor(offset / pageSize) + 1;

              const pages = [];
              // 항상 첫 페이지
              if (currentPage > 3) pages.push(1, "...");
              // 현재 페이지 앞뒤 2개
              for (let i = Math.max(1, currentPage - 2); i <= Math.min(totalPages, currentPage + 2); i++) {
                pages.push(i);
              }
              // 항상 마지막 페이지
              if (currentPage < totalPages - 2) pages.push("...", totalPages);

              return (
                <div style={{ position: "absolute", left: "50%", transform: "translateX(-50%)", display: "flex", alignItems: "center", gap: 3 }}>
                  <button
                    style={{ ...s.pageBtn, opacity: currentPage === 1 ? 0.3 : 1 }}
                    disabled={currentPage === 1}
                    onClick={() => setOffset((currentPage - 2) * pageSize)}
                  >‹</button>
                  {pages.map((p, i) =>
                    p === "..." ? (
                      <span key={`ellipsis-${i}`} style={{ color: "#475569", fontSize: 11, padding: "0 2px" }}>…</span>
                    ) : (
                      <button
                        key={p}
                        style={{ ...s.pageBtn, ...(p === currentPage ? s.pageBtnActive : {}) }}
                        onClick={() => setOffset((p - 1) * pageSize)}
                      >{p}</button>
                    )
                  )}
                  <button
                    style={{ ...s.pageBtn, opacity: currentPage === totalPages ? 0.3 : 1 }}
                    disabled={currentPage === totalPages}
                    onClick={() => setOffset(currentPage * pageSize)}
                  >›</button>
                </div>
              );
            })()}
            <select
              style={s.pageSizeSelect}
              value={pageSize}
              onChange={e => { setPageSize(Number(e.target.value)); setOffset(0); }}
            >
              {[25, 50, 100].map(n => (
                <option key={n} value={n}>{n}건 표시</option>
              ))}
              <option value={9999}>전체 표시</option>
            </select>
          </div>
        </div>

        {/* 오른쪽: 상세 미리보기 */}
        <div style={s.previewPanel}>
          <div style={s.panelTitle}>상세 미리보기</div>
          {selected ? (
            <div style={s.detailWrap}>
              {/* 이미지 영역 */}
              <div style={s.previewImgBox}>
                <img
                  src={selected.image || null}
                  alt={selected.id}
                  style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 8 }}
                  onError={e => { e.target.style.display = "none"; e.target.nextSibling.style.display = "flex"; }}
                />
                <div style={{ ...s.previewImgPlaceholder, display: "none" }}>
                  <span style={{ fontSize: 32, color: "#334155" }}>🖼</span>
                  <span style={{ fontSize: 11, color: "#475569", marginTop: 6 }}>이미지 없음</span>
                </div>
              </div>
              {/* 항목 정보 */}
              <div style={s.detailGrid}>
                {[
                  ["제품명", selected.file_name],
                  ["공정",  selected.process],
                  ["유형",  selected.type],
                  ["신뢰도", `${selected.confidence.toFixed(1)}%`],
                  ["상태",  selected.status],
                ].map(([label, value]) => (
                  <div key={label} style={s.detailRow}>
                    <span style={s.detailLabel}>{label}</span>
                    <span style={{
                      ...s.detailValue,
                      color:
                        label === "신뢰도" ? confidenceColor(selected.confidence) :
                        label === "상태"   ? STATUS_COLOR[selected.status]       :
                        label === "공정"   ? (selected.process === "SD" ? "#38bdf8" : "#a78bfa") :
                        "#e2e8f0",
                    }}>{value}</span>
                  </div>
                ))}
              </div>
              {/* 액션 버튼 */}
              <div style={s.detailActions}>
                {selected.status_num !== 2 && (
                  <button
                    style={selected.status_num === 1 ? s.detailApproveDoneBtn : s.detailApproveBtn}
                    onClick={() => { if (selected.status_num === 0) approve(selected.id); }}
                  >✓ 승인</button>
                )}
                {selected.status_num !== 1 && (
                  <button
                    style={selected.status_num === 2 ? s.detailRejectDoneBtn : s.detailRejectBtn}
                    onClick={() => { if (selected.status_num === 0) reject(selected.id); }}
                  >✗ 반려</button>
                )}
              </div>
            </div>
          ) : (
            <div style={s.previewEmpty}>
              <span style={{ fontSize: 36, color: "#334155" }}>🏷</span>
              <span style={{ fontSize: 12, color: "#475569", marginTop: 10 }}>항목을 선택하여 미리보기</span>
            </div>
          )}
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
    padding: "12px 16px",
    gap: 12,
  },

  topBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    flexShrink: 0,
  },
  tabs: { display: "flex", gap: 4 },
  tab: {
    background: "transparent",
    border: "none",
    fontSize: 13,
    fontWeight: 600,
    padding: "6px 10px",
    borderRadius: 6,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  tabActive: { background: "#1e293b" },
  tabBadge: {
    fontSize: 11,
    fontWeight: 700,
    padding: "1px 7px",
    borderRadius: 10,
  },
  mainLayout: {
    display: "flex",
    gap: 12,
    flex: 1,
    minHeight: 0,
  },

  tablePanel: {
    flex: "0 0 55%",
    background: "#1e293b",
    borderRadius: 10,
    padding: "12px 14px 0 14px",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "hidden",
  },
  tablePanelHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  panelTitle: {
    fontSize: 13,
    fontWeight: 700,
    color: "#ffffff",
  },
  pageSizeSelect: {
    background: "#0f172a",
    border: "1px solid #334155",
    color: "#94a3b8",
    fontSize: 11,
    padding: "3px 6px",
    borderRadius: 5,
  },
  pageBtn: {
    background: "#0f172a",
    border: "1px solid #334155",
    color: "#94a3b8",
    fontSize: 11,
    fontWeight: 600,
    width: 26,
    height: 24,
    borderRadius: 4,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  pageBtnActive: {
    background: "#1e3a5f",
    border: "1px solid #38bdf8",
    color: "#38bdf8",
  },
  dateInput: {
    background: "#0f172a",
    border: "1px solid #334155",
    color: "#94a3b8",
    fontSize: 11,
    padding: "3px 6px",
    borderRadius: 5,
    outline: "none",
    colorScheme: "dark",
  },
  searchBtn: {
    background: "#1e3a5f",
    border: "1px solid #38bdf8",
    color: "#38bdf8",
    fontSize: 11,
    fontWeight: 700,
    padding: "3px 12px",
    borderRadius: 5,
    cursor: "pointer",
  },

  table: {
    width: "100%",
    borderCollapse: "collapse",
    fontSize: 12,
  },
  thSelect: {
    background: "transparent",
    border: "none",
    color: "#94a3b8",
    fontWeight: 600,
    fontSize: 11,
    cursor: "pointer",
    outline: "none",
    textAlign: "center",
    letterSpacing: 0.3,
  },
  th: {
    color: "#94a3b8",
    fontWeight: 600,
    fontSize: 11,
    padding: "8px 10px",
    borderBottom: "1px solid #334155",
    borderRight: "1px solid #334155",
    textAlign: "center",
    letterSpacing: 0.3,
    position: "sticky",
    top: 0,
    background: "#1e293b",
    zIndex: 1,
  },
  tr: {
    borderBottom: "1px solid #1e293b",
    transition: "background 0.15s",
  },
  td: {
    padding: "8px 14px",
    color: "#cbd5e1",
    fontSize: 12,
    borderRight: "1px solid #334155",
    textAlign: "center",
  },
  badge: {
    display: "inline-block",
    padding: "2px 8px",
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 700,
  },
  approveBtn: {
    background: "#14532d",
    border: "none",
    color: "#22c55e",
    fontWeight: 700,
    fontSize: 13,
    width: 26,
    height: 26,
    borderRadius: 5,
    cursor: "pointer",
  },
  rejectBtn: {
    background: "#450a0a",
    border: "none",
    color: "#ef4444",
    fontWeight: 700,
    fontSize: 13,
    width: 26,
    height: 26,
    borderRadius: 5,
    cursor: "pointer",
  },
  approveDoneBtn: {
    background: "#14532d",
    border: "2px solid #22c55e",
    color: "#22c55e",
    fontWeight: 700,
    fontSize: 13,
    width: 26,
    height: 26,
    borderRadius: 5,
    cursor: "default",
    opacity: 1,
  },
  rejectDoneBtn: {
    background: "#450a0a",
    border: "2px solid #ef4444",
    color: "#ef4444",
    fontWeight: 700,
    fontSize: 13,
    width: 26,
    height: 26,
    borderRadius: 5,
    cursor: "default",
    opacity: 1,
  },

  previewPanel: {
    flex: 1,
    background: "#1e293b",
    borderRadius: 10,
    padding: "12px 14px 0 14px",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  previewEmpty: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },

  detailWrap: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 10,
    marginTop: 10,
    overflowY: "auto",
  },
  previewImgBox: {
    background: "#0f172a",
    borderRadius: 8,
    flex: 1,
    minHeight: 0,
    overflow: "hidden",
  },
  previewImgPlaceholder: {
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    border: "2px dashed #334155",
    borderRadius: 8,
    boxSizing: "border-box",
  },
  detailGrid: {
    display: "flex",
    flexDirection: "column",
    gap: 0,
    background: "#0f172a",
    borderRadius: 8,
    overflow: "hidden",
  },
  detailRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "8px 12px",
    borderBottom: "1px solid #1e293b",
  },
  detailLabel: {
    color: "#64748b",
    fontSize: 11,
    fontWeight: 600,
  },
  detailValue: {
    fontSize: 12,
    fontWeight: 700,
  },
  detailActions: {
    display: "flex",
    gap: 8,
    marginTop: 4,
    flexShrink: 0,
  },
  detailApproveBtn: {
    flex: 1,
    background: "#14532d",
    border: "none",
    color: "#22c55e",
    fontWeight: 700,
    fontSize: 13,
    padding: "8px 0",
    borderRadius: 6,
    cursor: "pointer",
  },
  detailRejectBtn: {
    flex: 1,
    background: "#450a0a",
    border: "none",
    color: "#ef4444",
    fontWeight: 700,
    fontSize: 13,
    padding: "8px 0",
    borderRadius: 6,
    cursor: "pointer",
  },
  detailApproveDoneBtn: {
    flex: 1,
    background: "#14532d",
    border: "2px solid #22c55e",
    color: "#22c55e",
    fontWeight: 700,
    fontSize: 13,
    padding: "8px 0",
    borderRadius: 6,
    cursor: "default",
  },
  detailRejectDoneBtn: {
    flex: 1,
    background: "#450a0a",
    border: "2px solid #ef4444",
    color: "#ef4444",
    fontWeight: 700,
    fontSize: 13,
    padding: "8px 0",
    borderRadius: 6,
    cursor: "default",
  },

  // 확인 팝업
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.55)",
    zIndex: 1000,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  popup: {
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 12,
    padding: "22px 26px 18px",
    minWidth: 260,
    maxWidth: 320,
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    display: "flex",
    flexDirection: "column",
    gap: 14,
  },
  popupTitle: {
    fontSize: 14,
    fontWeight: 700,
    color: "#e2e8f0",
    borderBottom: "1px solid #334155",
    paddingBottom: 10,
  },
  popupMsg: {
    fontSize: 13,
    color: "#cbd5e1",
    lineHeight: 1.6,
  },
  popupCheck: {
    display: "flex",
    alignItems: "center",
    cursor: "pointer",
  },
  popupBtns: {
    display: "flex",
    gap: 8,
    marginTop: 2,
  },
  popupCancel: {
    flex: 1,
    background: "#0f172a",
    border: "1px solid #334155",
    color: "#94a3b8",
    fontSize: 12,
    fontWeight: 600,
    padding: "7px 0",
    borderRadius: 6,
    cursor: "pointer",
  },
  popupApprove: {
    flex: 1,
    background: "#14532d",
    border: "none",
    color: "#22c55e",
    fontSize: 12,
    fontWeight: 700,
    padding: "7px 0",
    borderRadius: 6,
    cursor: "pointer",
  },
  popupReject: {
    flex: 1,
    background: "#450a0a",
    border: "none",
    color: "#ef4444",
    fontSize: 12,
    fontWeight: 700,
    padding: "7px 0",
    borderRadius: 6,
    cursor: "pointer",
  },
};
