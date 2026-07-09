const ITEMS = [
  { id: "1", file_name: "PR_DEF_MF_A_20250902-183838_01658", process: "PR", type: "납부족", confidence: 91.2, status: "대기", status_num: 0, image: null },
  { id: "2", file_name: "SD_DEF_MF_A_20250902-183942_01661", process: "SD", type: "냉납",   confidence: 87.4, status: "대기", status_num: 0, image: null },
  { id: "3", file_name: "PR_DEF_MF_A_20250902-184005_01662", process: "PR", type: "밀림",   confidence: 76.1, status: "승인", status_num: 1, image: null },
  { id: "4", file_name: "SD_DEF_MF_A_20250902-184114_01665", process: "SD", type: "납볼",   confidence: 82.3, status: "반려", status_num: 2, image: null },
  { id: "5", file_name: "PR_DEF_MF_A_20250902-184210_01666", process: "PR", type: "미납",   confidence: 94.5, status: "대기", status_num: 0, image: null },
  { id: "6", file_name: "SD_DEF_MF_A_20250902-184330_01667", process: "SD", type: "쇼트",   confidence: 88.0, status: "대기", status_num: 0, image: null },
  { id: "7", file_name: "PR_DEF_MF_A_20250902-184450_01668", process: "PR", type: "역삽",   confidence: 79.3, status: "승인", status_num: 1, image: null },
  { id: "8", file_name: "SD_DEF_MF_A_20250902-184610_01669", process: "SD", type: "브릿지", confidence: 95.1, status: "대기", status_num: 0, image: null },
];

let state = ITEMS.map(i => ({ ...i }));

export async function fetchErrorItems(status = null) {
  const statusMap = { 검수대기: "대기", 승인: "승인", 반려: "반려" };
  const filtered = status ? state.filter(i => i.status === (statusMap[status] ?? status)) : state;
  return { items: filtered, total: filtered.length };
}

export async function fetchErrorStats() {
  return {
    total:    state.length,
    pending:  state.filter(i => i.status === "대기").length,
    approved: state.filter(i => i.status === "승인").length,
    rejected: state.filter(i => i.status === "반려").length,
  };
}

export async function approveItem(id) {
  state = state.map(i => i.id === id ? { ...i, status: "승인", status_num: 1 } : i);
  return { success: true, id, status: "승인됨" };
}

export async function rejectItem(id) {
  state = state.map(i => i.id === id ? { ...i, status: "반려", status_num: 2 } : i);
  return { success: true, id, status: "반려됨" };
}
