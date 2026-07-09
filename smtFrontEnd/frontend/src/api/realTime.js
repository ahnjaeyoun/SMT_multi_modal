import { BASE_URL } from "../config";
import * as mock from "./mock/realTime.js";

const USE_MOCK = import.meta.env.VITE_MOCK === "true";

export async function fetchDashboardStats() {
  if (USE_MOCK) return mock.fetchDashboardStats();
  const res = await fetch(`${BASE_URL}/api/realtime/stats`);
  if (!res.ok) throw new Error("stats fetch 실패");
  return res.json();
}

export async function fetchRecentItems() {
  if (USE_MOCK) return { items: [] };
  const res = await fetch(`${BASE_URL}/api/realtime/recent`);
  if (!res.ok) throw new Error("recent fetch 실패");
  return res.json();
}

export async function fetchProducts(limit = 100) {
  if (USE_MOCK) return mock.fetchProducts(limit);
  const res = await fetch(`${BASE_URL}/api/realtime/products?limit=${limit}`);
  if (!res.ok) throw new Error("products fetch 실패");
  return res.json();
}

export async function fetchThresholds() {
  if (USE_MOCK) return mock.fetchThresholds();
  const res = await fetch(`${BASE_URL}/api/realtime/thresholds`);
  if (!res.ok) throw new Error("thresholds fetch 실패");
  return res.json();
}

export async function fetchInferenceResult(fileName) {
  if (USE_MOCK) return mock.fetchInferenceResult(fileName);
  const res = await fetch(`${BASE_URL}/api/realtime/inference/${encodeURIComponent(fileName)}`);
  if (!res.ok) throw new Error("inference fetch 실패");
  return res.json();
}

export async function fetchSensorImportance(fileName) {
  if (USE_MOCK) return mock.fetchSensorImportance(fileName);
  const res = await fetch(`${BASE_URL}/api/realtime/sensor/${encodeURIComponent(fileName)}`);
  if (!res.ok) throw new Error("sensor fetch 실패");
  return res.json();
}

export function subscribeRealtimeStream(onMessage) {
  if (USE_MOCK) return mock.subscribeRealtimeStream(onMessage);
  const es = new EventSource(`${BASE_URL}/api/realtime/stream`);
  es.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* JSON 파싱 실패 시 무시 */ }
  };
  return es;
}
