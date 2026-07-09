import { BASE_URL } from "../config";
import * as mock from "./mock/errorCheck.js";

const USE_MOCK = import.meta.env.VITE_MOCK === "true";

export async function fetchErrorItems(status = null, limit = 50) {
  if (USE_MOCK) return mock.fetchErrorItems(status);
  const params = new URLSearchParams({ limit });
  if (status) params.append("status", status);
  const res = await fetch(`${BASE_URL}/api/errorcheck/items?${params}`);
  if (!res.ok) throw new Error("items fetch 실패");
  return res.json();
}

export async function fetchErrorStats() {
  if (USE_MOCK) return mock.fetchErrorStats();
  const res = await fetch(`${BASE_URL}/api/errorcheck/stats`);
  if (!res.ok) throw new Error("stats fetch 실패");
  return res.json();
}

export async function approveItem(id) {
  if (USE_MOCK) return mock.approveItem(id);
  const res = await fetch(`${BASE_URL}/api/errorcheck/items/${id}/approve`, { method: "PATCH" });
  if (!res.ok) throw new Error("approve 실패");
  return res.json();
}

export async function rejectItem(id) {
  if (USE_MOCK) return mock.rejectItem(id);
  const res = await fetch(`${BASE_URL}/api/errorcheck/items/${id}/reject`, { method: "PATCH" });
  if (!res.ok) throw new Error("reject 실패");
  return res.json();
}
