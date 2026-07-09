// 백엔드 API 주소 — Docker 배포 시 .env.production의 VITE_API_URL로 주입
// 예) VITE_API_URL=http://100.70.106.105:8000
export const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
