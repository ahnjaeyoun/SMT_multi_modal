import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx'],
  },
  server: {
    // 프론트엔드 서버 포트 설정 (기본값: 5173)
    // 포트를 바꾸면 backend/main.py의 allow_origins도 같은 포트로 반드시 변경해야 함
    // 예) port: 3000 으로 바꾸면 main.py에서 "http://localhost:3000" 으로 수정
    port: 5173,
  }
})
