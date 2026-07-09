FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# docker-compose에서 build.args로 주입받은 값으로 빌드
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

COPY . .

RUN npm run build

# nginx로 빌드 결과물 서빙
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# React Router (SPA) 새로고침 대응 — 모든 경로를 index.html로 fallback
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
