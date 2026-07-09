# FastAPI 앱의 진입점 — 서버 시작/종료 시 DB 연결과 Kafka consumer를 관리하고, 라우터를 자동 등록
import os
import asyncio
import importlib
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from database import connect_db, close_db
from kafka_manager import kafka_consumer_task

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버가 시작될 때 실행되는 초기화 블록
    # MongoDB에 연결해 DB를 사용할 수 있는 상태로 만듦
    await connect_db()

    # Kafka consumer를 백그라운드 태스크로 실행
    # create_task는 비동기 함수를 이벤트 루프에서 병렬로 실행시키는 함수
    task = asyncio.create_task(kafka_consumer_task())

    # yield 이전은 서버 시작, yield 이후는 서버 종료 시점에 실행
    yield

    # 서버가 종료될 때 Kafka consumer 태스크를 취소 요청
    task.cancel()

    # 태스크가 실제로 종료될 때까지 기다림
    # return_exceptions=True : CancelledError가 발생해도 예외로 처리하지 않고 정상 종료
    await asyncio.gather(task, return_exceptions=True)

    # MongoDB 연결 해제
    await close_db()


# FastAPI 애플리케이션 인스턴스 생성
# lifespan을 지정해 서버 시작/종료 시 초기화·정리 로직이 자동으로 실행되도록 함
app = FastAPI(title="SMT Multimodal API", lifespan=lifespan)

# FRONTEND_URL 환경변수로 여러 주소 지정 가능 (쉼표로 구분)
# 예) FRONTEND_URL=http://localhost:5173,http://myserver.com
_frontend_urls = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [u.strip() for u in _frontend_urls.split(",") if u.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers/ 하위 폴더를 자동 스캔해서 router.py 가 있으면 등록
# 새 기능을 추가할 때 routers/<폴더명>/router.py 만 만들면 됨 — main.py 수정 불필요
ROUTERS_DIR = Path(__file__).parent / "routers"
for folder in sorted(ROUTERS_DIR.iterdir()):
    # router.py 파일이 있는 폴더만 처리
    if folder.is_dir() and (folder / "router.py").exists():
        # "routers.폴더명.router" 형태의 모듈 경로를 동적으로 임포트
        module = importlib.import_module(f"routers.{folder.name}.router")
        app.include_router(module.router)
        print(f"라우터 등록: routers/{folder.name}/router.py")


# ★ 이미지 경로 변경 시 수정 위치 ★
# DB의 image_path 필드가 "/pr/bbox_image/파일명.jpg" 형식으로 들어옴
# 서버 배포 후 실제 파일이 저장된 절대경로로 아래 directory= 값을 변경해야 함
# 예) directory="/data/pr"  또는  directory="D:/smt/pr"
# 폴더가 없으면 마운트를 건너뜀 — 개발 단계에서 이미지 경로 미설정 시 서버가 뜨도록 예외 처리
if Path("/pr").exists():
    app.mount("/pr", StaticFiles(directory="/pr"), name="pr_images")
if Path("/sd").exists():
    app.mount("/sd", StaticFiles(directory="/sd"), name="sd_images")


@app.get("/")
async def root():
    # 서버 정상 동작 여부를 빠르게 확인하는 헬스체크 용도
    return {"message": "SMT API 서버 실행 중"}


if __name__ == "__main__":
    import uvicorn
    # 개발 모드로 직접 실행 시 사용 — reload=True 로 코드 변경 시 자동 재시작
    # 운영 환경에서는 uvicorn main:app --host 0.0.0.0 --port 8000 으로 직접 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
