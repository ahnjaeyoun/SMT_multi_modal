# 실시간 SMT 공정 데이터를 프론트엔드에 제공하는 라우터
# /api/realtime/stats    : DB 기반 통계 반환
# /api/realtime/recent   : 최근 수신 데이터 일괄 반환
# /api/realtime/stream   : SSE(Server-Sent Events)로 Kafka 메시지를 실시간 스트리밍
import asyncio
import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from kafka_manager import realtime_buffer, subscribers, parse_excel_sensor
from database import get_db

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# prefix를 지정하면 이 라우터의 모든 엔드포인트 URL 앞에 /api/realtime이 자동으로 붙음
router = APIRouter(prefix="/api/realtime", tags=["realtime"])


@router.get("/stats")
async def get_stats():
    """obb_result 컬렉션 기반으로 상단 카드 통계 계산"""
    db = get_db()
    col = db["obb_results"]

    total  = await col.count_documents({})
    defect = await col.count_documents({"file_name": {"$regex": "_DEF_"}})
    normal = total - defect
    defect_rate = round(defect / total * 100, 2) if total > 0 else 0.0

    return {
        "total":        total,
        "defect":       defect,
        "normal":       normal,
        "avg_accuracy": 0.0,
        "defect_rate":  defect_rate,
        "per_minute":   0,
    }


@router.get("/thresholds")
async def get_thresholds():
    """threshold_results에서 PR/SD별 센서 임계값(mean) 반환"""
    db = get_db()
    doc = await db["threshold_results"].find_one({}, {"thresholds": 1, "_id": 0})
    if not doc:
        return {"pr": {}, "sd": {}}
    return doc.get("thresholds", {"pr": {}, "sd": {}})


@router.get("/inference/{file_name}")
async def get_inference(file_name: str):
    """file_name 기준으로 obb_results에서 불량 유형 조회.
    defect_types 필드가 없으면 predictions.detections[].class_name 에서 추출."""
    db = get_db()
    # 파일명 끝 쉼표 여부 양쪽 검색
    clean = file_name.rstrip(",")
    doc = await db["obb_results"].find_one(
        {"file_name": {"$in": [clean, clean + ","]}},
    )
    if not doc:
        return {"defect_types": [], "accuracy": None}

    # defect_types 필드가 있으면 그대로 사용
    if doc.get("defect_types"):
        return {"defect_types": doc["defect_types"], "accuracy": doc.get("accuracy")}

    # 없으면 predictions.detections 에서 class_name 추출 (중복 제거, 순서 유지)
    detections = doc.get("predictions", {}).get("detections", [])
    seen = []
    for d in detections:
        name = d.get("class_name")
        if name and name not in seen:
            seen.append(name)
    return {"defect_types": seen, "accuracy": doc.get("accuracy")}


@router.get("/sensor/{file_name}")
async def get_sensor(file_name: str):
    """file_name 기준으로 sensor_data 컬렉션에서 shap_result 조회"""
    db = get_db()
    doc = await db["sensor_data"].find_one(
        {"file_name": file_name.rstrip(",")},
        {"shap_result": 1, "_id": 0}
    )
    if not doc:
        return {"sensor_importance": {}}
    return {"sensor_importance": doc.get("shap_result", {}).get("sensor_importance", {})}


@router.get("/recent")
async def get_recent():
    """메모리 버퍼에 쌓인 최근 수신 데이터 반환 (최대 100건)"""
    return {"items": list(realtime_buffer)}


@router.get("/products")
async def get_products(limit: int = 100):
    """obb_results에서 NOR/DEF 모두 포함한 최근 제품 목록 반환"""
    db = get_db()
    col = db["obb_results"]
    cursor = col.find({}, {"file_name": 1, "category": 1, "cause": 1, "confidence": 1, "img_path": 1, "excel_file": 1, "_id": 0}).sort("_id", -1).limit(limit)
    docs = await cursor.to_list(limit)
    items = []
    for doc in docs:
        file_name = doc.get("file_name", "").rstrip(",")
        category  = (doc.get("category") or "").upper()
        parts     = file_name.upper().split("_")
        status    = parts[1] if len(parts) > 1 else "??"
        excel_b64 = doc.get("excel_file")
        items.append({
            "file_name":  file_name,
            "category":   category or (parts[0] if parts else "??"),
            "status":     status,
            "cause":      doc.get("cause"),
            "confidence": doc.get("confidence"),
            "img_path": f"/api/realtime/image/{file_name}" if doc.get("img_path") else "",
            "sensor_data": parse_excel_sensor(excel_b64) if excel_b64 else [],
        })
    return {"items": items}


@router.get("/image/{file_name}")
async def get_image(file_name: str):
    """file_name 기준으로 obb_results에서 img_path 조회 후 이미지 파일 반환"""
    db = get_db()
    clean = file_name.rstrip(",")
    doc = await db["obb_results"].find_one(
        {"file_name": {"$in": [clean, clean + ","]}},
        {"img_path": 1}
    )
    if not doc or not doc.get("img_path"):
        raise HTTPException(status_code=404, detail="이미지 경로 없음")

    img_path = Path(doc["img_path"].strip().rstrip(","))
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"파일 없음: {img_path}")

    return FileResponse(str(img_path), media_type="image/jpeg")


@router.get("/stream")
async def stream():
    """
    SSE(Server-Sent Events) 엔드포인트.
    프론트에서 EventSource('/api/realtime/stream') 로 연결하면
    Kafka 메시지가 들어올 때마다 실시간으로 데이터를 받습니다.
    """
    # 이 클라이언트 전용 Queue 생성
    # maxsize=50 : 프론트가 처리하지 못한 메시지가 50개를 초과하면 새 데이터를 버림
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)

    # 전역 구독자 목록에 이 클라이언트의 Queue를 등록
    # 이후 kafka_manager._broadcast()가 이 Queue에 데이터를 넣어줌
    subscribers.append(queue)

    async def event_generator():
        # SSE 규격: 각 이벤트는 "data: <내용>\n\n" 형식으로 전송해야 함
        # 프론트엔드 EventSource 객체가 이 형식을 자동으로 파싱
        try:
            while True:
                # Queue에 데이터가 들어올 때까지 비동기로 대기
                # 데이터가 없으면 이 줄에서 멈춰 있다가, Kafka 메시지가 오면 재개
                data = await queue.get()

                # ensure_ascii=False : 한글 불량 유형명이 유니코드 이스케이프 없이 그대로 전송됨
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # 클라이언트가 브라우저 탭을 닫거나 서버가 종료될 때 발생
            # 별도 처리 없이 generator 함수를 정상 종료
            pass
        finally:
            # 연결이 끊긴 클라이언트의 Queue를 구독 목록에서 제거해 메모리 누수 방지
            if queue in subscribers:
                subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        # SSE 표준 Content-Type — 브라우저가 이 응답을 EventSource로 인식
        media_type="text/event-stream",
        headers={
            # 중간 캐시 서버가 SSE 응답을 저장하지 않도록 설정
            "Cache-Control": "no-cache",
            # nginx 리버스 프록시 환경에서 응답을 모아서 보내지 않고 즉시 전달하도록 설정
            # 이 헤더가 없으면 nginx가 데이터를 버퍼링해 실시간성이 깨질 수 있음
            "X-Accel-Buffering": "no",
        },
    )
