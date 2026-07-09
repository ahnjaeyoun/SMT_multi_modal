import os
import math
from datetime import datetime, timezone
from pathlib import Path
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from database import get_db

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

router = APIRouter(prefix="/api/errorcheck", tags=["errorcheck"])

OBB_COL   = "obb_results"
CHECK_COL = "err_chk"

CAUSE_MAP = {
    17: "미납",
    18: "납부족",
    19: "납쇼트",
    20: "납볼",
    21: "납좌표밀림",
    22: "납형성불량",
    23: "냉납",
    24: "밀림",
    25: "쇼트",
    26: "오삽",
    27: "미삽",
    28: "역삽",
    29: "뒤집힘",
    30: "일어섬",
    31: "납금감/핀홀",
    32: "납고드름",
}

STATUS_MAP = {0: "대기", 1: "승인", 2: "반려"}


def _format_doc(doc: dict) -> dict:
    """obb_results + err_chk 조인 결과 → 프론트 응답 형식 변환"""
    obb_id    = str(doc["_id"])
    file_name = doc.get("file_name", "")
    process   = (doc.get("category") or "").upper()

    cause_num = doc.get("cause")
    if cause_num is not None:
        type_str = CAUSE_MAP.get(int(cause_num), str(cause_num))
    else:
        type_str = "알 수 없음"

    raw_conf = doc.get("confidence", 0)
    try:
        conf_val = float(raw_conf)
        confidence = round(conf_val * 100, 1) if not math.isnan(conf_val) else 0.0
    except (TypeError, ValueError):
        confidence = 0.0

    image_url = f"{API_BASE_URL}/api/errorcheck/image/{file_name}" if file_name else ""

    review_list = doc.get("review", [])
    status_num  = review_list[0]["status"] if review_list else 0
    status_str  = STATUS_MAP.get(status_num, "대기")
    chk_id      = str(review_list[0]["_id"]) if review_list else None

    return {
        "id":         obb_id,
        "chk_id":     chk_id,
        "file_name":  file_name,
        "date":       "",
        "time":       "",
        "process":    process,
        "type":       type_str,
        "confidence": confidence,
        "status":     status_str,
        "status_num": status_num,
        "image":      image_url,
    }


@router.get("/items")
async def get_error_items(
    status: str = None,
    process: str = None,
    limit: int = 25,
    offset: int = 0,
):
    """검수 대기열 목록 반환 (obb_results + err_chk 조인)"""
    db  = get_db()
    obb = db[OBB_COL]

    query = {"confidence": {"$lte": 0.7}, "file_name": {"$regex": "_DEF_"}}
    if process and process in ("PR", "SD"):
        query["category"] = process.lower()

    pipeline = [{"$match": query}]
    pipeline.append({
        "$lookup": {
            "from":         CHECK_COL,
            "localField":   "_id",
            "foreignField": "obb_id",
            "as":           "review",
        }
    })

    if status == "대기":
        pipeline.append({"$match": {"$or": [
            {"review": {"$size": 0}},
            {"review.status": 0},
        ]}})
    elif status == "승인":
        pipeline.append({"$match": {"review.status": 1}})
    elif status == "반려":
        pipeline.append({"$match": {"review.status": 2}})

    count_result = await obb.aggregate(pipeline + [{"$count": "total"}]).to_list(1)
    total        = count_result[0]["total"] if count_result else 0

    docs = await obb.aggregate(pipeline + [{"$skip": offset}, {"$limit": limit}]).to_list(limit)

    return {"items": [_format_doc(d) for d in docs], "total": total}


@router.get("/stats")
async def get_errorcheck_stats():
    """탭 카운트용 통계 반환"""
    db  = get_db()
    obb = db[OBB_COL]
    chk = db[CHECK_COL]

    base_filter = {"confidence": {"$lte": 0.7}, "file_name": {"$regex": "_DEF_"}}
    obb_ids  = [doc["_id"] async for doc in obb.find(base_filter, {"_id": 1})]
    total    = len(obb_ids)
    approved = await chk.count_documents({"obb_id": {"$in": obb_ids}, "status": 1})
    rejected = await chk.count_documents({"obb_id": {"$in": obb_ids}, "status": 2})
    pending  = total - approved - rejected

    return {
        "total":    total,
        "pending":  max(pending, 0),
        "approved": approved,
        "rejected": rejected,
    }


@router.get("/image/{file_name}")
async def get_image(file_name: str):
    """file_name 기준으로 /bbox_image에서 이미지 파일 직접 반환"""
    img_path = Path(f"/app/bbox_image/{file_name}.jpg")
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"파일 없음: {img_path}")

    return FileResponse(str(img_path), media_type="image/jpeg")


@router.patch("/items/{obb_id}/approve")
async def approve_item(obb_id: str):
    """검수 항목 승인 — err_chk에 upsert"""
    db  = get_db()
    await db[CHECK_COL].update_one(
        {"obb_id": ObjectId(obb_id)},
        {"$set": {"status": 1, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"success": True, "id": obb_id, "status": "승인", "status_num": 1}


@router.patch("/items/{obb_id}/reject")
async def reject_item(obb_id: str):
    """검수 항목 반려 — err_chk에 upsert"""
    db  = get_db()
    await db[CHECK_COL].update_one(
        {"obb_id": ObjectId(obb_id)},
        {"$set": {"status": 2, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return {"success": True, "id": obb_id, "status": "반려", "status_num": 2}
