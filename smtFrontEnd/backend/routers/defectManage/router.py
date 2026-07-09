from fastapi import APIRouter
from database import get_db

router = APIRouter(prefix="/api/defectmanage", tags=["defectmanage"])

COL = "obb_results"

# category_id 17~32 → 불량 유형명 매핑
CAUSE_NAMES = {
    17: "미납",       18: "납부족",     19: "납쇼트",    20: "납볼",
    21: "납좌표밀림", 22: "납형성불량", 23: "냉납",      24: "밀림",
    25: "쇼트",       26: "오삽",       27: "미삽",      28: "역삽",
    29: "뒤집힘",     30: "일어섬",     31: "납금감/핀홀", 32: "납고드름",
}


def _process_match(process) -> dict:
    """process 필터 → category 조건 변환"""
    if process in ("PR", "SD"):
        return {"category": process.lower()}
    return {}


@router.get("/summary")
async def get_defect_summary():
    """
    요약 카드 — 전체 누계
    반환: { total, pr, sd, defect_rate }
    """
    db = get_db()
    col = db[COL]

    def_filter = {"file_name": {"$regex": "_DEF_"}}

    pr    = await col.count_documents({**def_filter, "category": "pr"})
    sd    = await col.count_documents({**def_filter, "category": "sd"})
    total = pr + sd

    inspected   = await col.count_documents({})
    defect_rate = round(total / inspected * 100, 1) if inspected > 0 else 0.0

    return {"total": total, "pr": pr, "sd": sd, "defect_rate": defect_rate}


@router.get("/by-type")
async def get_defects_by_type(process: str = None):
    """
    불량 유형별 집계 — detections[0].class_name 기준
    반환: { items: [{ type, count }, ...] }
    """
    db = get_db()
    col = db[COL]

    match = {"file_name": {"$regex": "_DEF_"}, **_process_match(process)}

    pipeline = [
        {"$match": {**match, "cause": {"$ne": None}}},
        {"$group": {"_id": "$cause", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]

    docs = await col.aggregate(pipeline).to_list(32)
    return {"items": [
        {"type": CAUSE_NAMES.get(d["_id"], str(d["_id"])), "count": d["count"]}
        for d in docs if d["_id"] is not None
    ]}


@router.get("/trend")
async def get_defect_trend(period: str = "daily"):
    """
    불량 추이 — file_name에서 날짜 추출
    period: daily | monthly | quarterly | yearly
    반환: { trend: [{ date, PR, SD }, ...] }
    """
    db = get_db()
    col = db[COL]

    # $regexFind로 match 객체 추출 후 .match 필드에서 $substr (인자 3개)
    raw = {"$getField": {"field": "match", "input": {"$regexFind": {"input": "$file_name", "regex": r"\d{8}"}}}}

    if period == "daily":
        date_expr = {"$concat": [{"$substr": [raw, 4, 2]}, "-", {"$substr": [raw, 6, 2]}]}
    elif period == "monthly":
        date_expr = {"$concat": [{"$substr": [raw, 0, 4]}, "-", {"$substr": [raw, 4, 2]}]}
    elif period == "quarterly":
        date_expr = {"$substr": [raw, 0, 6]}  # yyyymm → 파이썬에서 분기 변환
    else:  # yearly
        date_expr = {"$substr": [raw, 0, 4]}

    pipeline = [
        {"$match": {"file_name": {"$regex": "_DEF_"}}},
        {
            "$group": {
                "_id": {
                    "date":    date_expr,
                    "process": "$category",
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.date": 1}},
    ]

    docs = await col.aggregate(pipeline).to_list(500)

    trend_map: dict = {}
    for doc in docs:
        raw_date = doc["_id"]["date"] or ""
        process  = (doc["_id"]["process"] or "").upper()

        if period == "quarterly" and len(raw_date) == 6:
            year  = raw_date[:4]
            month = int(raw_date[4:6])
            q     = (month - 1) // 3 + 1
            date  = f"{year} Q{q}"
        else:
            date = raw_date

        if not date:
            continue
        if date not in trend_map:
            trend_map[date] = {"date": date, "PR": 0, "SD": 0}
        if process in ("PR", "SD"):
            trend_map[date][process] += doc["count"]

    return {"trend": sorted(trend_map.values(), key=lambda x: x["date"])}


@router.get("/recent")
async def get_recent_defects(limit: int = 20, process: str = None):
    """
    최근 불량 목록
    반환: { items: [{ id, file_name, process, types, confidence }, ...] }
    """
    db = get_db()
    col = db[COL]

    query = {"file_name": {"$regex": "_DEF_"}, **_process_match(process)}

    cursor = col.find(query).sort("_id", -1).limit(limit)
    docs   = await cursor.to_list(limit)

    import re, math
    items = []
    for doc in docs:
        file_name = doc.get("file_name", "")

        # file_name에서 시간 추출: PR_DEF_MF_A_20250902-183838_01658 → "18:38:38"
        time_str = ""
        m = re.search(r"\d{8}-(\d{6})", file_name)
        if m:
            t = m.group(1)
            time_str = f"{t[0:2]}:{t[2:4]}:{t[4:6]}"

        raw_conf = doc.get("confidence", None)
        confidence = 0.0
        if raw_conf is not None:
            try:
                v = float(raw_conf)
                confidence = round(v * 100, 1) if not math.isnan(v) else 0.0
            except (TypeError, ValueError):
                confidence = 0.0

        cause = doc.get("cause", None)
        type_name = CAUSE_NAMES.get(cause, "-") if cause is not None else "-"

        items.append({
            "id":         str(doc.get("_id", "")),
            "file_name":  file_name,
            "time":       time_str,
            "process":    (doc.get("category", "")).upper(),
            "type":       type_name,
            "confidence": confidence,
        })

    return {"items": items}


@router.get("/cause")
async def get_cause_distribution(process: str = None):
    """
    불량 원인별 분포 — sensor_data 컬렉션의 shap_result.critical_sensor 집계
    query param process: "PR" | "SD" | None (전체)
    반환: { items: [{ name, value }, ...] }
    """
    db = get_db()
    col = db["sensor_data"]

    match: dict = {"shap_result.critical_sensor": {"$exists": True, "$ne": None}}
    if process in ("PR", "SD"):
        # file_name 앞 두 글자로 공정 구분: PR_NOR_... / SD_DEF_...
        match["file_name"] = {"$regex": f"^{process}_"}

    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$shap_result.critical_sensor", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]

    docs  = await col.aggregate(pipeline).to_list(10)
    total = sum(d["count"] for d in docs)

    SENSOR_KO = {
        "temperature": "온도",
        "humidity":    "습도",
        "vibration":   "진동",
        "acceleration":"가속도",
        "noise":       "소음",
    }

    return {
        "items": [
            {
                "name":  SENSOR_KO.get(d["_id"], d["_id"]),
                "value": round(d["count"] / total * 100, 1) if total > 0 else 0,
            }
            for d in docs if d["_id"]
        ]
    }
