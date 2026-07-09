# pydantic: 데이터 유효성 검사 라이브러리. 요청/응답 데이터의 타입을 자동으로 검사해 줍니다.
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# API로 센서 데이터를 주고받을 때 사용하는 데이터 형식(스키마)을 정의합니다.
# 클라이언트가 보낸 JSON이 이 형식과 맞지 않으면 FastAPI가 자동으로 오류를 반환합니다.
class SensorData(BaseModel):
    timestamp: datetime      # 데이터 수집 시각 (ISO 8601 형식, 예: "2024-01-01T12:00:00")
    temperature: float       # 온도 (°C)
    humidity: float          # 습도 (%)
    vibration: float         # 진동 (m/s²)
    acceleration: float      # 가속도 (m/s²)
    noise: float             # 소음 (dB)
    process: str             # 공정 구분: "PR"(리플로 후 검사) 또는 "SD"(납땜 공정)
    status: str              # 양불 구분: "NOR"(정상) 또는 "DEF"(불량)


# MongoDB에 저장된 문서를 읽어올 때 사용하는 스키마입니다.
# SensorData를 그대로 상속하고, MongoDB가 자동 생성하는 "_id" 필드만 추가합니다.
class SensorDataInDB(SensorData):
    # MongoDB의 "_id" 필드를 Python에서 "id"라는 이름으로 사용합니다.
    # alias="_id" 덕분에 DB 문서의 "_id" 값이 이 필드에 자동으로 매핑됩니다.
    id: Optional[str] = Field(None, alias="_id")

    class Config:
        # alias(_id)와 실제 필드명(id) 양쪽 모두로 값을 채울 수 있게 허용합니다.
        populate_by_name = True
