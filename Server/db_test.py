# 이거는 다른컴퓨터에서 tailscale 연결하고 실행시키는 파일

from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')

#접속 대상: 현재 서버 PC의 Tailscale IP (100.70.106.105)
#계정 정보: woo / young
MONGO_URL = "mongodb://woo:young@100.70.106.105:27017"

#외부 네트워크 통신이므로 서버 응답 대기 시간을 약간 넉넉하게(5초) 줍니다.
client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db = client["ai_factory_db"]

async def check_remote_db_connection():
    try:
        print("🌐 외부 DB 서버 응답을 기다리는 중...")
        await client.admin.command('ping')
        print("✅ 성공: 외부에서 MongoDB 서버에 정상적으로 접속 및 인증되었습니다!")

        # 데이터 적재 테스트
        collection = db["test_collection"]
        test_data = {
            "source": "remote_db_test",
            "message": "데이터 적재 테스트입니다.",
            "timestamp": datetime.now(timezone.utc)
        }
        print("📝 데이터 적재를 시도합니다...")
        result = await collection.insert_one(test_data)
        print(f"✅ 데이터 적재 성공! (Inserted ID: {result.inserted_id})")

        # 적재된 데이터 조회 확인
        doc = await collection.find_one({"_id": result.inserted_id})
        print(f"🔍 DB에서 조회된 데이터: {doc}")

    except Exception as e:
        print(f"❌ 실패: 원격 DB 연결 또는 작업에 실패했습니다.\n상세 에러: {e}")

if __name__ == "__main__":
    asyncio.run(check_remote_db_connection())