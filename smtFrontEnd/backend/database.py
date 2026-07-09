import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 읽어옵니다
load_dotenv()

# 실제 값은 .env 파일에 명시되어 있습니다
# MONGODB_URL=mongodb://아이디:비밀번호@서버IP:포트/DB명
# DB_ID=아이디
# DB_PWD=비밀번호
# DB_NAME = DB명
MONGODB_URL = os.getenv("MONGODB_URL")
DB_ID = os.getenv("DB_ID")
DB_PWD = os.getenv("DB_PWD")
DB_NAME = os.getenv("DB_NAME")

# 앱 실행 중 전역으로 사용할 클라이언트·DB 객체 (초기값은 None, connect_db() 호출 후 채워짐)
client: AsyncIOMotorClient = None
db = None


async def connect_db():
    # FastAPI 앱 시작 시 호출 — MongoDB 클라이언트를 생성하고 실제 연결이 되는지 ping으로 확인합니다
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    await client.admin.command("ping")  # 연결 실패 시 여기서 예외가 발생해 서버 시작을 중단시킵니다
    print(f"MongoDB 연결 완료: {MONGODB_URL.split('@')[-1]} / DB: {DB_NAME} / DB_NAME")


async def close_db():
    # FastAPI 앱 종료 시 호출 — 열려 있는 클라이언트 연결을 안전하게 닫습니다
    global client
    if client:
        client.close()
        print("MongoDB 연결 종료")


def get_db():
    # 라우터·서비스에서 DB 객체가 필요할 때 이 함수를 통해 가져옵니다
    return db
