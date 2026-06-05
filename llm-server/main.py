import uvicorn
from fastapi import FastAPI

from routers import convrstn, convrstnHistory, agent

# 데이터베이스 초기화를 위한 임포트 추가
from db.database import Base, engine


# 데이터베이스 초기화
Base.metadata.create_all(bind=engine)

# FastAPI 인스턴스 생성
app = FastAPI(
    title="Debate Arena API",
    description="AI Debate Arena 서비스를 위한 API",
    version="0.1.0",
)

# router 추가
app.include_router(convrstn.router)
app.include_router(convrstnHistory.router)
app.include_router(agent.router)

# 실행은 server 경로에서
# . venv/bin/activate
# uvicorn main:app --port=8001

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9001,
        timeout_keep_alive=1800
        )