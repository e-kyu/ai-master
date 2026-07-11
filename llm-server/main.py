import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import conversation, conversation_history, agent, upload, ai_tools, notice_scan

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

# React 개발 서버(Vite)에서의 cross-origin 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# router 추가
app.include_router(conversation.router)
app.include_router(conversation_history.router)
app.include_router(agent.router)
app.include_router(upload.router)
app.include_router(ai_tools.router)
app.include_router(notice_scan.router)

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