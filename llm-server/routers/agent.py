from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db
from db.schemas import AgentSchema

from repository import agent_repository

router = APIRouter(prefix="/api/v1/agents", tags=["agent"])


# Agent 추가
@router.post("/", response_model=AgentSchema)
def create_agent(agent: AgentSchema, db: Session = Depends(get_db)):
    return agent_repository.create_agent(agent, db)


# Agent목록 조회
@router.get("/", response_model=List[AgentSchema])
def read_agents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = agent_repository.read_agents(access_level=1)
    if not results:
        return []
    return results



# Agent 수정
@router.put("/{agent_id}", response_model=AgentSchema)
def update_agent(agent_id: str, agent: AgentSchema, db: Session = Depends(get_db)):
    return agent_repository.update_agent(agent_id, agent, db)




# Agent 삭제
@router.delete("/{agent_id}")
def delete_agent(agent_id: str, db: Session = Depends(get_db)):
    success = agent_repository.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete agent")
    return {"message": "Successfully deleted"}