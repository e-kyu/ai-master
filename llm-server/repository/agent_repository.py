import uuid
from sqlalchemy import cast, Integer
from typing import List
from sqlalchemy import insert, select, update, delete
from db.database import SessionLocal
from db.models import Agent as modelAgent
from db.schemas import AgentSchema



def create_agent(agent_id: str,  access_level: str, mode: str, name: str, description: str, resource: str, final_prompt: str):
    """
    새로운 에이전트 정보를 데이터베이스에 생성합니다.

    Args:
        agent_id (str): 에이전트 고유 식별자 (비어있을 경우 UUID 생성)
        access_level (str): 접근 권한 레벨
        mode (str): 에이전트 작동 모드
        name (str): 에이전트 이름
        description (str): 에이전트 설명
        resource (str): 관련 리소스 정보
        final_prompt (str): 에이전트에게 적용될 최종 프롬프트

    Returns:
        modelAgent: 생성된 에이전트 객체
    """
    db = SessionLocal()
    existing_item = None

    if agent_id is None or agent_id.strip() == "":
        agent_id = str(uuid.uuid4())  # 고유한 agent_id 생성 (UUID 사용)
    else:
        # 동일한 agent_id가 존재하는지 확인 (중복 생성 방지)
        existing_item = db.query(modelAgent).filter(modelAgent.agent_id == agent_id).first()
    
    if not existing_item:
        db.execute(
            insert(modelAgent)
            .values({
                       "agent_id": agent_id
                     , "access_level": access_level
                     , "mode": mode
                     , "name": name
                     , "description": description
                     , "resource": resource
                     , "final_prompt": final_prompt
            })
        )
        db.commit()
    
    # 2. 생성된 객체 찾아서 반환 (None 에러 방지)
    inserted_item = db.query(modelAgent).filter(modelAgent.agent_id == agent_id).first()
    
    if not inserted_item:
        raise Exception("에이전트 생성에 실패하였습니다.")
        
    return inserted_item


def read_agents(access_level: str = 1)->List[AgentSchema]:
    """
    특정 접근 수준 이하의 모든 에이전트 목록을 조회합니다.

    Args:
        access_level (str): 조회 기준이 되는 최대 접근 레벨 (기본값: 1)

    Returns:
        List[AgentSchema]: 조회된 에이전트 스키마 리스트
    """
    db = SessionLocal()

    # 접근 수준에 맞는 에이전트 정보 조회
    agent_list = db.query(modelAgent).filter(cast(modelAgent.access_level, Integer) <= cast(access_level, Integer)).all()

    if agent_list is None:
        return []
    
    return agent_list

def read_agent(agent_id: str)->AgentSchema:
    """
    에이전트 ID를 기반으로 특정 에이전트의 상세 정보를 조회합니다.

    Args:
        agent_id (str): 조회할 에이전트의 고유 ID

    Returns:
        AgentSchema: 조회된 에이전트 정보 (없을 경우 None)
    """
    db = SessionLocal()

    # 특정 에이전트 정보 조회
    agent = db.query(modelAgent).filter(modelAgent.agent_id == agent_id).first()

    if agent is None:
        agent = None
    
    return agent



def update_agent(agent_id: str,  access_level: str, mode: str, name: str, description: str, resource: str, final_prompt: str):
    """
    기존 에이전트의 상세 정보를 업데이트합니다.

    Args:
        agent_id (str): 수정할 에이전트의 고유 ID
        access_level (str): 수정할 접근 권한 레벨
        mode (str): 수정할 작동 모드
        name (str): 수정할 이름
        description (str): 수정할 설명
        resource (str): 수정할 리소스 정보
        final_prompt (str): 수정할 최종 프롬프트

    Returns:
        modelAgent: 업데이트된 에이전트 객체
    """
    db = SessionLocal()

    # 해당 에이전트 ID의 정보를 갱신
    db.execute(
        update(modelAgent)
        .where(modelAgent.agent_id == agent_id)
        .values({"access_level": access_level
                    , "mode": mode
                    , "name": name
                    , "description": description
                    , "resource": resource
                    , "final_prompt": final_prompt
        })
    )
    db.commit()
    
    # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
    updated_item = db.query(modelAgent).filter(modelAgent.agent_id == agent_id).first()
    
    if not updated_item:
        raise Exception("에이전트 업데이트에 실패하였습니다.")
        
    return updated_item



def delete_agent(agent_id: str):
    """
    에이전트 ID를 기반으로 특정 에이전트 정보를 삭제합니다.

    Args:
        agent_id (str): 삭제할 에이전트의 고유 ID

    Returns:
        bool: 삭제 성공 여부
    """
    db = SessionLocal()

    # 해당 에이전트 ID의 정보를 삭제
    db.execute(
        delete(modelAgent)
        .where(modelAgent.agent_id == agent_id)
    )
    db.commit()
    
    return True