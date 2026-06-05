from sqlalchemy import and_, func, insert, select, update
from typing import List
from datetime import datetime

import uuid
import json
from db.database import SessionLocal
from db.models import Convrstn as modelConvrstn
from db.models import ConvrstnDetails as modelConvrstnDetails
from db.models import Agent as modelAgent
from db.schemas import ConvrstnSchema, ConvrstnCreate, ConvrstnDetailsSchema, ConvrstnDetailsCreate , ConvrstnListSchema


def create_convrstn(convrstn_id: str, dict_convrstn: dict = {}):
    """
    새로운 대화 세션을 생성합니다. 이미 존재하는 ID인 경우 생성하지 않습니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()

    # 동일한 convrstn_id가 존재하는지 확인 (중복 생성 방지)
    existing_item = db.query(modelConvrstn).filter(modelConvrstn.convrstn_id == convrstn_id).first()
    if not existing_item:
        db.execute(
            insert(modelConvrstn)
            .values({"convrstn_id": convrstn_id
                    , "topic": dict_convrstn.get("topic", "")
            })
        )
        db.commit()
    
    # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
    inserted_item = db.query(modelConvrstn).filter(modelConvrstn.convrstn_id == convrstn_id).first()
    
    if not inserted_item:
        inserted_item = None
        
    return inserted_item





def update_convrstn_context(convrstn_id: str, context: dict):
    """
    대화의 진행 상황에 따라 요약된 맥락(JSON)과 주제를 업데이트합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()

    # 해당 대화 ID의 주제와 맥락 정보를 갱신
    db.execute(
        update(modelConvrstn)
        .where(modelConvrstn.convrstn_id == convrstn_id)
        .values({"topic": context.get("topic", ""), "context": json.dumps(context, ensure_ascii=False), "changed_at": func.now()})
    )
    db.commit()
    
    # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
    updated_item = db.query(modelConvrstn).filter(modelConvrstn.convrstn_id == convrstn_id).first()
    
    if not updated_item:
        updated_item = None
        
    return updated_item




def create_convrstn_question(convrstn_id: str, dict_convrstn_details: dict):
    """
    사용자의 질문과 시스템이 보강한 질문(enhanced_question)을 상세 내역 테이블에 저장합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()

    # 질문 관련 데이터 삽입
    convrstn_details_id = str(uuid.uuid4())  # 고유한 agent_id 생성 (UUID 사용)
    db.execute(
        insert(modelConvrstnDetails)
        .values({"convrstn_id": convrstn_id
                 , "convrstn_details_id": convrstn_details_id
                 , "context": dict_convrstn_details.get("context", "")
                 , "question": dict_convrstn_details.get("question", "")
                 , "enhanced_question": dict_convrstn_details.get("enhanced_question", "")
        })
    )
    db.commit()
    
    # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
    inserted_item = db.query(modelConvrstnDetails).filter(and_(modelConvrstnDetails.convrstn_id == convrstn_id, modelConvrstnDetails.convrstn_details_id == convrstn_details_id)).first()
    
    if not inserted_item:
        inserted_item = None
        
    return inserted_item


def create_convrstn_answer(convrstn_id: str, dict_convrstn_details: dict):
    """
    생성된 AI의 답변을 해당 대화 상세 내역에 업데이트합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()

    # 가장 최근의 대화 상세 내역에 답변 내용과 답변 시간을 기록
    db.execute(
        update(modelConvrstn)
        .where(modelConvrstn.convrstn_id == convrstn_id)
        .values({"agent_id": dict_convrstn_details.get("agent_id", None)})
    )

    # 가장 최근의 대화 상세 내역에 답변 내용과 답변 시간을 기록
    db.execute(
        update(modelConvrstnDetails)
        .where(and_(modelConvrstnDetails.convrstn_id == convrstn_id, modelConvrstnDetails.convrstn_details_id == dict_convrstn_details.get("convrstn_details_id")))
        .values({"answer": dict_convrstn_details.get("answer", ""), "answer_at": func.now(), "agent_id": dict_convrstn_details.get("agent_id", None)})
    )
    db.commit()
    
    # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
    updated_item = db.query(modelConvrstnDetails).filter(and_(modelConvrstnDetails.convrstn_id == convrstn_id, modelConvrstnDetails.convrstn_details_id == dict_convrstn_details.get("convrstn_details_id"))).first()
    
    if not updated_item:
        updated_item = None
        
    return updated_item



def read_context_convrstn(convrstn_id: str)->str:
    """
    특정 대화 ID에 대한 대화 맥락을 조회합니다.
    """
    
    db = SessionLocal()

    # 대화 기본 정보에서 context 필드만 추출
    convrstnInfo = db.query(modelConvrstn).filter(modelConvrstn.convrstn_id == convrstn_id).first()

    if convrstnInfo is None:
        return ""
    
    return convrstnInfo.context


def read_recent_convrstn(convrstn_id: str, limit: int=6)->List[str]:
    """
    특정 대화 ID에 대해 최신순으로 정렬하여 상위 6개의 내역만 조회합니다.
    """
    
    db = SessionLocal()
    # 생성 시간 역순으로 정렬하여 제한된 개수만큼 가져옴
    results = db.query(modelConvrstnDetails).filter(modelConvrstnDetails.convrstn_id == convrstn_id).order_by(modelConvrstnDetails.created_at.desc()).limit(limit).all()

    return results




def read_convrstn_list(skip: int = 0, limit: int = 100) -> List[ConvrstnListSchema]:
    """
    전체 대화 목록을 최신순으로 조회합니다.
    """
    db = SessionLocal()
    
    # 정렬 기준: 변경일시(changed_at)가 있으면 우선, 없으면 생성일시(created_at) 기준
    sort_col = func.coalesce(modelConvrstn.changed_at, modelConvrstn.created_at)

    results = (
            db.query(
                modelConvrstn.convrstn_id,
                modelConvrstn.created_at,
                modelConvrstn.topic,
                modelConvrstn.context,
                modelConvrstn.changed_at,
                modelAgent.agent_id,
                modelAgent.mode,
                modelAgent.name,
                modelAgent.description
            )
            # JOIN 조건에 access_level을 포함
            .outerjoin(
                modelAgent,
                and_(
                    modelAgent.agent_id == modelConvrstn.agent_id,
                    modelAgent.access_level == '1'
                )
            )
            .order_by(sort_col.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    return results


def read_convrstn_details(convrstn_id: str)-> List[ConvrstnDetailsSchema]:
    """
    특정 대화 ID의 상세 내역(질문/답변 리스트)을 조회합니다.
    """
    db = SessionLocal()
    convrstnDetails = db.query(modelConvrstnDetails).filter(modelConvrstnDetails.convrstn_id == convrstn_id).order_by(modelConvrstnDetails.created_at.asc()).all()
    return convrstnDetails


def delete_convrstn(convrstn_id: str):
    """
    특정 대화와 그에 속한 모든 상세 내역을 삭제합니다.
    """
    db = SessionLocal()
    try:
        db.query(modelConvrstnDetails).filter(modelConvrstnDetails.convrstn_id == convrstn_id).delete()
        db.query(modelConvrstn).filter(modelConvrstn.convrstn_id == convrstn_id).delete()
        db.commit()    
        return True
    except Exception as e:
        db.rollback()
        print(f"Error deleting conversation: {e}")
        return False
    finally:
        db.close()