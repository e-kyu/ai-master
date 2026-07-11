from sqlalchemy import and_, func, insert, select, update
from typing import List
from datetime import datetime

import uuid
import json
from db.database import SessionLocal
from db.models import Convrstn as model_conversation
from db.models import ConvrstnDetails as model_conversation_details
from db.models import Agent as model_agent
from db.schemas import ConvrstnSchema, ConvrstnCreate, ConvrstnDetailsSchema, ConvrstnDetailsCreate , ConvrstnListSchema
from utils import config

logger = config.get_logger("./log", "llm-server")


def create_conversation(conversation_id: str, dict_conversation: dict = {}):
    """
    새로운 대화 세션을 생성합니다. 이미 존재하는 ID인 경우 생성하지 않습니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()
    try:
        # 동일한 conversation_id가 존재하는지 확인 (중복 생성 방지)
        existing_item = db.query(model_conversation).filter(model_conversation.convrstn_id == conversation_id).first()
        if not existing_item:
            db.execute(
                insert(model_conversation)
                .values({"convrstn_id": conversation_id
                        , "topic": dict_conversation.get("topic", "")
                })
            )
            db.commit()

        # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
        inserted_item = db.query(model_conversation).filter(model_conversation.convrstn_id == conversation_id).first()

        if not inserted_item:
            inserted_item = None

        return inserted_item
    finally:
        db.close()





def update_conversation_context(conversation_id: str, context: dict):
    """
    대화의 진행 상황에 따라 요약된 맥락(JSON)과 주제를 업데이트합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()
    try:
        # 해당 대화 ID의 주제와 맥락 정보를 갱신
        db.execute(
            update(model_conversation)
            .where(model_conversation.convrstn_id == conversation_id)
            .values({"topic": context.get("topic", ""), "context": json.dumps(context, ensure_ascii=False), "changed_at": func.now()})
        )
        db.commit()

        # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
        updated_item = db.query(model_conversation).filter(model_conversation.convrstn_id == conversation_id).first()

        if not updated_item:
            updated_item = None

        return updated_item
    finally:
        db.close()




def create_conversation_question(conversation_id: str, dict_conversation_details: dict):
    """
    사용자의 질문과 시스템이 보강한 질문(enhanced_question)을 상세 내역 테이블에 저장합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()
    try:
        # 질문 관련 데이터 삽입
        conversation_details_id = str(uuid.uuid4())  # 고유한 agent_id 생성 (UUID 사용)
        db.execute(
            insert(model_conversation_details)
            .values({"convrstn_id": conversation_id
                     , "convrstn_details_id": conversation_details_id
                     , "context": dict_conversation_details.get("context", "")
                     , "question": dict_conversation_details.get("question", "")
                     , "enhanced_question": dict_conversation_details.get("enhanced_question", "")
            })
        )
        db.commit()
        logger.debug(f"[ConvrstnRepository] 질문 저장 완료 conversation_id={conversation_id} conversation_details_id={conversation_details_id}")

        # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
        inserted_item = db.query(model_conversation_details).filter(and_(model_conversation_details.convrstn_id == conversation_id, model_conversation_details.convrstn_details_id == conversation_details_id)).first()

        if not inserted_item:
            inserted_item = None

        return inserted_item
    finally:
        db.close()


def create_conversation_answer(conversation_id: str, dict_conversation_details: dict):
    """
    생성된 AI의 답변을 해당 대화 상세 내역에 업데이트합니다.
    """

    # 1. 실제 DB 세션 생성
    db = SessionLocal()
    try:
        # 가장 최근의 대화 상세 내역에 답변 내용과 답변 시간을 기록
        db.execute(
            update(model_conversation)
            .where(model_conversation.convrstn_id == conversation_id)
            .values({"agent_id": dict_conversation_details.get("agent_id", None)})
        )

        # 가장 최근의 대화 상세 내역에 답변 내용과 답변 시간을 기록
        db.execute(
            update(model_conversation_details)
            .where(and_(model_conversation_details.convrstn_id == conversation_id, model_conversation_details.convrstn_details_id == dict_conversation_details.get("convrstn_details_id")))
            .values({"answer": dict_conversation_details.get("answer", ""), "answer_at": func.now(), "agent_id": dict_conversation_details.get("agent_id", None)})
        )
        db.commit()
        logger.info(f"[ConvrstnRepository] 답변 저장 완료 conversation_id={conversation_id} conversation_details_id={dict_conversation_details.get('convrstn_details_id')}")

        # 2. 업데이트된 객체 찾아서 반환 (None 에러 방지)
        updated_item = db.query(model_conversation_details).filter(and_(model_conversation_details.convrstn_id == conversation_id, model_conversation_details.convrstn_details_id == dict_conversation_details.get("convrstn_details_id"))).first()

        if not updated_item:
            updated_item = None

        return updated_item
    finally:
        db.close()



def read_context_conversation(conversation_id: str)->str:
    """
    특정 대화 ID에 대한 대화 맥락을 조회합니다.
    """

    db = SessionLocal()
    try:
        # 대화 기본 정보에서 context 필드만 추출
        conversation_info = db.query(model_conversation).filter(model_conversation.convrstn_id == conversation_id).first()

        if conversation_info is None:
            return ""

        return conversation_info.context
    finally:
        db.close()


def read_recent_conversation(conversation_id: str, limit: int=6)->List[str]:
    """
    특정 대화 ID에 대해 최신순으로 정렬하여 상위 6개의 내역만 조회합니다.
    """

    db = SessionLocal()
    try:
        # 생성 시간 역순으로 정렬하여 제한된 개수만큼 가져옴
        results = db.query(model_conversation_details).filter(model_conversation_details.convrstn_id == conversation_id).order_by(model_conversation_details.created_at.desc()).limit(limit).all()

        return results
    finally:
        db.close()




def read_conversation_list(skip: int = 0, limit: int = 100) -> List[ConvrstnListSchema]:
    """
    전체 대화 목록을 최신순으로 조회합니다.
    """
    db = SessionLocal()
    try:
        # 정렬 기준: 변경일시(changed_at)가 있으면 우선, 없으면 생성일시(created_at) 기준
        sort_col = func.coalesce(model_conversation.changed_at, model_conversation.created_at)

        results = (
                db.query(
                    model_conversation.convrstn_id,
                    model_conversation.created_at,
                    model_conversation.topic,
                    model_conversation.context,
                    model_conversation.changed_at,
                    model_agent.agent_id,
                    model_agent.mode,
                    model_agent.name,
                    model_agent.description
                )
                # JOIN 조건에 access_level을 포함
                .outerjoin(
                    model_agent,
                    and_(
                        model_agent.agent_id == model_conversation.agent_id,
                        model_agent.access_level == '1'
                    )
                )
                .order_by(sort_col.desc())
                .offset(skip)
                .limit(limit)
                .all()
            )

        return results
    finally:
        db.close()


def read_conversation_details(conversation_id: str)-> List[ConvrstnDetailsSchema]:
    """
    특정 대화 ID의 상세 내역(질문/답변 리스트)을 조회합니다.
    """
    db = SessionLocal()
    try:
        conversation_details = db.query(model_conversation_details).filter(model_conversation_details.convrstn_id == conversation_id).order_by(model_conversation_details.created_at.asc()).all()
        return conversation_details
    finally:
        db.close()


def delete_conversation(conversation_id: str):
    """
    특정 대화와 그에 속한 모든 상세 내역을 삭제합니다.
    """
    db = SessionLocal()
    try:
        db.query(model_conversation_details).filter(model_conversation_details.convrstn_id == conversation_id).delete()
        db.query(model_conversation).filter(model_conversation.convrstn_id == conversation_id).delete()
        db.commit()
        logger.info(f"[ConvrstnRepository] 대화 삭제 완료 conversation_id={conversation_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"[ConvrstnRepository] 대화 삭제 실패 conversation_id={conversation_id}: {e}")
        return False
    finally:
        db.close()