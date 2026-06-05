from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, ForeignKey, Sequence, UUID
from sqlalchemy.sql import func

from db.database import Base

# Agent 모델
class Agent(Base):
    """
    Agent 모델은 에이전트의 정보를 저장하는 테이블입니다. 각 에이전트는 고유한 ID, 접근 수준, 모드, 이름, 설명, 리소스 정보 및 생성 시간을 가집니다.
    """
    __tablename__ = "agent"

    agent_id = Column(UUID(as_uuid=False), primary_key=True)
    access_level = Column(String(50), default="1", nullable=False)
    mode = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    if_stream = Column(Boolean, default=False)
    resource = Column(Text, nullable=True)
    persona_prompt = Column(Text, nullable=True)
    question_division_prompt = Column(Text, nullable=True)
    final_prompt = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Conversation 모델
class Convrstn(Base):
    """
    Convrstn 모델은 대화의 기본 정보를 저장하는 테이블입니다. 각 대화는 고유한 ID, 생성 시간, 주제, 맥락 및 변경 시간을 가집니다.
    """
    __tablename__ = "convrstn"

    convrstn_id = Column(String(36), primary_key=True)
    agent_id = Column(UUID(as_uuid=False), nullable=True)
    topic = Column(Text, nullable=True)
    context = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    changed_at = Column(DateTime(timezone=True), nullable=True)

# Conversation Details 모델
class ConvrstnDetails(Base):
    """
    ConvrstnDetails 모델은 대화의 상세 정보를 저장하는 테이블입니다. 각 대화 상세 정보는 고유한 대화 ID, 생성 시간, 질문, 보강된 질문, 답변 및 답변 시간을 가집니다.
    """
    __tablename__ = "convrstn_details"

    ## TODO: llm-app에서 대화내역 insert하던 것은 llm-server에서 insert하도록 수정하고 대화의 맥락과 보강한 사용자질의문도 db에 입력하도록 수정필요
    convrstn_id = Column(String(36), primary_key=True)
    convrstn_details_id = Column(String(36), primary_key=True)
    agent_id = Column(UUID(as_uuid=False), nullable=True)
    question = Column(Text, nullable=False)
    enhanced_question = Column(Text, nullable=True)
    context = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)  # JSON 문자열로 저장
    answer_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())