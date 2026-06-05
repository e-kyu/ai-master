from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# DTO 클래스 정의
class ConvrstnDetailsBase(BaseModel):
    convrstn_id: str
    created_at: datetime

class ConvrstnDetailsCreate(ConvrstnDetailsBase):
    pass


class ConvrstnDetailsSchema(ConvrstnDetailsBase):
    question: str
    answer: Optional[str] = None
    answer_at: Optional[datetime] = None

    class Config:
        from_attributes = True




# DTO 클래스 정의
class ConvrstnBase(BaseModel):
    convrstn_id: str
    created_at: datetime


class ConvrstnCreate(ConvrstnBase):
    pass


class ConvrstnSchema(ConvrstnBase):
    topic: Optional[str] = None
    context: Optional[str] = None
    changed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ConvrstnListSchema(ConvrstnBase):
    topic: Optional[str] = None
    context: Optional[str] = None
    changed_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    mode: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

    class Config:
        from_attributes = True
        



# DTO 클래스 정의
class AgentBase(BaseModel):
    agent_id: str
    created_at: datetime


class AgentCreate(AgentBase):
    pass


class AgentSchema(AgentBase):
    access_level: str
    mode: str
    name: str
    resource: Optional[str] = None
    description: Optional[str] = None
    persona_prompt: Optional[str] = None
    question_division_prompt: Optional[str] = None
    final_prompt: Optional[str] = None
    agent_id: Optional[str] = None

    class Config:
        from_attributes = True