from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db
from db.schemas import ConvrstnSchema, ConvrstnDetailsSchema, ConvrstnListSchema

from repository import conversation_repository

router = APIRouter(prefix="/api/v1", tags=["conversation_history"])

# 대화목록 조회 (URL 경로는 프론트엔드와의 API 계약 유지를 위해 변경하지 않음)
@router.get("/convrstnHistory/", response_model=List[ConvrstnListSchema])
def read_conversation_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = conversation_repository.read_conversation_list(skip=skip, limit=limit)
    if not results:
        return []
    return results

# 대화내역 조회
@router.get("/convrstnHistory/{conversation_id}", response_model=List[ConvrstnDetailsSchema])
def read_conversation_details(conversation_id: str, db: Session = Depends(get_db)):
    details = conversation_repository.read_conversation_details(conversation_id)
    if not details:
        raise HTTPException(status_code=404, detail="Conversation details not found")
    return details

# 대화 삭제
@router.delete("/convrstnHistory/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    success = conversation_repository.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
    return {"message": "Successfully deleted"}