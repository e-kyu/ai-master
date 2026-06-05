from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from db.database import get_db
from db.schemas import ConvrstnSchema, ConvrstnDetailsSchema, ConvrstnListSchema

from repository import convrstn_repository

router = APIRouter(prefix="/api/v1", tags=["convrstnHistory"])


# 대화목록 조회
@router.get("/convrstnHistory/", response_model=List[ConvrstnListSchema])
def read_convrstn_list(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    results = convrstn_repository.read_convrstn_list(skip=skip, limit=limit)
    if not results:
        return []
    return results


# 대화내역 조회
@router.get("/convrstnHistory/{convrstn_id}", response_model=List[ConvrstnDetailsSchema])
def read_convrstn_details(convrstn_id: str, db: Session = Depends(get_db)):
    details = convrstn_repository.read_convrstn_details(convrstn_id)
    if not details:
        raise HTTPException(status_code=404, detail="Conversation details not found")
    return details


# 대화 삭제
@router.delete("/convrstnHistory/{convrstn_id}")
def delete_convrstn(convrstn_id: str, db: Session = Depends(get_db)):
    success = convrstn_repository.delete_convrstn(convrstn_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
    return {"message": "Successfully deleted"}