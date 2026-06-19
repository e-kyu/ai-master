import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File

from utils import fileUtils

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])


# 공고문/첨부파일 업로드 (React 프런트에서 호출, 브라우저는 서버 디스크에 직접 쓸 수 없으므로 경유)
@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    today_str = datetime.now().strftime("%Y%m%d")
    directory = f"./uploadFile/{today_str}/"

    # 원본 파일명에서 경로 요소를 제거하고, 충돌 방지를 위해 접두사를 붙인다.
    safe_name = os.path.basename(file.filename or "upload")
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"

    content = await file.read()
    file_full_path = fileUtils.save_uploaded_file(directory, stored_name, content)

    return {"fileFullPath": file_full_path, "fileName": safe_name}
