import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from agent.notice_scan_agent import PureLangNoticeScanAgent
from utils import config, file_utils

logger = config.get_logger("./log", "llm-server")

router = APIRouter(prefix="/api/v1/notice-scan", tags=["notice-scan"])


# 공고서 첨부파일을 전달받아 저장 후, NoticeScanAgent로 분석하여 결과를 반환
@router.post("/")
async def scan_notice_file(file: UploadFile = File(...)):
    today_str = datetime.now().strftime("%Y%m%d")
    directory = f"./uploadFile/{today_str}/"

    # 원본 파일명에서 경로 요소를 제거하고, 충돌 방지를 위해 접두사를 붙인다.
    safe_name = os.path.basename(file.filename or "upload")
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"

    content = await file.read()
    file_full_path = file_utils.save_uploaded_file(directory, stored_name, content)

    logger.info(f"[NoticeScanRouter] 공고서 스캔 요청 수신 file={file_full_path}")

    agent = PureLangNoticeScanAgent()
    final_state = await agent.run(file_full_path)

    if final_state.get("status") != "success":
        error_message = final_state.get("error_message") or "공고서 분석에 실패했습니다."
        logger.error(f"[NoticeScanRouter] 공고서 스캔 실패 file={file_full_path} reason={error_message}")
        raise HTTPException(status_code=422, detail=error_message)

    logger.info(f"[NoticeScanRouter] 공고서 스캔 완료 file={file_full_path}")

    return {
        "fileFullPath": file_full_path,
        "fileName": safe_name,
        "extractedData": final_state.get("extracted_data"),
        "isViolating": final_state.get("is_violating"),
    }
