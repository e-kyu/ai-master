
import os
import logging
from datetime import datetime
def get_logger(log_dir="./log",log_filename_prefix="default_log"):

    # 1. 로그 디렉토리가 없으면 생성 (exist_ok=True는 폴더가 이미 있어도 에러를 내지 않음)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 2. 파일명 및 경로 설정
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(log_dir, f"{log_filename_prefix}_{today}.log")

    # 4. 로깅 설정 (터미널 출력 포함)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler() # 터미널 출력 추가
        ]
    )

    return logging.getLogger(__name__)
    