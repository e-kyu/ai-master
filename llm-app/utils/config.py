import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 파일에서 환경 변수 로드
load_dotenv()

class Settings(BaseSettings):
    # Azure OpenAI 설정
    API_BASE_URL: str

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


# 설정 인스턴스 생성
settings = Settings()