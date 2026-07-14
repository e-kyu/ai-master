import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

import logging
from datetime import datetime


# .env 파일에서 환경 변수 로드
load_dotenv()

class Settings(BaseSettings):
    # Azure OpenAI 설정
    AOAI_ENDPOINT: str
    AOAI_API_KEY: str
    AOAI_DEPLOY_GPT_MINI: str
    AOAI_DEPLOY_GPT: str
    AOAI_DEPLOY_EMBED_3_LARGE: str
    AOAI_DEPLOY_EMBED_3_SMALL: str
    AOAI_DEPLOY_EMBED_ADA: str
    AOAI_API_VERSION: str

    # LLM Model 설정
    LLM_MODEL: str
    TC_LLM_MODEL: str
    API_BASE_URL: str
    
    # EMBEDDING Model 설정
    EMBEDDING_MODEL_NM: str

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Debate Arena API"

    # SQLite 데이터베이스 설정
    DB_PATH: str = "history.db"
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///./{DB_PATH}"

    # MCP Server 설정
    MCP_DEEP_RESEARCH_URL: str
    MCP_DEEP_RESEARCH_SYS_MESG: str
    
    MCP_DOC_RAG_URL: str
    MCP_DOC_RAG_SYS_MESG: str
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    _llm = None
    _tc_llm = None

    # 환경변수에서 설정한 LLM model의 인스턴스를 반환합니다. (싱글턴)
    def get_llm(self):
        if self._llm is None:
            api_base_url = self.API_BASE_URL
            llm_model = self.LLM_MODEL
            logger.info(f"[Config] LLM 인스턴스 생성: LLM_MODEL={llm_model}")

            if llm_model == "AzureChatOpenAI":
                from langchain_openai import AzureChatOpenAI

                self._llm = AzureChatOpenAI(
                     api_key=self.AOAI_API_KEY,
                     azure_endpoint=self.AOAI_ENDPOINT,
                     azure_deployment=self.AOAI_DEPLOY_GPT_MINI,
                     api_version=self.AOAI_API_VERSION,
                     temperature=0.7,
                )
            else:
                from langchain_ollama import ChatOllama

                self._llm = ChatOllama(base_url=api_base_url
                                       , model=llm_model
                                       , streaming=True
                                       , reasoning=False, # 강화된 질의문 생성 시 내부 추론 과정 생략 (최종 답변에 바로 집중)
                                      )
        return self._llm

    # 환경변수에서 설정한 LLM model의 인스턴스를 반환합니다. (싱글턴)
    def get_tc_llm(self):
        if self._tc_llm is None:
            api_base_url = self.API_BASE_URL
            tc_llm_model = self.TC_LLM_MODEL
            logger.info(f"[Config] Tool-Calling LLM 인스턴스 생성: TC_LLM_MODEL={tc_llm_model}")

            if tc_llm_model == "AzureChatOpenAI":
                from langchain_openai import AzureChatOpenAI

                self._tc_llm = AzureChatOpenAI(
                     openai_api_key=self.AOAI_API_KEY,
                     azure_endpoint=self.AOAI_ENDPOINT,
                     azure_deployment=self.AOAI_DEPLOY_GPT_MINI,
                     api_version=self.AOAI_API_VERSION,
                     temperature=0.7,
                     streaming=True,
                )
            else:
                from langchain_ollama import ChatOllama

                self._tc_llm = ChatOllama(
                    base_url=api_base_url,
                    model=tc_llm_model,
                    temperature=0.3,
                    reasoning=False, # 강화된 질의문 생성 시 내부 추론 과정 생략 (최종 답변에 바로 집중)
                )
        return self._tc_llm




    _embedding_model = None
    # 환경변수에서 설정한 Embedding model의 인스턴스를 반환합니다.
    def get_embeddings(self):
        
        if self._embedding_model is None:
            model = self.EMBEDDING_MODEL_NM
            if model == "AzureOpenAIEmbeddings":
                from langchain_openai import AzureOpenAIEmbeddings

                self._embedding_model = AzureOpenAIEmbeddings(
                                            model=self.AOAI_DEPLOY_EMBED_3_SMALL,
                                            openai_api_version=self.AOAI_API_VERSION,
                                            api_key=self.AOAI_API_KEY,
                                            azure_endpoint=self.AOAI_ENDPOINT,
                                            )

            else:
                from langchain_huggingface import HuggingFaceEmbeddings

                # Default로 HuggingFaceEmbeddings 인스턴스를 반환합니다.
                self._embedding_model = HuggingFaceEmbeddings( model_name='jhgan/ko-sroberta-nli'
                                                            , model_kwargs={'device':'cpu'} # cpu, cuda
                                                            , encode_kwargs={'normalize_embeddings':True}
                                                            )
        return self._embedding_model
    
# 설정 인스턴스 생성
settings = Settings()

# 편의를 위한 함수들, 하위 호환성을 위해 유지
def get_llm():
    return settings.get_llm()

def get_tc_llm():
    return settings.get_tc_llm()

def get_embeddings():
    return settings.get_embeddings()

def describe_llm_model() -> str:
    """추론 로그에 표시할, 실제로 사용 중인 답변 생성용 LLM 이름."""
    if settings.LLM_MODEL == "AzureChatOpenAI":
        return f"AzureOpenAI/{settings.AOAI_DEPLOY_GPT_MINI}"
    return f"Ollama/{settings.LLM_MODEL}"

def describe_tc_llm_model() -> str:
    """추론 로그에 표시할, 실제로 사용 중인 도구 호출(Tool-Calling)용 LLM 이름."""
    if settings.TC_LLM_MODEL == "AzureChatOpenAI":
        return f"AzureOpenAI/{settings.AOAI_DEPLOY_GPT_MINI}"
    return f"Ollama/{settings.TC_LLM_MODEL}"

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


# Settings 내부(get_llm/get_tc_llm 등)에서 공용으로 사용하는 로거 인스턴스
logger = get_logger("./log", "llm-server")
