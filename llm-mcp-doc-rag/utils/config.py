from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils import ragUtils


from llama_index.llms.langchain import LangChainLLM




# .env 파일에서 환경 변수 로드
load_dotenv()

class Settings(BaseSettings):
    # Server 설정
    SERVER_NAME: str
    SERVER_PORT: str

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
    API_BASE_URL: str
    
    # EMBEDDING Model 설정
    EMBEDDING_MODEL_NM: str

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    _llm = None

    # 환경변수에서 설정한 LLM model의 인스턴스를 반환합니다. (싱글턴)
    def get_llm(self):
        if self._llm is None:
            apiBaseUrl = self.API_BASE_URL
            llmModel   = self.LLM_MODEL
            print("model ========>>>>>>LLM_MODEL = " + llmModel)
            
            if llmModel == "AzureChatOpenAI" :
                from langchain_openai import AzureChatOpenAI
                self._llm = LangChainLLM(llm=AzureChatOpenAI(   api_key=self.AOAI_API_KEY,
                                                                azure_endpoint=self.AOAI_ENDPOINT,
                                                                azure_deployment=self.AOAI_DEPLOY_GPT_MINI,
                                                                api_version=self.AOAI_API_VERSION,
                                                                temperature=0.7,
                                                            )
                                        )
            else :
                # llamaindex의 Ollama를 사용하면 Langchain에서 실행한 llm을 종료하고 새로실행하는 문제가 발생하여 ChatOllama의 인스턴스를 직접 생성하여 LangChainLLM의 래퍼로 감싸는 방식으로 변경
                from langchain_ollama import ChatOllama
                self._llm = LangChainLLM(llm=ChatOllama(base_url=apiBaseUrl
                                                        , model=llmModel
                                                        , reasoning=False, # 강화된 질의문 생성 시 내부 추론 과정 생략 (최종 답변에 바로 집중)
                                                        )
                                        )
        return self._llm
    
    # 환경변수에서 설정한 Embedding model의 인스턴스를 반환합니다.
    def get_embeddings(self):
        model = self.EMBEDDING_MODEL_NM

        if model == "AzureOpenAIEmbeddings" :
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

            self._embedding_model = AzureOpenAIEmbedding(
                                        model=self.AOAI_DEPLOY_EMBED_3_SMALL,
                                        openai_api_version=self.AOAI_API_VERSION,
                                        api_key=self.AOAI_API_KEY,
                                        azure_endpoint=self.AOAI_ENDPOINT,
                                        )

        else :
            from torch import cuda
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            device = "cpu" 
            if cuda.is_available():
                device = "cuda"
            
            # 모델을 로컬 디렉토리에 저장하여 매번 다운로드하지 않도록 설정
            cache_folder = "./HuggingFaceEmbedding/model_cache"
            

            # Default로 HuggingFaceEmbeddings 인스턴스를 반환합니다.
            return HuggingFaceEmbedding(
                model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                device=device,
                cache_folder=cache_folder,
                trust_remote_code=False 
            )
        
      
    _query_engines = {}

    # 모드에 따라 쿼리엔진을 생성하고 싱글턴패턴으로 관리하는 함수
    def set_query_engine(self, mode, resource_paths):
        key = f"{mode}"
        if key not in self._query_engines:
            self._query_engines[key] = []
        
            # resource_paths는 세미콜론(;)으로 구분된 여러 경로를 가질 수 있으므로, 각각의 경로에 대해 쿼리 엔진을 생성하여 리스트에 추가
            for resource_path in resource_paths.split(";"):
                # 빈 문자열이 아닌 경우에만 처리
                if not resource_path.strip():
                    continue
                
                self._query_engines[key].append(ragUtils.make_rag_query_engine(resource_path, is_save=True, is_base_resource=True))

        return self._query_engines[key]

    # 모드에 따라 쿼리엔진을 생성하고 싱글턴패턴으로 관리하는 함수
    def get_query_engine(self, mode):
        return self._query_engines[mode]
        

# 설정 인스턴스 생성
settings = Settings()

# 편의를 위한 함수들, 하위 호환성을 위해 유지
def get_llm():
    return settings.get_llm()

def get_embeddings():
    return settings.get_embeddings()

def get_query_engine(mode):
    return settings.get_query_engine(mode)