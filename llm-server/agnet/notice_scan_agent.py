import os
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from pyexpat.errors import messages

from typing import List, Dict, Any, TypedDict, Optional, Literal
from pydantic import BaseModel, Field

# LangGraph: AI 에이전트들이 순서대로 협업하는 '워크플로우(흐름도)'를 짜게 해주는 라이브러리입니다.
from langgraph.graph import StateGraph, END

from utils import config


# 1.1 최종 추출될 조달 데이터 스키마 (Pydantic)
class ContractMethod(BaseModel):
    competition_type: Optional[str] = Field(None, description="경쟁 형태 (예: 제한경쟁(총액))")
    selection_type: Optional[str] = Field(None, description="낙찰자 결정 방식 (예: 협상에 의한 계약)")
    extraction_note: Optional[str] = Field(None, description="계약방법 관련 특이사항")

class ProcurementData(BaseModel):
    project_name: Optional[str] = Field(None, description="공식 사업명")
    estimated_amount_krw: Optional[int] = Field(None, description="부가가치세 포함 예산")
    bidding_qualifications: List[str] = Field(default_factory=list, description="입찰 참가 자격")
    announcement_period_days: Optional[int] = Field(None, description="공고 기간")
    contract_method: ContractMethod = Field(default_factory=ContractMethod)
    regulatory_review_notice: Optional[str] = Field(None, description="규정 위반 의심 사항")
    extraction_note: Optional[str] = Field(None, description="미명시 항목 사유")

# 1.2 LangGraph 전역 상태 정의
class AgentState(TypedDict):
    file_path: str                      # 입력 파일 경로
    document_text: Optional[str]        # 파싱된 텍스트
    extracted_data: Optional[Dict[str, Any]] # 최종 추출된 구조화 데이터
    is_violating: bool                  # 규정 위반 감지 여부 플래그
    status: Literal["success", "failed"] # 프로세스 처리 결과 상태
    error_message: Optional[str]        # 에러 발생 시 메시지


class PureLangNoticeScanAgent:

    # -------------------------------------------------------------
    # [Node 1] LangChain PyPDFLoader를 이용한 PDF 텍스트 파싱
    # -------------------------------------------------------------
    def parse_document_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Node: ParseDocument] 문서 텍스트 추출 중...")
        file_path = state["file_path"]

        if not os.path.exists(file_path):
            return {
                "status": "failed",
                "error_message": f"파일을 찾을 수 없습니다: {file_path}"
            }
            
        try:
            # 파일 확장자에 따른 로더 선택
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in [".html", ".htm"]:
                loader = BSHTMLLoader(file_path, open_encoding="utf-8")
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

            docs = loader.load()
            combined_text = "\n".join([doc.page_content for doc in docs])
            
            if not combined_text.strip():
                raise ValueError("문서 원문에서 추출된 텍스트가 비어 있습니다.")

            return {"document_text": combined_text, "status": "success"}
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"문서 파싱 중 에러 발생: {str(e)}"
            }

    # -------------------------------------------------------------
    # [Node 2] LangChain LLM Chain을 통한 정보 추출 및 스크리닝
    # -------------------------------------------------------------
    def extract_metadata_node(self, state: AgentState) -> Dict[str, Any]:
        print("[Node: ExtractMetadata] 메타데이터 구조화 및 규정 위반 검증 중...")
        
        chain = self.prompt | self.structured_llm
        try:
            response: ProcurementData = chain.invoke({"document_text": state["document_text"]})
            data_dict = response.model_dump()
            
            # 규정 위반 사항 기재 여부에 따른 플래그 세팅
            is_violating = bool(data_dict.get("regulatory_review_notice"))
            
            return {
                "extracted_data": data_dict,
                "is_violating": is_violating,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"구조화 데이터 생성 실패: {str(e)}"
            }

    # -------------------------------------------------------------
    # [Node 3] 에러 처리 노드
    # -------------------------------------------------------------
    def error_handling_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"[Node: ErrorHandling] 파이프라인 에러 처리 중 -> {state['error_message']}")
        return {
            "extracted_data": {
                "error": "Pipeline Interrupted",
                "reason": state["error_message"]
            }
        }

    # -------------------------------------------------------------
    # [Edge / Router] 라우팅 판단 함수
    # -------------------------------------------------------------
    def check_parsing_status(self, state: AgentState) -> Literal["continue", "error"]:
        if state["status"] == "failed":
            return "error"
        return "continue"

    # -------------------------------------------------------------
    # 워크플로우 그래프 선언 및 조립
    # -------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # 노드 배치
        workflow.add_node("parse_document", self.parse_document_node)
        workflow.add_node("extract_metadata", self.extract_metadata_node)
        workflow.add_node("error_handler", self.error_handling_node)
        
        # 진입점 설정
        workflow.set_entry_point("parse_document")
        
        # 조건부 라우팅 연결 (파싱 실패 시 대응)
        workflow.add_conditional_edges(
            "parse_document",
            self.check_parsing_status,
            {
                "continue": "extract_metadata",
                "error": "error_handler"
            }
        )
        
        # 종료 엣지 연결
        workflow.add_edge("extract_metadata", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()

    
    
    def __init__(self):
        # 1. LangChain LLM 및 구조화 출력 설정
        self.llm = config.get_llm().bind(stream=True, temperature=0)
        self.structured_llm = self.llm.with_structured_output(ProcurementData)
        
        # 로거 인스턴스 생성
        self.logger = config.get_logger("./log", "llm-server")
        
        # 2. 프롬프트 구성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "너는 비정형 조달 공고서에서 필수 항목을 추출하는 전문 에이전트이다.\n"
                "원문에 없는 내용은 절대로 추론하지 말고 null로 처리한 뒤 extraction_note에 사유를 적어라.\n"
                "소프트웨어 진흥법 등 규정 저촉 조항이 의심되면 regulatory_review_notice에 상세 경고를 기재하라."
            )),
            ("human", "공고서 본문:\n\n{document_text}")
        ])
        
        # 3. 그래프 빌드 및 컴파일
        self.graph = self._build_workflow()

        
    def run(self, file_path: str) -> Dict[str, Any]:
        """시나리오 가동 엔트리포인트 메소드"""
        initial_state: AgentState = {
            "file_path": file_path,
            "document_text": None,
            "extracted_data": None,
            "is_violating": False,
            "status": "success",
            "error_message": None
        }

        return self.graph.invoke(initial_state)