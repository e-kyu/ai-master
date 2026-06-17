import os
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from pyexpat.errors import messages

from typing import List, Dict, Any, TypedDict, Optional, Literal
from pydantic import BaseModel, Field

# LangGraph: AI 에이전트들이 순서대로 협업하는 '워크플로우(흐름도)'를 짜게 해주는 라이브러리입니다.
from langgraph.graph import StateGraph, END

from utils import config
 
from typing import List, Optional, Literal
 
from pydantic import BaseModel, Field, model_validator


# 'Y' / 'N' 외 다른 값이 들어올 수 없는 필드에 사용. 정보가 없으면 None 허용.
YNType = Optional[Literal["Y", "N"]]

# 1.1 최종 추출될 조달 데이터 스키마 (Pydantic)
class GeneralInfo(BaseModel):
    """공고 일반 정보 (general)"""
 
    noticeType: str = Field(
        description="공고 종류. 예: 물품, 공사, 용역, 외자, 비축물자 등"
    )
    noticeNo: str = Field(description="공고번호 (예: 20240601-001)")
    refNo: str = Field(default="", description="참조번호/관리번호. 없으면 빈 문자열")
    noticeName: str = Field(description="공고명(제목)")
    postDate: str = Field(
        default="", description="게시(공고)일시. 'YYYY-MM-DDTHH:MM' 형식으로 정규화"
    )
    agency: str = Field(default="", description="공고기관명 (예: 조달청)")
    demandAgency: str = Field(default="", description="수요기관명")
    contractType: str = Field(
        default="", description="계약구분 (예: 일반, 다수공급자계약, 협상계약 등)"
    )
    contractForm: str = Field(
        default="", description="계약형태 (예: 단가, 총액, 회차)"
    )
    bidMethod: str = Field(
        default="", description="입찰방식 (예: 전자입찰, 수기입찰)"
    )
    stockType: str = Field(
        default="",
        description="비축구분 (예: 비축, 해당없음 등). 비축공고가 아니면 '해당없음'",
    )
    contractMethod: str = Field(
        default="",
        description="계약방법 (예: 일반경쟁, 지명경쟁, 제한경쟁, 수의계약)",
    )
    awardMethod: str = Field(
        default="", description="낙찰방법 (예: 최저가, 적격심사, 협상에의한계약)"
    )
    awardDetail: str = Field(
        default="", description="낙찰방법에 대한 상세 설명 (예: 예가 이하 최저가)"
    )
    rebidYn: YNType = Field(default=None, description="재입찰 여부. Y 또는 N")
 
class ExecutionInfo(BaseModel):
    """입찰 집행(일정) 정보 (execution)"""
 
    manager: str = Field(default="", description="담당자 성명")
    bidStartDate: str = Field(default="", description="입찰 시작일시")
    bidEndDate: str = Field(default="", description="입찰 마감일시")
    openDate: str = Field(default="", description="개찰일시")
    openPlace: str = Field(default="", description="개찰장소")
    depositExemptYn: YNType = Field(
        default=None, description="입찰보증금 면제 여부. Y 또는 N"
    )
    depositDate: str = Field(
        default="", description="보증금 납부 마감일. 해당 없으면 빈 문자열"
    )
    relatedNotice: str = Field(
        default="", description="관련(연계) 공고번호. 없으면 빈 문자열"
    )
 
 
class Item(BaseModel):
    """공고 품목 정보 (items 배열의 원소)"""
 
    itemNo: str = Field(description="품목 순번 (1, 2, 3 ...)")
    itemName: str = Field(description="품명")
    standard: str = Field(default="", description="규격")
    unit: str = Field(default="", description="단위 (예: 박스, EA, kg)")
    quantity: Optional[float] = Field(default=None, description="수량")
    unitPrice: Optional[float] = Field(default=None, description="단가")
    amount: Optional[float] = Field(
        default=None, description="금액. 공고문에 명시되어 있지 않으면 수량 x 단가로 계산"
    )
 
    @model_validator(mode="after")
    def _fill_amount(self) -> "Item":
        if (
            self.amount is None
            and self.quantity is not None
            and self.unitPrice is not None
        ):
            self.amount = self.quantity * self.unitPrice
        return self
 
 
class Progress(BaseModel):
    """진행 단계 정보 (progresses 배열의 원소)"""
 
    stepName: str = Field(
        description="진행 단계명 (예: 공고, 입찰등록, 입찰마감, 개찰, 낙찰)"
    )
    stepDate: str = Field(default="", description="해당 단계 진행 일자")
    remark: str = Field(default="", description="비고")
 
 
class Status(BaseModel):
    """투찰/낙찰 현황 정보 (statuses 배열의 원소)"""
 
    bidderName: str = Field(description="투찰업체명")
    bidAmount: Optional[float] = Field(default=None, description="투찰금액")
    result: str = Field(
        default="", description="투찰 결과 (예: 낙찰, 낙찰탈락, 무효)"
    )

class ProcurementData(BaseModel):
    """
    조달청 나라장터 비축공고서 최종 구조체.
 
    LangGraph 노드에서:
        structured_llm = llm.with_structured_output(ProcurementNotice)
        result: ProcurementNotice = structured_llm.invoke(messages)
        payload = result.model_dump()   # 시스템 입력용 dict
    형태로 사용한다.
    """
 
    general: GeneralInfo
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    items: List[Item] = Field(default_factory=list)
    progresses: List[Progress] = Field(default_factory=list)
    statuses: List[Status] = Field(default_factory=list)

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