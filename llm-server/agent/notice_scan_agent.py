import asyncio
from typing import AsyncGenerator, Dict, Any, Optional, Literal, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, END
from utils import config, mcp_utils
from agent.notice_scan_structure import ProcurementData


class AgentState(TypedDict):
    file_path: str
    file_type: Optional[str]
    document_text: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    is_violating: bool
    status: Literal["success", "failed"]
    error_message: Optional[str]


class PureLangNoticeScanAgent:

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

        # 3. 그래프 빌드 및 컴파일 (한 번만 수행)
        self.graph = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("convert_to_markdown", self.convert_to_markdown_node)
        workflow.add_node("error_handler", self.error_handling_node)
        workflow.add_node("extract_metadata", self.extract_metadata_node)

        workflow.set_entry_point("convert_to_markdown")

        workflow.add_conditional_edges(
            "convert_to_markdown",
            self.check_parsing_status,
            {
                "continue": "extract_metadata",
                "error": "error_handler",
            },
        )

        workflow.add_edge("extract_metadata", END)
        workflow.add_edge("error_handler", END)

        return workflow.compile()

    def convert_to_markdown_node(self, state: AgentState) -> Dict[str, Any]:
        writer = get_stream_writer()
        self.logger.info(f"[NoticeScan][convert_to_markdown] 시작 file_path={state['file_path']}")
        writer({"type": "progress", "step": "convert_to_markdown", "label": "문서 변환", "status": "running", "message": "문서를 마크다운으로 변환하는 중..."})

        tool_info = {
            "tool": "convert_to_markdown",
            "input": {"file_path": state["file_path"]},
        }

        try:
            response = mcp_utils.call_tool(config.settings.MCP_DOC_RAG_URL, tool_info)
        except Exception as e:
            error_message = f"Markdown 변환 도구 호출 실패: {str(e)}"
            self.logger.error(f"[NoticeScan][convert_to_markdown] 실패 file_path={state['file_path']}: {error_message}")
            writer({"type": "progress", "step": "convert_to_markdown", "label": "문서 변환", "status": "failed", "message": error_message or "변환 실패"})
            return {
                "status": "failed",
                "error_message": error_message,
            }

        document_text = response.get("response") if isinstance(response, dict) else None
        if "status" in response and response["status"] == "success" and isinstance(document_text, str):
            pass
        else:
            error_message = "변환 도구 응답이 예상한 형식이 아닙니다."
            self.logger.error(f"[NoticeScan][convert_to_markdown] 실패 file_path={state['file_path']}: {error_message}")
            writer({"type": "progress", "step": "convert_to_markdown", "label": "문서 변환", "status": "failed", "message": error_message})
            return {
                "status": "failed",
                "error_message": error_message,
            }
        if not document_text:
            error_message = "Markdown 변환 도구가 텍스트를 반환하지 않았습니다."
            self.logger.error(f"[NoticeScan][convert_to_markdown] 실패 file_path={state['file_path']}: {error_message}")
            writer({"type": "progress", "step": "convert_to_markdown", "label": "문서 변환", "status": "failed", "message": error_message})
            return {
                "status": "failed",
                "error_message": error_message,
            }

        self.logger.info(f"[NoticeScan][convert_to_markdown] 완료 file_path={state['file_path']} document_length={len(document_text)}")
        writer({"type": "progress", "step": "convert_to_markdown", "label": "문서 변환", "status": "done", "message": "변환 완료"})
        return {
            "document_text": document_text,
            "status": "success",
        }

    def extract_metadata_node(self, state: AgentState) -> Dict[str, Any]:
        writer = get_stream_writer()
        self.logger.info(f"[NoticeScan][extract_metadata] 시작 file_path={state['file_path']}")
        writer({"type": "progress", "step": "extract_metadata", "label": "정보 추출", "status": "running", "message": "메타데이터 구조화 및 규정 위반 검증 중..."})

        chain = self.prompt | self.structured_llm
        try:
            response: ProcurementData = chain.invoke({"document_text": state["document_text"]})
            data_dict = response.model_dump()
            is_violating = bool(data_dict.get("regulatory_review_notice"))

            self.logger.info(f"[NoticeScan][extract_metadata] 완료 file_path={state['file_path']} is_violating={is_violating}")
            writer({"type": "progress", "step": "extract_metadata", "label": "정보 추출", "status": "done", "message": "추출 완료"})
            return {
                "extracted_data": data_dict,
                "is_violating": is_violating,
                "status": "success"
            }
        except Exception as e:
            error_message = f"구조화 데이터 생성 실패: {str(e)}"
            self.logger.error(f"[NoticeScan][extract_metadata] 실패 file_path={state['file_path']}: {error_message}")
            writer({"type": "progress", "step": "extract_metadata", "label": "정보 추출", "status": "failed", "message": error_message or "추출 실패"})
            return {
                "status": "failed",
                "error_message": error_message
            }

    def error_handling_node(self, state: AgentState) -> Dict[str, Any]:
        self.logger.error(f"[NoticeScan][error_handler] 파이프라인 중단 file_path={state['file_path']} reason={state['error_message']}")
        return {
            "extracted_data": {
                "error": "Pipeline Interrupted",
                "reason": state["error_message"]
            }
        }

    def check_parsing_status(self, state: AgentState) -> Literal["continue", "error"]:
        if state["status"] == "failed":
            return "error"
        return "continue"

    async def run(self, file_path: str) -> Dict[str, Any]:
        """시나리오 가동 엔트리포인트 메소드"""
        self.logger.info(f"[NoticeScan][run] 파이프라인 시작 file_path={file_path}")
        initial_state: AgentState = {
            "file_path": file_path,
            "file_type": None,
            "document_text": None,
            "extracted_data": None,
            "is_violating": False,
            "status": "success",
            "error_message": None
        }

        final_state = await self.graph.ainvoke(initial_state)
        self.logger.info(f"[NoticeScan][run] 파이프라인 종료 file_path={file_path} status={final_state.get('status')}")
        return final_state

    async def run_stream(self, file_path: str) -> AsyncGenerator[Dict[str, Any], None]:
        """실시간 진행상태를 SSE 이벤트로 yield하는 스트리밍 엔트리포인트"""
        initial_state: AgentState = {
            "file_path": file_path,
            "file_type": None,
            "document_text": None,
            "extracted_data": None,
            "is_violating": False,
            "status": "success",
            "error_message": None,
        }

        # Notify overall start
        self.logger.info(f"[NoticeScan][run_stream] 파이프라인 시작 file_path={file_path}")
        yield {"type": "progress", "step": "pipeline", "label": "파이프라인", "status": "running", "message": "파이프라인 실행 중..."}

        # 컴파일된 워크플로우를 스트리밍합니다: "custom"은 get_stream_writer()를 통해 전달되는
        # 노드 수준의 진행 이벤트를 포함하고, "values"는 각 노드 이후의 누적 상태를 담아
        # 최종 결과에 대한 최신 스냅샷을 항상 유지합니다.
        final_state: Dict[str, Any] = dict(initial_state)
        try:
            async for mode, payload in self.graph.astream(initial_state, stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield payload
                elif mode == "values":
                    final_state = payload
        except Exception as e:
            error_message = f"파이프라인 실행 중 예외 발생: {str(e)}"
            self.logger.error(f"[NoticeScan][run_stream] 파이프라인 예외 file_path={file_path}: {error_message}")
            final_state = {
                **final_state,
                "status": "failed",
                "error_message": error_message,
                "extracted_data": {"error": "Pipeline Exception", "reason": str(e)},
            }
            yield {"type": "progress", "step": "pipeline", "label": "파이프라인", "status": "failed", "message": error_message}

        self.logger.info(f"[NoticeScan][run_stream] 파이프라인 종료 file_path={file_path} status={final_state.get('status')}")
        yield {"type": "result", "data": dict(final_state)}