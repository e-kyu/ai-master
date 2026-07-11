import os
import mcp.types as types
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, Literal, TypedDict
from markitdown import MarkItDown
import pymupdf4llm

from utils import config, logger_util


# 로거 인스턴스 생성
logger = logger_util.get_logger("./log", "llm-mcp-doc-rag")

# LangGraph 상태 정의
class GraphState(TypedDict):
    file_path: str                           # 입력 파일 경로
    file_type: Optional[str]                 # 입력 파일 타입 (pdf, html 
    document_text: Optional[str]             # 파싱된 텍스트
    status: Literal["success", "failed"]     # 프로세스 처리 결과 상태
    error_message: Optional[str]             # 에러 발생 시 메시지



class ConvertToMarkdownAgent:
    def detect_file_type(self, state: GraphState) -> Dict[str, Any]:
        """확장자를 검사해 파일 타입을 결정"""
        ext = os.path.splitext(state["file_path"])[1].lower()

        if ext in [".html", ".htm"]:
            file_type = "html"
        elif ext == ".pdf":
            file_type = "pdf"
        else:
            file_type = "unknown"

        return {"file_type": file_type}

    def convert_html_node(self, state: GraphState) -> Dict[str, Any]:
        """markitdown으로 HTML -> Markdown 변환"""
        try:
            md_converter = MarkItDown()
            result = md_converter.convert(state["file_path"])
            return {
                "document_text": result.text_content,
                "status": "success",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"HTML 변환 실패: {e}",
            }


    def convert_pdf_node(self, state: GraphState) -> Dict[str, Any]:
        """pymupdf4llm으로 PDF -> Markdown 변환 (표/레이아웃 인식)"""
        try:
            markdown_text = pymupdf4llm.to_markdown(state["file_path"])
            return {
                "document_text": markdown_text,
                "status": "success",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"PDF 변환 실패: {e}",
            }
    

    def handle_unknown_node(self, state: GraphState) -> Dict[str, Any]:
        """지원하지 않는 파일 형식 처리"""
        ext = os.path.splitext(state["file_path"])[1]
        return {
            "status": "failed",
            "error_message": f"지원하지 않는 파일 형식입니다: {ext}",
        }

    # ============================================================
    # 3. 라우팅 함수
    # ============================================================
    def route_by_file_type(self, state: GraphState) -> str:
        return state["file_type"]

    # -------------------------------------------------------------
    # 워크플로우 그래프 선언 및 조립
    # -------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        
        # 노드 배치
        workflow.add_node("detect_file_type", self.detect_file_type)
        workflow.add_node("convert_html"    , self.convert_html_node)
        workflow.add_node("convert_pdf"     , self.convert_pdf_node)
        workflow.add_node("handle_unknown"  , self.handle_unknown_node)
        
        # 진입점 설정
        workflow.set_entry_point("detect_file_type")

        workflow.add_conditional_edges(
            "detect_file_type",
            self.route_by_file_type,
            {
                "html": "convert_html",
                "pdf": "convert_pdf",
                "unknown": "handle_unknown",
            },
        )

        workflow.add_edge("convert_html", END)
        workflow.add_edge("convert_pdf", END)
        workflow.add_edge("handle_unknown", END)
        
        return workflow.compile()

def execute(file_path: str = "") -> Optional[str]:
    try:
        agent = ConvertToMarkdownAgent()
        workflow = agent._build_workflow()

        initial_state = {
            "file_path": file_path,
            "file_type": None,
            "document_text": None,
            "status": None,
            "error_message": None
        }

        result_state = workflow.invoke(initial_state)

        if result_state["status"] == "success":
            return  [types.TextContent(type="text", text=result_state["document_text"])]    
        else:
            logger.error(f"Markdown 변환 실패: {result_state['error_message']}")
            return None

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    
    finally:
        logger.info("RAG processing completed.")   
