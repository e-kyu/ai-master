import json
from pyexpat.errors import messages
from typing import AsyncGenerator, List, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field

# LangChain: 대형 언어 모델(LLM)을 편리하게 다루도록 돕는 도구 모음입니다.
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# LangGraph: AI 에이전트들이 순서대로 협업하는 '워크플로우(흐름도)'를 짜게 해주는 라이브러리입니다.
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, END


from utils import config, conversation_context_utils, mcp_utils, default_prompt
from utils.reasoning_log import ReasoningLog, planning_event
from repository import conversation_repository

# 로거 인스턴스 생성
logger = config.get_logger("./log", "llm-server")

AGENT_NAME = "PpsAssistAgent"


# ----------------------------------------------------------------------
# 1. 에이전트 노드(Node) 및 핵심 기능 구현
# ----------------------------------------------------------------------
# 메인 두뇌가 될 최신 AI 모델(GPT-4o)을 불러옵니다. 창의성을 낮추어(temperature=0) 정확한 사실만 말하게 합니다.
llm = config.get_llm().bind(stream=True, temperature=0)


# ----------------------------------------------------------------------
# 2. Output Parsing (결과물 구조화 정의)
# ----------------------------------------------------------------------
# AI가 자유롭게 답변하면 매번 형식이 바뀌어 프로그램에 입력하기 어렵습니다.
# Pydantic 라이브러리를 사용해 AI가 '반드시 이 규격(틀)에 맞춰서 대답'하도록 강제합니다.
class SystemInputForm(BaseModel):
    project_name: str = Field(description="공고서에서 추출한 사업명 또는 프로젝트 이름")
    budget: str = Field(description="부가가치세를 포함한 총 배정 예산 또는 추정 가격 (원 단위 포함)")
    eligibility: str = Field(description="입찰 참가 자격 핵심 요약")
    period: str = Field(description="공고 기간 또는 수행 기간")
    contract_method: str = Field(description="계약 방법 (예: 협상에 의한 계약, 제한경쟁 등)")

# ----------------------------------------------------------------------
# 3. LangGraph 오케스트레이션 상태(State) 정의
# ----------------------------------------------------------------------
# 에이전트(공고분석봇, 규정검토봇)들이 서로 대화하고 협업할 때, 
# 각자 알아낸 정보를 담아서 다음 로봇에게 넘겨줄 '공유 가방(메모리 메모)'을 정의합니다.
class AgentState(TypedDict):
    messages: List[BaseMessage]      # 전체 대화 내용 기록 기록부
    file_full_path: str              # 사용자가 업로드한 파일의 경로
    conversation_id: str                 # 대화 ID (세션 구분자)
    conversation_details_id:str          # 대화 내역 ID (질문-답변 쌍 구분자)
    question: str                    # 사용자가 물어본 원래 질문
    conversation_context: str            # 대화의 맥락을 담은 텍스트 (대화 내역 + 외부 지식 등)
    contextual_query: str            # 맥락이 강화된 질의문 (원래 질문 + 대화 맥락이 합쳐진 형태)
    queries: List[str]               # 에이전트가 순차적으로 다룰 질문 목록 (예: 1번 로봇이 3개의 질문을 만들어냈다면 ["질문1", "질문2", "질문3"])
    rag_answer: List[BaseMessage|SystemMessage]          # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    qna_doc_answer: str                  # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    qna_law_base_answer: str                  # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    qna_web_search_answer: str                  # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    current_agent: str               # 현재 이 가방을 쥐고 있는 로봇의 이름
    enable_ext_docs: bool            # 외부 문서 검색을 활성화할지 여부
    error_message: str                  # 에러 발생 시, 에러 메시지를 담는 필드
    status: Literal["success", "failed"]  # 에이전트 상태 (성공/실패)


def start_conversation_node(state: AgentState) -> Dict[str, Any]:
    """[Start Conversation Agent] 대화 세션을 생성하고 이전 대화 맥락을 불러옵니다."""
    writer = get_stream_writer()
    rlog = ReasoningLog(logger, writer, AGENT_NAME)
    logger.info(f"[Start Conversation Agent] 시작 conversation_id={state['conversation_id']} question={state['question'][:50]}")

    # 대화 ID로 대화 시작 기록 생성
    conversation_repository.create_conversation(state['conversation_id'], {"topic": state['question']})

    # 대화의 맥락을 조회 또는 생성 (내부에서 LLM 호출 — rlog로 상세 로그 전달)
    conversation_context = conversation_context_utils.get_conversation_context(state['conversation_id'], state['question'], rlog=rlog)

    logger.info(f"[Start Conversation Agent] 완료 conversation_id={state['conversation_id']}")
    # 알아낸 정보를 공유 가방에 저장하고, 다음 바톤을 넘겨줄 로봇 이름을 지정합니다.
    return {
        "conversation_context": conversation_context,
    }


def query_rewrite_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 1: 파싱 전문봇] 비정형 줄글 텍스트를 분석해 규격화된 서식(JSON)을 만듭니다."""
    writer = get_stream_writer()
    rlog = ReasoningLog(logger, writer, AGENT_NAME)
    logger.info(f"[Query Rewrite Agent] 시작 conversation_id={state['conversation_id']}")

    # 강화된 질의문 생성 (내부에서 LLM 호출 — rlog로 상세 로그 전달)
    contextual_query = conversation_context_utils.enhance_query_with_context(state['conversation_context'], state['question'], rlog=rlog)

    # 대화내역에 강화된 질의문과 맥락 저장 (질의문과 답변이 같은 테이블에 저장되도록 수정)
    create_conversation_question = conversation_repository.create_conversation_question(state['conversation_id'], {"context": json.dumps(state['conversation_context'], ensure_ascii=False), "question": state['question'], "enhanced_question": contextual_query})

    logger.info(f"[Query Rewrite Agent] 완료 conversation_id={state['conversation_id']} contextual_query={contextual_query[:100]}")
    # 알아낸 정보를 공유 가방에 저장하고, 다음 바톤을 넘겨줄 로봇 이름을 지정합니다.
    return {
        "conversation_details_id": create_conversation_question.convrstn_details_id,
        "contextual_query": contextual_query
    }


def qna_doc_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 2: 규정 검토봇] 사용자가 첨부한 파일을 RAG탐색하여 질의문의 내용을 찾아옵니다."""
    writer = get_stream_writer()
    rlog = ReasoningLog(logger, writer, AGENT_NAME)
    logger.info(f"[Document Search Agent] 시작 conversation_id={state.get('conversation_id')} file={state.get('file_full_path')}")

    qna_doc_answer = ""
    if not state["file_full_path"]:
        logger.info(f"[Document Search Agent] 첨부 파일 없어 건너뜀 conversation_id={state.get('conversation_id')}")
    else:
        tool_info = {
            "tool": "qna_doc",
            "input": {"question":state["contextual_query"], "fileFullPath": state["file_full_path"]},
        }

        response = mcp_utils.call_tool_with_self_correction(
            rlog, config.settings.MCP_DOC_RAG_URL, tool_info, context_name="Doc-RAG MCP (첨부문서 RAG)"
        )

        if response.get("status") != "success":
            error_message = response.get("error_message") or f"{tool_info['tool']} 도구 호출 실패"
            logger.error(f"[Document Search Agent] 실패 conversation_id={state.get('conversation_id')}: {error_message}")
            return {
                "status": "failed",
                "error_message": error_message,
            }

        qna_doc_answer = response.get("response")
        logger.info(f"[Document Search Agent] 완료 conversation_id={state.get('conversation_id')} answer_length={len(qna_doc_answer) if qna_doc_answer else 0}")

    # EVALUATING: 다음 단계(법령 베이스 RAG vs 실시간 웹 검색) 경로를 ToT로 평가하여 로그로 남긴다.
    enable_ext_docs = bool(state.get("enable_ext_docs"))
    if enable_ext_docs:
        paths = [
            {"name": "법령/가이드 기반 검색 (내부 지식베이스)", "score": 55, "reason": "사용자가 외부 문서 검색을 허용함"},
            {"name": "실시간 웹 검색 (외부 최신정보)", "score": 85, "reason": "사용자가 외부 문서 검색을 허용함"},
        ]
        selected = "실시간 웹 검색 (외부 최신정보)"
    else:
        paths = [
            {"name": "법령/가이드 기반 검색 (내부 지식베이스)", "score": 90, "reason": "외부 문서 검색이 비활성화되어 내부 지식베이스가 신뢰도 높음"},
            {"name": "실시간 웹 검색 (외부 최신정보)", "score": 40, "reason": "외부 문서 검색이 비활성화됨"},
        ]
        selected = "법령/가이드 기반 검색 (내부 지식베이스)"
    rlog.evaluating(paths, selected)

    # 찾아낸 리스크 리스트를 공유 가방에 담고, 전체 프로세스를 끝마칩니다(Output_Agent로 이동).
    return {
        "qna_doc_answer": qna_doc_answer
    }


def qna_law_base_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 3: 규정 검토봇] 법령, 가이드 파일로 준비된 VectorDB를 RAG탐색하여 질의문의 내용을 찾아옵니다."""
    writer = get_stream_writer()
    rlog = ReasoningLog(logger, writer, AGENT_NAME)
    logger.info(f"[Law Base Search Agent] 시작 conversation_id={state.get('conversation_id')}")
    # TODO: 질의문과 맥락으로 agent_mode를 결정하는 로직을 추가해야 합니다. 현재는 임시로 PpsStockpilingAgent로 고정되어 있습니다.
    """
                    "agent_mode": {
                        "type": "string",
                        "description": ("사용자 질문에 적합한 에이전트를 아래 목록에서 '하나만' 선택:\n"
                                        "- PpsGeneralServiceAgent: 조달청 일반용역 전문 상담\n"
                                        "- PpsTechnicalServicesAgent: 조달청 기술용역 전문 상담\n"
                                        "- PpsConstructionAgent: 조달청 시설공사 전문 상담\n"
                                        "- PpsProductsAgent: 조달청 물품 관련 전문 상담\n"
                                        "- PpsStockpilingAgent: 조달청 비축 관련 전문 상담")
                    },
    """
    
    llm = config.get_llm().bind(stream=False)

    prompt = ("당신은 조달청 상담의 전문분야를 지정하는 에이전트입니다.\n\n"
                "사용자가 입력한 질문과 대화 맥락을 분석하여, 아래 목록에서 가장 적합한 에이전트를 '하나만' 선택하세요.\n\n"
                "- PpsGeneralServiceAgent: 조달청 일반용역 전문 상담\n"
                "- PpsTechnicalServicesAgent: 조달청 기술용역 전문 상담\n"
                "- PpsConstructionAgent: 조달청 시설공사 전문 상담\n"
                "- PpsProductsAgent: 조달청 물품 관련 전문 상담\n"
                "- PpsStockpilingAgent: 조달청 비축 관련 전문 상담\n\n"
                "선택 규칙\n"
                "1. 사용자 질문과 대화 맥락만을 기반으로 판단합니다.\n"
                "2. 하나의 에이전트로 명확하게 분류할 수 있는 경우 해당 에이전트 이름만 출력합니다.\n"
                "3. 질문이 조달청 상담과 무관하거나, 제공된 정보만으로 특정 전문분야를 판단할 수 없거나, 둘 이상의 에이전트가 동일한 수준으로 적합한 경우에는 `None`을 출력합니다.\n"
                "4. 다른 설명, 이유, 부가 문구, 마크다운은 절대 출력하지 않습니다.\n\n"
                "사용자 질문: {contextual_query}\n"
                "대화 맥락: {context}\n\n"
                "출력 형식\n"
                "- PpsGeneralServiceAgent\n"
                "- PpsTechnicalServicesAgent\n"
                "- PpsConstructionAgent\n"
                "- PpsProductsAgent\n"
                "- PpsStockpilingAgent\n"
                "- None\n"
                )
    classification_model = config.describe_llm_model()
    try:
        logger.debug(f"Determining agent_mode for conversation_id={state.get('conversation_id')}")
        rlog.llm_call(
            classification_model, "5개 조달청 전문분야 에이전트 중 질문에 가장 적합한 분야 분류(Zero-shot Classification)",
            prompt_preview=f"질문: \"{state['contextual_query']}\" / 맥락: {state['conversation_context'] or '(없음)'}",
            params={"candidates": ["PpsGeneralServiceAgent", "PpsTechnicalServicesAgent", "PpsConstructionAgent", "PpsProductsAgent", "PpsStockpilingAgent"]},
        )
        llm_response = llm.invoke(prompt.format(contextual_query=state["contextual_query"], context=state["conversation_context"]))
        rlog.llm_response(classification_model, "분류 결과 원문 수신", output_preview=getattr(llm_response, "content", None))
    except Exception as e:
        logger.error(f"agent_mode determination failed for conversation_id={state.get('conversation_id')}: {e}")
        llm_response = None


    agent_mode = getattr(llm_response, "content", None) if llm_response else None
    if not agent_mode or not isinstance(agent_mode, str) or agent_mode == "None":
        error_message = (
            "문의 내용을 하나의 전문분야로 판단하기 어렵습니다.\n"
            "아래 항목 중 가장 가까운 분야를 하나만 선택해 주세요.\n"
            "조달청 일반용역 상담\n"
            "조달청 기술용역 상담\n"
            "조달청 시설공사 상담\n"
            "조달청 물품 관련 상담\n"
            "조달청 비축 관련 상담\n"
            "선택하신 분야를 기준으로 상담을 이어가겠습니다."
        )
        logger.error(f"[Law Base Search Agent] agent_mode missing conversation_id={state.get('conversation_id')}")
        return {
            "status": "failed",
            "error_message": error_message,
        }

    # agent_mode 값이 없거나 유효하지 않은 경우, 에러 메시지를 리턴합니다.
    allowed_agent_modes = [
        "PpsGeneralServiceAgent",
        "PpsTechnicalServicesAgent",
        "PpsConstructionAgent",
        "PpsProductsAgent",
        "PpsStockpilingAgent",
    ]

    agent_mode = agent_mode.strip()
    if agent_mode not in allowed_agent_modes:
        error_message = (
            "문의 내용을 하나의 전문분야로 판단하기 어렵습니다.\n"
            "아래 항목 중 분야를 선택해 주세요.\n"
            "조달청 일반용역 상담\n"
            "조달청 기술용역 상담\n"
            "조달청 시설공사 상담\n"
            "조달청 물품 관련 상담\n"
            "조달청 비축 관련 상담\n"
            "선택하신 분야를 기준으로 상담을 이어가겠습니다."
        )
        logger.error(f"[Law Base Search Agent] invalid agent_mode conversation_id={state.get('conversation_id')} agent_mode={agent_mode}")
        return {
            "status": "failed",
            "error_message": error_message,
        }

    logger.info(f"[Law Base Search Agent] agent_mode 결정 conversation_id={state.get('conversation_id')} agent_mode={agent_mode}")

    tool_info = {
        "tool": "qna_law_base",
        "input": {"question":state["contextual_query"], "agent_mode": agent_mode},
    }

    response = mcp_utils.call_tool_with_self_correction(
        rlog, config.settings.MCP_DOC_RAG_URL, tool_info, context_name="Doc-RAG MCP (법령 RAG)"
    )

    if response.get("status") != "success":
        error_message = response.get("error_message") or f"{tool_info['tool']} 도구 호출 실패"
        logger.error(f"[Law Base Search Agent] 실패 conversation_id={state.get('conversation_id')} agent_mode={agent_mode}: {error_message}")
        return {
            "status": "failed",
            "error_message": error_message,
        }

    rag_answer = response.get("response")

    logger.info(f"[Law Base Search Agent] 완료 convrstn_id={state.get('convrstn_id')} answer_length={len(rag_answer) if rag_answer else 0}")
    # 찾아낸 리스크 리스트를 공유 가방에 담고, 전체 프로세스를 끝마칩니다(Output_Agent로 이동).
    return {
        "qna_law_base_answer": rag_answer
    }


def qna_web_search_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 4: 규정 검토봇] 인터넷 검색 결과를 RAG 검색을 통해 관련 법령 텍스트를 찾아옵니다."""
    writer = get_stream_writer()
    rlog = ReasoningLog(logger, writer, AGENT_NAME)
    logger.info(f"[Web Search Agent] 시작 conversation_id={state.get('conversation_id')}")
    tool_info = {
        "tool": "qna_web_search",
        "input": {"question":state["contextual_query"], "allow_search": True},
    }

    response = mcp_utils.call_tool_with_self_correction(
        rlog, config.settings.MCP_DOC_RAG_URL, tool_info, context_name="Doc-RAG MCP (웹 검색 RAG)"
    )

    if response.get("status") != "success":
        error_message = response.get("error_message") or f"{tool_info['tool']} 도구 호출 실패"
        logger.error(f"[Web Search Agent] 실패 conversation_id={state.get('conversation_id')}: {error_message}")
        return {
            "status": "failed",
            "error_message": error_message,
        }

    rag_answer = response.get("response")

    logger.info(f"[Web Search Agent] 완료 convrstn_id={state.get('convrstn_id')} answer_length={len(rag_answer) if rag_answer else 0}")
    # 찾아낸 리스크 리스트를 공유 가방에 담고, 전체 프로세스를 끝마칩니다(Output_Agent로 이동).
    return {
        "qna_web_search_answer": rag_answer
    }


def check_enable_ext_docs(state: AgentState) -> Literal["law_base", "web_search"]:
    branch = "web_search" if state["enable_ext_docs"] else "law_base"
    logger.info(f"[Router] enable_ext_docs={state['enable_ext_docs']} -> '{branch}' 노드로 분기 conversation_id={state.get('conversation_id')}")
    if state["enable_ext_docs"] == False:
        return "law_base"
    return "web_search"


async def run_stream(agent_info, conversation_id: str, question: str, file_full_path: str, enable_ext_docs: bool) -> AsyncGenerator[Dict[str, Any], None]:
    """실시간 추론 로그를 SSE 이벤트로 yield하고, 마지막에 {"type":"state", ...}로 최종 상태를 전달하는 스트리밍 엔트리포인트"""
    # ----------------------------------------------------------------------
    # 4. LangGraph 파이프라인(협업 지도) 구성
    # ----------------------------------------------------------------------
    # 도면(그래프)을 그리는 과정입니다. 어떤 에이전트가 존재하고, 업무가 어떻게 흘러가는지 명시합니다.
    workflow = StateGraph(AgentState)

    # 1. 일꾼(노드) 등록하기
    workflow.add_node("Start Conversation Agent", start_conversation_node)
    workflow.add_node("Query Rewrite Agent", query_rewrite_node)
    workflow.add_node("Document Search Agent", qna_doc_node)
    workflow.add_node("Law Base Search Agent", qna_law_base_node)
    workflow.add_node("Web Search Agent", qna_web_search_node)

    # 2. 이동 경로(엣지) 연결하기
    workflow.set_entry_point("Start Conversation Agent")                 # 시작은 무조건 파싱 전문봇이 합니다.
    workflow.add_edge("Start Conversation Agent", "Query Rewrite Agent") # 파싱이 끝나면 자동으로 법령 검토봇에게 이동합니다.
    workflow.add_edge("Query Rewrite Agent", "Document Search Agent") # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).

    workflow.add_conditional_edges(
        "Document Search Agent",
        check_enable_ext_docs,
        {
            "law_base": "Law Base Search Agent",
            "web_search": "Web Search Agent",
        },
    )

    workflow.add_edge("Law Base Search Agent", END)                   # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).
    workflow.add_edge("Web Search Agent", END)                   # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).

    # 완성된 설계도를 실제로 작동 가능한 프로그램으로 컴파일(구동 준비)합니다.
    app_graph = workflow.compile()

    # ----------------------------------------------------------------------
    # 2. 실행할 때 초기값(Initial State)을 딕셔너리로 세팅합니다.
    # ----------------------------------------------------------------------
    initial_values = {
        "conversation_id": conversation_id,
        "question": question,
        "file_full_path": file_full_path,
        "enable_ext_docs": enable_ext_docs,
        "rag_answer": [],
        "qna_doc_answer" : "",
        "qna_law_base_answer" : "",
        "qna_web_search_answer" : "",
    }

    logger.info(f"Starting agent run: conversation_id={conversation_id} question={question[:50]}")

    # PLANNING: 그래프 진입 전이라 get_stream_writer()가 동작하지 않으므로 빌더 함수를 직접 호출해 yield한다.
    branch_label = "실시간 웹 검색 경로 (MCP: qna_web_search)" if enable_ext_docs else "법령 베이스 RAG 검색 경로 (MCP: qna_law_base)"
    plan_event, plan_text = planning_event(AGENT_NAME, [
        {"label": "대화 세션 시작 및 이전 맥락 조회"},
        {"label": "질의 강화 (Query Rewrite)"},
        {"label": "첨부문서 RAG 검색", "tool": "qna_doc"},
        {"label": f"경로 분기 결정 (ToT 평가) → {branch_label}"},
    ])
    logger.info(f"[{AGENT_NAME}] {plan_text}")
    yield plan_event

    # 3. 컴파일된 워크플로우를 스트리밍합니다: "custom"은 get_stream_writer()를 통해 전달되는
    # 노드 수준의 추론 로그를 포함하고, "values"는 각 노드 이후의 누적 상태를 담아
    # 최종 결과에 대한 최신 스냅샷을 항상 유지합니다.
    agent_state: Dict[str, Any] = dict(initial_values)
    async for mode, payload in app_graph.astream(initial_values, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield payload
        elif mode == "values":
            agent_state = payload

    logger.info(f"Completed workflow for conversation_id={conversation_id}")
    logger.debug(f"agent_state keys: {list(agent_state.keys())}")


    # 최종 답변 생성을 위한 시스템 지시사항 추가
    final_prompt = (
        "{persona_prompt}\n\n"
        "{final_answer_prompt}\n\n"
    )

    if not agent_info.persona_prompt:
        agent_info.persona_prompt = default_prompt.persona_prompt

    if not agent_info.final_prompt:
        agent_info.final_prompt = default_prompt.final_prompt

    final_prompt = final_prompt.format(persona_prompt=agent_info.persona_prompt if agent_info else ""
                                    , final_answer_prompt=agent_info.final_prompt if agent_info else ""
                                    )
    
    final_prompt += (
        "# INPUT DATA\n"
        f"- **Conversation Context:** {agent_state['conversation_context']}\n"
        f"- **Original Question:** {agent_state['question']}\n"
        f"- **Enhanced Query:** {agent_state['contextual_query']}\n\n"
    )
    

    if agent_state.get('status') == "failed":
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('error_message', '')))
    else:
        agent_state['rag_answer'].append(SystemMessage(content=final_prompt))
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('status', '')))
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('error_message', '')))
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('qna_doc_answer', '')))
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('qna_law_base_answer', '')))
        agent_state['rag_answer'].append(SystemMessage(content=agent_state.get('qna_web_search_answer', '')))

    logger.info(f"Prepared final RAG messages for conversation_id={conversation_id}")

    yield {"type": "state", "data": agent_state}