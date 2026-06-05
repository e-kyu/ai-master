import os
import json
from pyexpat.errors import messages
import streamlit as st
from typing import List, Dict, Any, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field

# LangChain: 대형 언어 모델(LLM)을 편리하게 다루도록 돕는 도구 모음입니다.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# LangGraph: AI 에이전트들이 순서대로 협업하는 '워크플로우(흐름도)'를 짜게 해주는 라이브러리입니다.
from langgraph.graph import StateGraph, END


from utils import config, convrstnContextUtils, mcpUtils, defaultPrompt
from repository import convrstn_repository

# 로거 인스턴스 생성
logger = config.get_logger("./log", "llm-server")


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
    convrstn_id: str                 # 대화 ID (세션 구분자)
    convrstn_details_id:str          # 대화 내역 ID (질문-답변 쌍 구분자)
    question: str                    # 사용자가 물어본 원래 질문
    convrstn_context: str            # 대화의 맥락을 담은 텍스트 (대화 내역 + 외부 지식 등)
    contextual_query: str            # 맥락이 강화된 질의문 (원래 질문 + 대화 맥락이 합쳐진 형태)
    queries: List[str]               # 에이전트가 순차적으로 다룰 질문 목록 (예: 1번 로봇이 3개의 질문을 만들어냈다면 ["질문1", "질문2", "질문3"])
    rag_answer: str                  # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    current_agent: str               # 현재 이 가방을 쥐고 있는 로봇의 이름


def start_convrstn_node(state: AgentState) -> Dict[str, Any]:
    # 대화 ID로 대화 시작 기록 생성
    convrstn_repository.create_convrstn(state['convrstn_id'], {"topic": state['question']})

    # 대화의 맥락을 조회 또는 생성
    convrstn_context = convrstnContextUtils.get_convrstn_context(state['convrstn_id'], state['question'])
    
    # 알아낸 정보를 공유 가방에 저장하고, 다음 바톤을 넘겨줄 로봇 이름을 지정합니다.
    return {
        "convrstn_context": convrstn_context
    }


def query_rewrite_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 1: 파싱 전문봇] 비정형 줄글 텍스트를 분석해 규격화된 서식(JSON)을 만듭니다."""
    
    # 강화된 질의문 생성
    contextual_query = convrstnContextUtils.enhance_query_with_context(state['convrstn_context'], state['question'])

    # 대화내역에 강화된 질의문과 맥락 저장 (질의문과 답변이 같은 테이블에 저장되도록 수정)
    create_convrstn_question = convrstn_repository.create_convrstn_question(state['convrstn_id'], {"context": json.dumps(state['convrstn_context'], ensure_ascii=False), "question": state['question'], "enhanced_question": contextual_query})
    
    # 알아낸 정보를 공유 가방에 저장하고, 다음 바톤을 넘겨줄 로봇 이름을 지정합니다.
    return {
        "convrstn_details_id": create_convrstn_question.convrstn_details_id,
        "contextual_query": contextual_query
    }


async def rag_tool_call_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 2: 규정 검토봇] RAG 검색을 통해 관련 법령 텍스트를 찾아옵니다."""

    sse_client_addr = config.settings.MCP_DOC_RAG_URL # SSE 클라이언트 주소
    sys_message = (
        "  - The first tool: 사용자가 첨부한 문서 내용에 근거한 답변이 필요한 경우 하위 문서경로를 사용한다. \n"
        f"    - 첨부문서경로(file_full_path) = {state['file_full_path']}"
    )

    # MCP 클라이언트를 통해 도구 호출 및 대화 프로세스 수행 (LLM과 MCP 서버 간의 반복 상호작용)
    mcpclient_manager = mcpUtils.get_mcp_manager()
    rag_answer = await mcpclient_manager.complete(sse_client_addr, sys_message, state['convrstn_context'], state['question'], False, HumanMessage(content=state['contextual_query']))

    # 찾아낸 리스크 리스트를 공유 가방에 담고, 전체 프로세스를 끝마칩니다(Output_Agent로 이동).
    return {
        "rag_answer": rag_answer
    }


async def excute_convrstn_agent(agent_info, convrstn_id: str, question: str, file_full_path: str) -> Any:

    

    # ----------------------------------------------------------------------
    # 4. LangGraph 파이프라인(협업 지도) 구성
    # ----------------------------------------------------------------------
    # 도면(그래프)을 그리는 과정입니다. 어떤 에이전트가 존재하고, 업무가 어떻게 흘러가는지 명시합니다.
    workflow = StateGraph(AgentState)

    # 1. 일꾼(노드) 등록하기
    workflow.add_node("Start Convrstn Agent", start_convrstn_node)
    workflow.add_node("Query Rewrite Agent", query_rewrite_node)
    workflow.add_node("Rag Tool Call Agent", rag_tool_call_node)

    # 2. 이동 경로(엣지) 연결하기
    workflow.set_entry_point("Start Convrstn Agent")                 # 시작은 무조건 파싱 전문봇이 합니다.
    workflow.add_edge("Start Convrstn Agent", "Query Rewrite Agent") # 파싱이 끝나면 자동으로 법령 검토봇에게 이동합니다.
    workflow.add_edge("Query Rewrite Agent", "Rag Tool Call Agent") # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).
    workflow.add_edge("Rag Tool Call Agent", END)                   # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).

    # 완성된 설계도를 실제로 작동 가능한 프로그램으로 컴파일(구동 준비)합니다.
    app_graph = workflow.compile()


    # ----------------------------------------------------------------------
    # 2. 실행할 때 초기값(Initial State)을 딕셔너리로 세팅합니다.
    # ----------------------------------------------------------------------
    initial_values = {
        "convrstn_id": convrstn_id,
        "question": question,
        "file_full_path": file_full_path,
    }

    # 3. 인보크(또는 스트림)할 때 첫 번째 인자로 전달합니다.
    agent_state = await app_graph.ainvoke(initial_values)
    print(agent_state)


    # 최종 답변 생성을 위한 시스템 지시사항 추가
    final_prompt = (
        "{persona_prompt}\n\n"
        "{final_answer_prompt}\n\n"
    )

    if not agent_info.persona_prompt:
        agent_info.persona_prompt = defaultPrompt.persona_prompt
    
    if not agent_info.final_prompt:
        agent_info.final_prompt = defaultPrompt.final_prompt

    final_prompt = final_prompt.format(persona_prompt=agent_info.persona_prompt if agent_info else ""
                                    , final_answer_prompt=agent_info.final_prompt if agent_info else ""
                                    )
    
    final_prompt += (
        "# INPUT DATA\n"
        f"- **Conversation Context:** {agent_state['convrstn_context']}\n"
        f"- **Original Question:** {agent_state['question']}\n"
        f"- **Enhanced Query:** {agent_state['contextual_query']}\n\n"
    )   
    
    rag_answer = agent_state['rag_answer']  # 이전 노드에서 전달된 대화 메시지 리스트
    rag_answer.append(SystemMessage(content=final_prompt))

    return agent_state