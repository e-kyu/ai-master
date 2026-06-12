import os
import json
import re
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
    result_data: Dict[str, str]      # 맥락이 강화된 질의문 (원래 질문 + 대화 맥락이 합쳐진 형태)
    queries: List[str]               # 에이전트가 순차적으로 다룰 질문 목록 (예: 1번 로봇이 3개의 질문을 만들어냈다면 ["질문1", "질문2", "질문3"])
    rag_answer: str                  # RAG 검색을 통해 찾아온 법령 텍스트 (규정 검토봇이 참고할 내용)
    current_agent: str               # 현재 이 가방을 쥐고 있는 로봇의 이름


def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """ 첨부한 문서를 분석하여 시스템에 반영할 수 있는 구조체를 추출한다. """

    file_path = state['file_full_path']
    
    # 문서 내용 읽기
    file_content = ""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        else:
            logger.warning(f"File not found: {file_path}")
            return {"rewrite_querys": {}}
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return {"rewrite_querys": {}}

    prompt = f"""
당신은 첨부한 공고서를 분석하여 시스템에 반영할 수 있는 구조체 값을 추출하는 전문가이다.

다음은 분석할 공고서 문서의 내용이다:

---
{file_content}
---

다음은 추출해야 할 필드 정의이다:

[
  {{
    "key": "noticeType",
    "description": "조달청 업무분류로 '물품', '공사', '일반용역', '기술용역', '비축' 중에서 하나만은 선택하여야 한다."
  }},
  {{
    "key": "noticeName",
    "description": "공고의 사업명을 의미한다."
  }}
]

위의 필드 정의에 따라 문서에서 해당 값을 추출하여 다음 JSON 객체 형식으로 반환하라.

반환 형식 (필드명:값 형태의 JSON 객체):
{{
"noticeType":"값",
"noticeName":"값"
}}

규칙:
1. 출력은 JSON 객체만 반환한다.
2. 각 필드의 값은 문서에서 직접 추출한 정확한 텍스트여야 한다.
3. description의 의미에 맞는 값을 찾아서 반환한다.
4. 불필요한 설명이나 markdown은 출력하지 않는다.
5. 만약 해당 값을 찾을 수 없다면 "정보 없음"으로 표시한다.
"""

    try:
        # LLM 호출 및 결과 처리
        response = config.get_llm().invoke(prompt)
        result_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        print("Document Analysis Result:", result_text)

        # JSON 객체 형식 추출 및 파싱
        result_data = {}
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                result_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response from LLM")
                result_data = {}
        
        return {
            "result_data": result_data
        }

    except Exception as e:
        logger.error(f"Error in query_rewriter_node: {e}")
        print(e)
        return {
            "result_data": {}
        }

async def rag_tool_call_node(state: AgentState) -> Dict[str, Any]:
    """[에이전트 2: 규정 검토봇] RAG 검색을 통해 관련 법령 텍스트를 찾아옵니다."""

    sse_client_addr = config.settings.MCP_DOC_RAG_URL # SSE 클라이언트 주소
    
    # 추출된 공고 정보를 JSON 문자열로 변환하여 시스템 메시지에 포함
    extracted_info = json.dumps(state['result_data'], ensure_ascii=False, indent=2)
    sys_message = (
        "  - The first tool: 사용자가 첨부한 문서 내용에 근거한 답변이 필요한 경우 하위 문서경로를 사용한다. \n"
        f"    - 첨부문서경로(file_full_path) = {state['file_full_path']}\n"
        f"    - 추출된 공고 정보:\n{extracted_info}"
    )

    # MCP 클라이언트를 통해 도구 호출 및 대화 프로세스 수행 (LLM과 MCP 서버 간의 반복 상호작용)
    mcpclient_manager = mcpUtils.get_mcp_manager()
    rag_answer = await mcpclient_manager.complete(sse_client_addr, sys_message, '', extracted_info, False, HumanMessage(content=extracted_info))

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
    workflow.add_node("Start Convrstn Agent", query_rewriter_node)
    workflow.add_node("Rag Tool Call Agent", rag_tool_call_node)

    # 2. 이동 경로(엣지) 연결하기
    workflow.set_entry_point("Start Convrstn Agent")                 # 시작은 무조건 파싱 전문봇이 합니다.
    workflow.add_edge("Start Convrstn Agent", "Rag Tool Call Agent") # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).
    workflow.add_edge("Rag Tool Call Agent", END)                    # 법령 검토까지 끝나면 모든 워크플로우를 마칩니다(END).


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
    #print(agent_state)
    
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
        f"- **Original Question:** {agent_state['question']}\n"
    )
    
    rag_answer = agent_state['rag_answer']  # 이전 노드에서 전달된 대화 메시지 리스트
    rag_answer.append(SystemMessage(content=final_prompt))

    return agent_state