import os
# OpenMP 중복 로드 에러를 무시하고 프로그램을 계속 실행하도록 설정합니다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import mcp.types as types
from utils import rag_utils, config, logger_util
import re
import time
import asyncio
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END

# 로거 인스턴스 생성
logger = logger_util.get_logger("./log", "llm-mcp-doc-rag")

# LangGraph 상태 정의
class GraphState(TypedDict):
    question: str
    pdf_path: str
    agent_mode: str
    sub_questions: List[str]
    query_engines: List[any]
    retrieved_contexts: List[dict]

# 1. 질문 분해 노드
async def decompose_question(state: GraphState):
    logger.info("---DECOMPOSING QUESTION---")
    llm = config.get_llm()
    decomposition_prompt = f"""
    사용자의 질문을 분석하여, 문서에서 정보를 찾기 위한 구체적이고 독립적인 하위 질문들로 나누어 주세요.
    질문이 단순하다면 하나만 유지하세요.
    결과는 반드시 파이썬 리스트 형식으로만 출력하세요. (예: ["질문1", "질문2"])
    사용자 질문: {state['question']}
    """
    try:
        decomp_res_obj = llm.complete(decomposition_prompt)
        sub_questions = re.findall(r'"([^"]*)"', decomp_res_obj.text)
        if not sub_questions:
            sub_questions = [state['question']]
    except Exception as e:
        logger.error(f"Decomposition error: {e}")
        sub_questions = [state['question']]
    
    logger.info(f"Decomposition failed, using original question: {state['question']}")
    logger.info(f"Decomposition failed, using original question: {sub_questions}")
    return {"sub_questions": sub_questions}

# 2. 엔진 준비 노드
async def prepare_engines(state: GraphState):
    logger.info("---PREPARING ENGINES---")
    engines = []
    if state['agent_mode']:
        engine_list = config.get_query_engine(state['agent_mode'])
        if engine_list:
            engines.extend(engine_list)
    
    if state['pdf_path']:
        pdf_engine = rag_utils.make_rag_query_engine(state['pdf_path'], is_save=True, is_base_resource=False)
        engines.append(pdf_engine)
    
    return {"query_engines": engines}

"""
# 3. RAG 실행 노드
async def execute_rag(state: GraphState):
    logger.info("---EXECUTING RAG---")
    # 엔진별로 순차 처리(for engine ...: await gather(...))하면, 한 모드에 엔진이
    # 여러 개 등록된 경우(세미콜론으로 구분된 다중 리소스, 예: PpsStockpilingAgent의
    # Guide.zip + Agent.zip) 지연시간이 엔진 개수만큼 그대로 누적된다. (엔진, 하위질문)
    # 조합을 전부 한 번에 gather로 병렬 실행해 전체 지연시간을 가장 느린 호출 1건 수준으로 줄인다.
    tasks = [
        engine.aquery(sub_q)
        for engine in state['query_engines']
        for sub_q in state['sub_questions']
    ]
    responses = await asyncio.gather(*tasks)

    contexts = [
        {
            "type": "text",
            "text": res.response,
            "metadata": res.metadata,
        }
        for res in responses
    ]
    return {"retrieved_contexts": contexts}
"""

# 3. RAG 실행 노드
async def execute_rag(state: GraphState):
    logger.info("---EXECUTING RAG---")
    # 엔진별로 순차 처리(for engine ...: await gather(...))하면, 한 모드에 엔진이
    # 여러 개 등록된 경우(세미콜론으로 구분된 다중 리소스, 예: PpsStockpilingAgent의
    # Guide.zip + Agent.zip) 지연시간이 엔진 개수만큼 그대로 누적된다. (엔진, 하위질문)
    # 조합을 전부 한 번에 gather로 병렬 실행해 전체 지연시간을 가장 느린 호출 1건 수준으로 줄인다.
    tasks = [
        engine.aquery(sub_q)
        for engine in state['query_engines']
        for sub_q in state['sub_questions']
    ]
    responses = await asyncio.gather(*tasks)

    contexts = [
        {
            "type": "text",
            "text": res.response,
            "metadata": res.metadata,
        }
        for res in responses
    ]
    return {"retrieved_contexts": contexts}

"""
# 3. RAG 실행 노드
async def execute_rag(state: GraphState):
    logger.info("---EXECUTING RAG---")
    contexts = []
    for engine in state['query_engines']:
        tasks = [engine.aquery(sub_q) for sub_q in state['sub_questions']]
        responses = await asyncio.gather(*tasks)
        for res in responses:
            contexts.append({
                "type": "text",
                "text": res.response,
                "metadata": res.metadata
            })
    return {"retrieved_contexts": contexts}
"""

async def execute(question: str, pdf_file_full_path: str = "" , agent_mode: str = "" ,allow_search: bool = False) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    start_time = time.time()
    try:        
        # 워크플로우 그래프 구성
        workflow = StateGraph(GraphState)
        
        workflow.add_node("decompose", decompose_question)
        workflow.add_node("prepare_engines", prepare_engines)
        workflow.add_node("execute_rag", execute_rag)
        
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "prepare_engines")
        workflow.add_edge("prepare_engines", "execute_rag")
        workflow.add_edge("execute_rag", END)
        
        app = workflow.compile()

        initial_state = {
            "question": question,
            "pdf_path": pdf_file_full_path,
            "agent_mode": agent_mode,
            "sub_questions": [],
            "query_engines": [],
            "retrieved_contexts": []
        }
        
        final_state = await app.ainvoke(initial_state)
        retrieved_contexts = final_state["retrieved_contexts"]

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"Final Response (Processing Time: {elapsed_time:.2f}s): {str(retrieved_contexts)}")
        return [types.TextContent(type="text", text=str(retrieved_contexts))]

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    
    finally:
        logger.info("RAG processing completed.")   
