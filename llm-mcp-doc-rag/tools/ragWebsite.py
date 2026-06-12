import mcp.types as types

from utils import config, ragUtils, loggerUtil
from retrieval import search_service
import re
import time
import asyncio
from typing import List, TypedDict
from langgraph.graph import StateGraph, END

# 로거 인스턴스 생성
logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")

# LangGraph 상태 정의
class GraphState(TypedDict):
    question: str
    sub_questions: List[str]
    documents: List[any]
    retrieved_contexts: List[dict]
    language: str

# 1. 질문 분해 노드
async def decompose_question(state: GraphState):
    logger.info("---DECOMPOSING QUESTION---")
    llm = config.get_llm()
    decomposition_prompt = f"""
    당신은 검색 전문가입니다. 사용자의 질문을 분석하여 검색 효율을 높이기 위한 하위 검색어로 분해하세요.
    - 복합적인 질문은 최대 3개의 독립적인 검색어로 나눕니다.
    - 질문이 단순하다면 원래 질문 1개만 유지합니다.
    - 출력은 반드시 다른 설명 없이 파이썬 리스트 형식만 반환하세요.
    - 예시: ["검색어1", "검색어2"]
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
    
    return {"sub_questions": sub_questions}

# 2. 웹 검색 및 문서 수집 노드
async def search_web(state: GraphState):
    logger.info("---SEARCHING WEB---")
    all_docs = []
    for sub_q in state['sub_questions']:
        improved_queries = search_service.improve_search_query(sub_q)
        docs = search_service.get_search_content(improved_queries, state['language'], 5)
        if docs:
            all_docs.extend(docs)
    return {"documents": all_docs}

# 3. RAG 실행 노드
async def execute_rag(state: GraphState):
    logger.info("---EXECUTING RAG---")
    contexts = []
    if state['documents']:
        query_engine = ragUtils.makeRagRetrieverFromDocs(state['documents'], False)
        # 각 하위 질문에 대해 검색 결과 도출
        for sub_q in state['sub_questions']:
            response = query_engine.query(sub_q)
            contexts.append({
                "type": "text",
                "text": response.response,
                "metadata": response.metadata
            })
    else:
        contexts.append({"type": "text", "text": "검색된 문서가 없습니다.", "metadata": {}})
    
    return {"retrieved_contexts": contexts}

async def execute(question: str, allow_search: bool = True, language: str = "ko"
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    start_time = time.time()

    # 검색 허용 여부 체크
    if not allow_search:
        return [types.TextContent(type="text", text="인터넷 검색이 비활성화되어 있어 정보를 검색할 수 없습니다.")]
    
    try:
        workflow = StateGraph(GraphState)
        workflow.add_node("decompose", decompose_question)
        workflow.add_node("search", search_web)
        workflow.add_node("execute_rag", execute_rag)

        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "search")
        workflow.add_edge("search", "execute_rag")
        workflow.add_edge("execute_rag", END)

        app = workflow.compile()

        initial_state = {
            "question": question,
            "sub_questions": [],
            "documents": [],
            "retrieved_contexts": [],
            "language": language
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
        logger.info("RAG Website processing completed.")