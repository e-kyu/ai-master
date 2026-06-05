import os
# OpenMP 중복 로드 에러를 무시하고 프로그램을 계속 실행하도록 설정합니다.
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import mcp.types as types
from utils import ragUtils, config, loggerUtil
import re
import time
import asyncio

# 로거 인스턴스 생성
logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")

async def execute(question: str, pdfFileFullPath: str = "" , agent_mode: str = "" ,allow_search: bool = False) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    start_time = time.time()

    # 1. 복합 질문을 개별 질문으로 분해 (Query Decomposition)
    llm = config.get_llm()
    decomposition_prompt = f"""
    사용자의 질문을 분석하여, 문서에서 정보를 찾기 위한 구체적이고 독립적인 하위 질문들로 나누어 주세요.
    질문이 단순하다면 하나만 유지하세요.
    결과는 반드시 파이썬 리스트 형식으로만 출력하세요. (예: ["질문1", "질문2"])
    
    사용자 질문: {question}
    """
    
    try:
        # 동기 LLM 호출로 변경
        decomp_res_obj = llm.complete(decomposition_prompt)
        decomp_res = decomp_res_obj.text
        # 리스트 형태의 문자열에서 질문 추출
        sub_questions = re.findall(r'"([^"]*)"', decomp_res)
        if not sub_questions:
            sub_questions = [question]
    except Exception as e:
        logger.error(f"Decomposition error: {e}")
        sub_questions = [question]

    try:
        logger.info(f"question: {question}")
        logger.info(f"pdfFileFullPath: {pdfFileFullPath}")
        logger.info(f"sub_questions: {str(sub_questions)}")
        
        query_engines = []
        
        if agent_mode:
            query_engine_list = config.get_query_engine(agent_mode)
            if query_engine_list:
                query_engines.extend(query_engine_list)

        if pdfFileFullPath:
            # 1. LlamaIndex Query Engine 생성 (문서 로드, 파싱, 인덱싱 및 검색기 구성)
            pdf_query_engine = ragUtils.make_rag_query_engine(pdfFileFullPath, is_save=True, is_base_resource=False)
            query_engines.append(pdf_query_engine)

        # 2. 각 하위 질문에 대해 순차적 RAG 수행
        retrieved_contexts = []

        for query_engine in query_engines:
            # aquery는 비동기 쿼리 메서드입니다.
            tasks = [query_engine.aquery(sub_q) for sub_q in sub_questions]
            
            # 모든 태스크가 완료될 때까지 기다립니다.
            responses = await asyncio.gather(*tasks)

            for res in responses:
                logger.info(res.response)
                
                retrieved_contexts.append({
                    "type": "text",
                    "text": res.response,
                    "metadata": res.metadata
                    })

            """ # 비동기 로직을 반영하여 동기 로직은 주석처리
            for sub_q in sub_questions:
                logger.info(f"sub_questions: {str(sub_q)}")
                responses = query_engine.query(sub_q)

                retrieved_contexts.append({
                    "type": "text",
                    "text": responses.response,
                    "metadata": responses.metadata
                    })
            """
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"Final Response (Processing Time: {elapsed_time:.2f}s): {str(retrieved_contexts)}")
        return [types.TextContent(type="text", text=str(retrieved_contexts))]

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    
    finally:
        logger.info("RAG processing completed.")   
