import mcp.types as types

from utils import config, ragUtils, loggerUtil
from retrieval import search_service
import re
import time

# 로거 인스턴스 생성
logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")

async def execute(question: str, allow_search: bool = True, language: str = "ko"
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    start_time = time.time()
    logger.info(f"Starting ragWebsite execution for question: {question}")

    # 검색 허용 여부 체크
    if not allow_search:
        return [types.TextContent(type="text", text="인터넷 검색이 비활성화되어 있어 정보를 검색할 수 없습니다.")]

    # 단계 1: 복잡한 질문을 여러 개의 작은 질문으로 나누기 (Query Decomposition)
    llm = config.get_llm()
    decomposition_prompt = f"""
    당신은 검색 전문가입니다. 사용자의 질문을 분석하여 검색 효율을 높이기 위한 하위 검색어로 분해하세요.
    - 복합적인 질문은 최대 3개의 독립적인 검색어로 나눕니다.
    - 질문이 단순하다면 원래 질문 1개만 유지합니다.
    - 출력은 반드시 다른 설명 없이 파이썬 리스트 형식만 반환하세요.
    - 예시: ["검색어1", "검색어2"]

    사용자 질문: {question}
    """

    try:
        # LLM 호출
        decomp_res_obj = llm.complete(decomposition_prompt)
        decomp_res = decomp_res_obj.text
        # 결과에서 따옴표 안에 있는 질문들만 뽑아내기
        sub_questions = re.findall(r'"([^"]*)"', decomp_res)
        if not sub_questions:
            sub_questions = [question]
        logger.info(f"Decomposed sub-questions: {sub_questions}")
    except Exception as e:
        logger.error(f"Decomposition error: {e}")
        sub_questions = [question]

    # TODO: 현재는 질문 분해 기능은 속도가 너무 느려서 일단 전체 질문을 하나의 검색어로 사용. 향후 개선 필요
    #sub_questions = [question]

    try:
        documents = []
        # 단계 2: 나누어진 각 질문들에 대해 문서에서 정답 찾아보기
        retrieved_contexts = []
        for sub_q in sub_questions:
            
            # 단계 2-1: 검색 엔진에서 더 잘 찾을 수 있도록 검색어 다듬기
            improved_queries = search_service.improve_search_query(sub_q)
            logger.info(f"Improved search queries: {improved_queries}")

            # 단계 2-2: 실제로 인터넷 검색을 수행하고 웹페이지 내용 가져오기
            document = search_service.get_search_content(improved_queries, language, 5)
            logger.info(f"검색된 문서 수: {len(document)}")

            if not document:
                logger.warning(f"No documents retrieved for sub-question: {sub_q}")
                continue

            documents.extend(document)
        
        logger.info(f"검색된 총 문서 수: {len(documents)}")
        if not documents:
            # 단계 2-3: 가져온 웹 문서들을 검색 가능한 데이터베이스(인덱스)로 만들기
            query_engine = ragUtils.makeRagRetrieverFromDocs(documents, False)
            
            responses = query_engine.query(sub_q)

            retrieved_contexts.append({
                    "type": "text",
                    "text": responses.response,
                    "metadata": responses.metadata
                })

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.info(f"Final Response (Processing Time: {elapsed_time:.2f}s): {str(retrieved_contexts)}")
        return [types.TextContent(type="text", text=str(retrieved_contexts))]

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise