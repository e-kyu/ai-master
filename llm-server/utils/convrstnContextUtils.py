import json
import re
from utils import config
from repository import convrstn_repository
import logging
from datetime import datetime

def get_convrstn_context(convrstn_id: str, question: str) -> str:
    """
    기존 대화 맥락을 조회하거나, 없는 경우 최근 대화 내역을 바탕으로 LLM을 통해 맥락을 요약하여 반환합니다.
    """

    # 1. 최근 대화 내역 조회 (최대 6개)
    convrstn_details = convrstn_repository.read_recent_convrstn(convrstn_id, limit=6)

    # 2. 대화 이력이 6회 이상인 경우에만 기존 저장된 맥락 조회
    existing_context = ""
    if len(convrstn_details) > 1:
        existing_context = convrstn_repository.read_context_convrstn(convrstn_id)

    # 3. LLM 프롬프트 구성을 위한 대화 이력 포맷팅
    history_str = ""
    if existing_context:
        history_str = (
                f"{existing_context}"
                "\n\n"
                "- **History of last conversation:**\n"
                f"{[f"Q: {r.question}\nA: {r.answer}" for r in [convrstn_details[-1]]]}"
            )
    else:
        history_str = "\n".join([f"Q: {r.question}\nA: {r.answer}" for r in convrstn_details])

    prompt = ("""
                # Role
                너는 대화 맥락 유지 여부를 판별하고 요약하는 지능형 어시스턴트이다.

                # Task
                아래 순서에 따라 작업을 수행하라:
                1. **연관성 평가:** [Current User Query]가 [Past Conversation History]의 주제, 목적, 혹은 논의 흐름과 관련이 있는지 분석함.
                2. **결정:** 
                - 만약 주제가 완전히 바뀌었거나 이전 맥락이 현재 질문 이해에 도움이 되지 않는다면 **공백(Empty String)**만 반환함.
                - 흐름이 이어진다면 아래 [Output Format]에 맞춰 JSON 형식으로 요약함.

                # Output Format (If connected)
                {"topic": "주제", "context": "핵심 요점", "current_goal": "사용자 의도", "status": "해결 및 남은 과제"}

                # Constraints
                - 모든 내용은 명사형 종결어미를 사용하며 1000토큰 이내로 작성.
                - 주제 전환 시 설명 없이 오직 **공백**만 출력할 것.
              """
              f"""
                # Input Data
                - **Past Conversation History:**
                {history_str}

                - **Current User Query:**
                {question}

                # Summary Output:
              """
             )

    try:
        # 4. LLM 호출 및 결과 처리
        response = config.get_llm().invoke(prompt)
        new_context = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        print("Conversation context: " + new_context)

        # JSON 형식 추출 및 파싱
        context_data = {}
        json_match = re.search(r'\{.*\}', new_context, re.DOTALL)
        if json_match:
            try:
                context_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # 5. 생성된 맥락이 있다면 DB 업데이트
        if context_data:
            convrstn_repository.update_convrstn_context(convrstn_id, context_data)

        
        return context_data
    except Exception as e:
        logging.error(f"Error generating conversation context: {e}")
        return ""

def enhance_query_with_context(convrstn_context: str, question: str) -> str:
    """
    이전 대화 맥락을 바탕으로 사용자의 질문을 구체화된 질의문으로 재작성합니다.

    Args:
        convrstn_context (str): 요약된 이전 대화 맥락
        question (str): 사용자의 현재 질문

    Returns:
        str: 대명사 복원 및 의도가 보완된 강화된 질의문
    """
    
    today_str = datetime.now().strftime("%Y년 %m월 %d일")

    contextual_query_enhancer_prompt = (
        "# ROLE\n"
        "Search Query Optimizer: Rewrite the user's query into a standalone, descriptive search term using the provided context.\n\n"
        "# INSTRUCTIONS\n"
        f"1. **Reference Date:** Today is {today_str}. Convert relative dates (today, tomorrow) to absolute dates.\n"
        "2. **De-reference:** Replace pronouns (it, they, that, there) with the specific entities mentioned in the [Conversation Summary].\n"
        "3. **Completeness:** If the query is fragmented, complete it so it can be understood without the history.\n"
        "4. **Independence:** If the query is a new topic, do not force context; output the original query.\n\n"
        "# CONSTRAINTS\n"
        "- Output ONLY the rewritten query text.\n"
        "- No conversational filler, no labels (e.g., 'Enhanced:'), no explanations.\n"
        "- Maintain the user's original language.\n\n"
        "# EXAMPLES\n"
        "- Context: Discussing Samsung's 2024 earnings.\n"
        "- User: \"What about operating profit?\"\n"
        "- Output: Samsung's 2024 operating profit\n\n"
        "- Context: Planning a trip to Seoul.\n"
        "- User: \"How's the weather tomorrow?\"\n"
        f"- Output: Weather in Seoul on (Calculated Date from {today_str})\n\n"
        "# INPUT DATA\n"
        f"- [Conversation Summary]: {convrstn_context}\n"
        f"- [Current User Query]: {question}\n\n"
        "# FINAL REWRITTEN QUERY:"
    )

    response = config.get_llm().invoke(contextual_query_enhancer_prompt)
    enhanced_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    print("Enhanced Query: " + enhanced_query)

    return enhanced_query