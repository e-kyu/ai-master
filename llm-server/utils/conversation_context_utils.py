import json
import re
from typing import Optional
from utils import config
from repository import conversation_repository
from datetime import datetime

logger = config.get_logger("./log", "llm-server")

def get_conversation_context(conversation_id: str, question: str, rlog: Optional[object] = None) -> str:
    """
    기존 대화 맥락을 조회하거나, 없는 경우 최근 대화 내역을 바탕으로 LLM을 통해 맥락을 요약하여 반환합니다.
    """
    logger.info(f"[ConversationContext] 맥락 조회 시작 conversation_id={conversation_id} question={question[:50]}")

    # 1. 최근 대화 내역 조회 (최대 6개)
    conversation_details = conversation_repository.read_recent_conversation(conversation_id, limit=6)
    logger.debug(f"[ConversationContext] 최근 대화 내역 {len(conversation_details)}건 조회 conversation_id={conversation_id}")

    # 2. 대화 이력이 6회 이상인 경우에만 기존 저장된 맥락 조회
    existing_context = ""
    if len(conversation_details) > 1:
        existing_context = conversation_repository.read_context_conversation(conversation_id)
        logger.debug(f"[ConversationContext] 기존 맥락 {'존재' if existing_context else '없음'} conversation_id={conversation_id}")

    # 3. LLM 프롬프트 구성을 위한 대화 이력 포맷팅
    history_str = ""
    if existing_context:
        history_str = (
                f"{existing_context}"
                "\n\n"
                "- **History of last conversation:**\n"
                f"{[f"Q: {r.question}\nA: {r.answer}" for r in [conversation_details[-1]]]}"
            )
    else:
        history_str = "\n".join([f"Q: {r.question}\nA: {r.answer}" for r in conversation_details])

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

    model = config.describe_llm_model()
    try:
        # 4. LLM 호출 및 결과 처리
        logger.debug(f"[ConvrstnContext] 맥락 요약 LLM 호출 conversation_id={conversation_id}")
        if rlog:
            rlog.llm_call(
                model, "이전 대화 맥락 연관성 판별 및 JSON 요약",
                prompt_preview=f"최근 대화 {len(conversation_details)}건 vs 현재 질문: {question}",
                params={"history_count": len(conversation_details)},
            )
        response = config.get_llm().invoke(prompt)
        new_context = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        logger.debug(f"[ConvrstnContext] 맥락 요약 LLM 응답 conversation_id={conversation_id} response={new_context[:200]}")

        # JSON 형식 추출 및 파싱
        context_data = {}
        json_match = re.search(r'\{.*\}', new_context, re.DOTALL)
        if json_match:
            try:
                context_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.error(f"[ConvrstnContext] 맥락 JSON 파싱 실패 conversation_id={conversation_id} raw={new_context[:200]}")

        # 5. 생성된 맥락이 있다면 DB 업데이트
        if context_data:
            conversation_repository.update_conversation_context(conversation_id, context_data)
            logger.info(f"[ConversationContext] 맥락 갱신 완료 conversation_id={conversation_id} topic={context_data.get('topic')}")
            if rlog:
                rlog.llm_response(
                    model,
                    f"대화 흐름이 이어진다고 판단 — topic=\"{context_data.get('topic', '')}\", current_goal=\"{context_data.get('current_goal', '')}\"",
                    output_preview=new_context,
                )
        else:
            logger.info(f"[ConversationContext] 주제 전환으로 판단되어 맥락을 비웁니다 conversation_id={conversation_id}")
            if rlog:
                rlog.llm_response(model, "이전 대화와 무관한 새 주제로 판단 — 맥락을 비움(empty)", output_preview=new_context)

        return context_data
    except Exception as e:
        logger.error(f"[ConversationContext] 맥락 생성 중 오류 발생 conversation_id={conversation_id}: {e}")
        return ""

def enhance_query_with_context(conversation_context: str, question: str, rlog: Optional[object] = None) -> str:
    """
    이전 대화 맥락을 바탕으로 사용자의 질문을 구체화된 질의문으로 재작성합니다.

    Args:
        conversation_context (str): 요약된 이전 대화 맥락
        question (str): 사용자의 현재 질문

    Returns:
        str: 대명사 복원 및 의도가 보완된 강화된 질의문
    """

    logger.info(f"[ConversationContext] 질의 강화 시작 question={question[:50]}")
    today_str = datetime.now().strftime("%Y년 %m월 %d일")

    contextual_query_enhancer_prompt = (
        "# 역할\n"
        "검색 쿼리 최적화 전문가: 제공된 컨텍스트(대화 맥락)를 활용하여 사용자의 질문을 독립적이고 서술적인 단일 검색어로 재작성합니다.\n\n"
        "# 지침\n"
        f"1. **기준 날짜:** 오늘은 {today_str}입니다. 상대적인 날짜(오늘, 내일, 이번 주 등)는 모두 절대적인 날짜(연-월-일)로 변환하세요.\n"
        "2. **지시어 제거:** 대명사나 지시어(그것, 그들, 저기, 그때 등)는 [대화 맥락]에 언급된 구체적인 개체명이나 명사로 대체하세요.\n"
        "3. **업무명확성:** 일반용역, 기술용역, 시설공사, 물품, 비축 등 [대화 맥락]에서 선택되거나 언급된 구체적인 업무 분야가 있다면 질의문에 반드시 결합하세요.\n"
        "4. **생략된 질문 복원 (핵심):** 사용자가 에이전트의 유도에 따라 '시설공사야', '일반용역' 등 단답형으로 대답한 경우, 이는 이전 대화에서 해결되지 않은 질문(예: '적격심사가 뭐야?')에 대한 구체화입니다. 이전 질문의 핵심 키워드와 현재 답변을 결합하여 완성된 질문 형태로 만드세요.\n"
        "5. **완전성:** 질문이 단편적이거나 생략되어 있다면, 이전 대화 기록 없이도 그 자체로 이해할 수 있도록 완전한 문장이나 키워드로 보완하세요.\n"
        "6. **독립성:** 사용자의 질문이 이전 대화와 관계없는 새로운 주제라면, 억지로 맥락을 붙이지 말고 사용자의 원본 질문을 그대로 출력하세요.\n\n"
        "# 제약 조건\n"
        "- 오직 재작성된 쿼리 텍스트'만' 출력하세요.\n"
        "- 서론/결론, 미사여구, 라벨(예: '최적화된 쿼리:'), 설명 등을 절대 포함하지 마세요.\n"
        "- 사용자가 대화에서 사용한 언어(예: 한국어, 영어 등)를 그대로 유지하세요.\n\n"
        "# 예시\n"
        "- 컨텍스트: USER가 \"적격심사가 뭐야?\"라고 묻자, AGENT가 업무 분야(일반용역, 시설공사 등)를 선택하라고 유도함.\n"
        "- 사용자: \"시설공사야\"\n"
        "- 출력: 시설공사 적격심사란?\n\n"
        "- 컨텍스트: 삼성전자의 2024년 실적에 대해 논의 중.\n"
        "- 사용자: \"영업이익은 어때?\"\n"
        "- 출력: 삼성전자 2024년 영업이익\n\n"
        "- 컨텍스트: 서울 여행 계획을 세우는 중.\n"
        "- 사용자: \"내일 날씨 어때?\"\n"
        f"- 출력: ({today_str} 기준으로 계산된 내일 날짜) 서울 날씨\n\n"
        "# 입력 데이터\n"
        f"- [대화 맥락]: {conversation_context}\n"
        f"- [현재 사용자 질문]: {question}\n\n"
        "# 최종 재작성된 쿼리:"
    )

    model = config.describe_llm_model()
    if rlog:
        rlog.llm_call(
            model, "대명사 복원 및 의도 보완을 통한 검색 질의 재작성",
            prompt_preview=f"원본 질문: \"{question}\" / 대화 맥락: {conversation_context or '(없음)'}",
        )

    response = config.get_llm().invoke(contextual_query_enhancer_prompt)
    enhanced_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    logger.info(f"[ConversationContext] 질의 강화 완료 enhanced_query={enhanced_query[:100]}")
    if rlog:
        changed = enhanced_query.strip() != question.strip()
        rlog.llm_response(
            model,
            f'질의 재작성 {"적용됨" if changed else "변경 없음(원본 유지)"} → "{enhanced_query}"',
        )

    return enhanced_query