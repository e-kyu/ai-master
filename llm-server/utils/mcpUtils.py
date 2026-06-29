import uuid
import json
import re
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from langchain_core.messages import SystemMessage, ToolMessage

from utils import config

# 로거 인스턴스 생성 (./log 폴더에 llm-server 라는 이름으로 로그 기록)
logger = config.get_logger("./log", "llm-server")

# ===== 상수 정의 =====
MAX_ITERATIONS = 5      # LLM <-> 도구 호출을 반복하는 최대 횟수 (무한 루프 방지용 안전장치)
MIN_ITERATIONS = 2      # 도구 호출이 없어도 최소 이 횟수만큼은 LLM에게 다시 기회를 줌
SESSION_INIT_WAIT = 1   # MCP 세션 초기화 직후, 서버가 준비될 시간을 벌기 위한 대기 시간(초)


@asynccontextmanager
async def mcp_session(sse_client_addr):
    """
    MCP 서버에 SSE(Server-Sent Events) 방식으로 접속하고, 세션을 초기화해서
    바로 사용 가능한 `session` 객체를 돌려주는 공용 함수.

    사용 예:
        async with mcp_session(주소) as session:
            await session.list_tools()
    """
    async with sse_client(sse_client_addr) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            # 세션이 막 초기화된 직후라 서버가 완전히 준비되지 않았을 수 있으므로 잠시 대기
            await asyncio.sleep(SESSION_INIT_WAIT)
            logger.info("MCP 세션 초기화 완료")
            yield session


class MCPClientManager:
    """
    MCP(Model Context Protocol) 서버와 통신하면서,
    LLM이 필요로 하는 도구를 찾아 호출해주는 역할을 하는 클래스.
    """

    def __init__(self):
        # 도구 호출(Tool Calling) 전용 LLM 인스턴스
        self.tc_llm = config.get_tc_llm()

    def convert_tool_format(self, tools):
        """
        MCP 서버가 알려준 도구 스키마를, LangChain/OpenAI가 이해할 수 있는
        Function Calling 형식으로 한 개씩 변환해서 리스트로 돌려준다.
        :param tools: MCP 서버로부터 받은 도구 목록
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    async def call_tool(self, session, response, messages):
        """
        LLM이 요청한 도구(tool_calls)들을 하나씩 실제로 호출하고,
        그 결과를 ToolMessage로 만들어 messages 리스트에 추가하는 함수.
        :param session: 활성화된 MCP 클라이언트 세션
        :param response: LLM의 응답 객체 (tool_calls 포함)
        :param messages: 현재까지의 대화 메시지 리스트 (이 함수 안에서 직접 추가됨)
        """
        messages.append(response)

        tool_calls = response.tool_calls
        if not tool_calls:
            return messages

        logger.info(f"호출할 도구 목록: {tool_calls}")

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                # MCP 세션을 통해 실제 도구 호출 실행
                tool_response = await session.call_tool(tool_name, tool_args)
                tool_result = {
                    "toolUseId": f"{tool_name}-{uuid.uuid4()}",
                    "content": [{"text": tool_response.content[0].text}],
                }
            except Exception as err:
                # 도구 호출이 실패해도 전체 흐름이 멈추지 않도록, 에러 내용을 결과에 담아 전달
                logger.error(f"도구 호출 실패: {tool_name} - {err}")
                tool_result = {
                    "toolUseId": tool_call["id"],
                    "content": [{"text": f"오류: {str(err)}"}],
                    "status": "error",
                }

            # 도구 호출 결과를 LLM이 이어서 읽을 수 있도록 메시지 목록에 추가
            messages.append(
                ToolMessage(
                    content=str({"toolResult": tool_result}),
                    tool_call_id=tool_call["id"],
                )
            )

        return messages

    def get_actual_tool_calls(self, response):
        """
        LLM 응답에서 도구 호출 정보를 추출하는 함수.

        - LangChain이 자동으로 `response.tool_calls`를 채워주면 그대로 사용한다.
        - Ollama처럼 일부 모델은 자동 인식이 안 되고, 답변 텍스트 안에
          JSON 배열 형태로 도구 호출 정보를 적어주는 경우가 있는데,
          이때는 정규식으로 JSON 부분만 찾아서 직접 파싱한다.
        :param response: LLM 응답 객체
        :return: tool_call 딕셔너리 리스트 / 못 찾으면 None 또는 빈 리스트
        """
        if not response.content:
            return []

        # 1. LangChain이 자동으로 인식한 경우 (Native)
        if getattr(response, "tool_calls", None):
            logger.info(f"[도구 호출 자동 인식] {len(response.tool_calls)}개")
            return response.tool_calls

        logger.info("[도구 호출 없음] 텍스트에서 JSON 패턴을 직접 추출합니다.")

        # 2. 텍스트 안에 "[{...}, {...}]" 형태의 JSON 배열이 섞여 있는 경우 (수동 파싱)
        content = response.content.strip()
        match = re.search(r"\[[\s\S]*?\]", content)
        if not match:
            return None

        try:
            raw_calls = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        return [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",  # tool_call["id"] 자리를 채워줌 (KeyError 방지)
                "name": data.get("name"),
                "args": data.get("parameters") or data.get("args") or {},
            }
            for data in raw_calls
        ]

    async def converse_mcp(self, session, messages, tools):
        """
        LLM 호출 -> (필요하면) 도구 호출 -> 결과 반영을, 정해진 횟수만큼 반복하는 메인 루프.

        반복 종료 조건:
        - 도구 호출이 한 번이라도 있으면, 그 호출을 처리한 뒤 즉시 종료한다.
        - 도구 호출이 없으면 MIN_ITERATIONS번까지는 LLM에게 다시 시도할 기회를 준다.
        - 어떤 경우든 MAX_ITERATIONS를 넘기지 않는다 (무한 루프 방지).
        :param session: MCP 세션
        :param messages: 대화 이력
        :param tools: 사용 가능한 MCP 도구 목록
        """
        converted_tools = self.convert_tool_format(tools)

        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(f"[반복 {iteration}] LLM 호출 중...")

            # LLM에 도구를 바인딩하여 호출
            llm_with_tools = self.tc_llm.bind_tools(converted_tools)
            response = llm_with_tools.invoke(messages)

            # LLM이 도구 호출을 자동으로 인식하지 못했다면, 응답 텍스트에서 직접 추출 시도
            if not getattr(response, "tool_calls", None):
                response.tool_calls = self.get_actual_tool_calls(response)

            if response.tool_calls:
                await self.call_tool(session, response, messages)

                # TODO: 도구 호출 결과를 분석해 추가로 호출할 도구가 있는지 판단하는
                #       로직을 넣으면, 다단계 검색/추론(예: 추가 법령 검색, 웹검색)도 가능해진다.
                #       다만 무한 반복을 막기 위해 MAX_ITERATIONS는 항상 지켜야 한다.
                break

            if iteration >= MIN_ITERATIONS:
                logger.info("호출할 도구가 없어 TOOL CALL을 종료합니다.")
                break
        else:
            logger.info("최대 반복 횟수에 도달하여 TOOL CALL을 종료합니다.")

        return messages

    def _build_tool_selection_prompt(self, tools, convrstn_context, original_question, sys_message):
        """
        "이 질문에 어떤 도구를 호출해야 하는가?"를 LLM이 판단하도록 안내하는
        시스템 프롬프트를 만들어주는 함수. complete()에서 분리해서
        프롬프트만 따로 수정하기 쉽게 했다.
        """
        tools_list = [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
            for tool in tools
        ]
        tools_list_str = json.dumps(tools_list, ensure_ascii=False)
        today_str = datetime.now().strftime("%Y-%m-%d")

        return (
            "# ROLE\n"
            "You are a Tool Selection Specialist. Analyze the context and query to call the necessary tools.\n\n"
            "# RULES\n"
            f"- Today's date is {today_str}.\n"
            "- Use conversation history to resolve pronouns (it, that, etc.) into specific terms.\n"
            "- Use the exact tool name as defined in the TOOLS section without any prefixes like 'functions.'.\n"
            "- Call multiple tools in one array if needed.\n"
            "- Output ONLY the JSON array or 'NONE'. No conversational filler.\n\n"
            "# TOOLS\n"
            f"{tools_list_str}\n\n"
            "# EXAMPLES\n"
            "1. Follow-up Query\n"
            "   - Context: \"User asked about Apple's 2025 revenue.\"\n"
            "   - User: \"Then what about Microsoft?\"\n"
            "   - Assistant: [{\"name\":\"internet_search_tool\",\"parameters\":{\"query\": \"Microsoft 2025 revenue\"}}]\n\n"
            "2. Ambiguous Reference\n"
            "   - Context: \"User and Assistant discussed the 'Project Alpha' PDF.\"\n"
            "   - User: \"Summary of the budget section in that doc.\"\n"
            "   - Assistant: [{\"name\":\"document_rag_tool\",\"parameters\":{\"question\": \"Summary of the budget section in Project Alpha document\"}}]\n\n"
            "3. General Greeting\n"
            "   - User: \"Hi\"\n"
            "   - Assistant: NONE\n\n"
            "# INPUT DATA\n"
            f"- History: {convrstn_context}\n"
            f"- Query: {original_question}\n"
            f"- Extra Info: {sys_message}\n\n"
            "# FINAL JSON OUTPUT:\n"
        )

    async def complete(self, sse_client_addr, sys_message, convrstn_context, original_question, enableExtDocse, human_message):
        """
        MCP 서버에 연결해서, 어떤 도구를 호출해야 하는지 LLM이 판단하고
        실제로 호출까지 수행한 결과가 담긴 messages를 반환하는 전체 흐름.
        :param sse_client_addr: MCP 서버 주소 (SSE 엔드포인트)
        :param sys_message: 시스템 프롬프트에 덧붙일 추가 지시사항 (예: 문서 경로)
        :param convrstn_context: 이전 대화를 요약한 맥락 정보
        :param original_question: 사용자의 원본 질문
        :param enableExtDocse: 외부 문서 검색 사용 여부 (현재는 로깅 용도로만 사용)
        :param human_message: 강화된 질의가 담긴 HumanMessage 객체
        """
        messages = []

        if not sse_client_addr:
            logger.info("SSE 클라이언트 주소가 설정되지 않았습니다.")
            return messages

        async with mcp_session(sse_client_addr) as session:
            # 1. MCP 서버가 제공하는 도구 목록 조회
            tools_result = await session.list_tools()

            # 2. "어떤 도구를 호출할지" 판단시키기 위한 시스템 프롬프트 + 사용자 질문 구성
            prompt = self._build_tool_selection_prompt(
                tools_result.tools, convrstn_context, original_question, sys_message
            )
            messages.append(SystemMessage(content=prompt))
            messages.append(human_message)

            # 3. LLM <-> 도구 호출 반복 수행
            await self.converse_mcp(session, messages, tools_result.tools)

            if enableExtDocse:
                logger.info(f"enableExtDocse: {enableExtDocse}")

        return messages


def get_mcp_manager():
    """MCPClientManager 인스턴스를 생성해서 반환하는 함수"""
    return MCPClientManager()


def call_tool(sse_client_addr, payload):
    """
    동기(sync) 코드에서 MCP 도구를 한 번만 호출하고 싶을 때 사용하는 헬퍼 함수.

    내부 로직은 비동기(async)지만, sse_client/ClientSession이 async context
    manager라서 asyncio.run()으로 감싸 동기 함수처럼 호출할 수 있게 했다.
    :param sse_client_addr: MCP 서버 주소
    :param payload: {"tool": 도구 이름, "input": 도구에 전달할 인자} 형태의 딕셔너리
    :return: {"status": "success", "response": ...} 또는
             {"status": "failed", "error_message": ...}
    """
    async def _call_tool_async():
        try:
            async with mcp_session(sse_client_addr) as session:
                tool_response = await session.call_tool(payload["tool"], payload["input"])
                return {
                    "status": "success",
                    "response": tool_response.content[0].text,
                }
        except Exception as err:
            return {
                "status": "failed",
                "error_message": f"MCP 도구 호출 중 오류 발생: {str(err)}",
            }

    return asyncio.run(_call_tool_async())
