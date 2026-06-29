import uuid
import json
import re
import asyncio
import logging
from datetime import datetime
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from langchain_core.messages import SystemMessage, ToolMessage
from utils import config

# 로거 인스턴스 생성
logger = config.get_logger("./log", "llm-server")

# 상수 정의
MAX_ITERATIONS = 5
MIN_ITERATIONS = 2
SESSION_INIT_WAIT = 1

class ToolCallError(Exception):
    """도구 호출 중 발생하는 커스텀 예외"""

class MCPClientManager:
    """
    MCP(Model Context Protocol) 클라이언트를 관리하고 LLM과의 도구 호출 상호작용을 처리하는 클래스
    """
    def __init__(self):
        # 도구 호출(Tool Calling) 전용 LLM 인스턴스 초기화
        self.tc_llm = config.get_tc_llm()

    def convert_tool_format(self, tools):
        """
        MCP 서버에서 제공하는 도구 스키마를 LangChain 및 OpenAI 호환 형식(Function Calling)으로 변환하는 함수
        :param tools: MCP 서버로부터 받은 도구 목록
        """
        converted_tools = []

        for tool in tools:
            converted_tool = {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.inputSchema
                }
            }

            converted_tools.append(converted_tool)

        return converted_tools
    

    async def call_tool(self, session, response, messages):
        """
        LLM이 요청한 도구를 호출하고, 결과를 다시 LLM에 전달하는 함수
        :param session: 활성화된 MCP 클라이언트 세션
        :param response: LLM의 응답 객체 (tool_calls 포함)
        :param messages: 현재까지의 대화 메시지 리스트
        """
    
        messages.append(response)
        
        # 도구 사용 요청. 도구를 호출하고 결과를 모델에 전송합니다.
        tool_calls = response.tool_calls
        if not tool_calls:
            return messages
        
        print("tool_calls: " + str(tool_calls))

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            try:
                # MCP 세션을 통해 도구 호출
                tool_response = await session.call_tool(tool_name, tool_args)
                
                # 도구 응답을 예상 형식으로 변환
                tool_result = {
                    "toolUseId": tool_name + str(uuid.uuid4()),
                    "content": [{"text": tool_response.content[0].text}]
                }

            except Exception as err:
                logger.error(f"도구 호출 실패: {tool_name} - {err}")
                tool_result = {
                    "toolUseId": tool_call["id"],
                    "content": [{"text": f"오류: {str(err)}"}],
                    "status": "error"
                }

            # 도구 결과를 메시지에 추가
            messages.append(ToolMessage(
                content=str({"toolResult": tool_result}),
                tool_call_id=tool_call["id"]
            ))

        return messages
    
    def get_actual_tool_calls(self, response):
        """
        자동으로 tools_calls를 인식하지 못하는 경우, 응답에서 JSON 패턴을 추출하여 도구 호출 정보를 얻는 함수
        Ollama 등 일부 모델이 JSON 형식으로 도구 호출을 반환할 때 수동 파싱을 수행
        :param response: LLM 응답 객체
        """
        if not response.content: return []

        # 1. Ollama가 자동으로 인식한 경우 (Native)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[도구 호출 자동 인식] {len(response.tool_calls)}개의 도구 호출이 자동으로 인식되었습니다.")
            return response.tool_calls
        
        print(f"[도구 호출 없음] 응답에서 도구 호출 정보를 찾을 수 없습니다. 수동으로 JSON 패턴을 추출합니다.")

        # 2. 모델이 인식 못 했지만 텍스트 내에 JSON 배열 형식이 포함된 경우 (Manual Parsing)
        content = response.content.strip()
        
        # JSON 패턴([{ ... }, { ... }]) 추출
        match =  re.search(r'\[[\s\S]*?\]', content) # re.search(r'\[\{.*\}\,.+?\]', content)
        
        if match:
            try:
                datas = json.loads(match.group(0))
                result = []

                for data in datas:
                    result.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}", # KeyError: 'id' 방지
                        "name": data.get("name"),             # tool_call["name"] 대응
                        "args": data.get("parameters") or data.get("args") or {} # tool_call["args"] 대응
                    })
                
                # call_tool 함수 내부의 tool_call["name"], tool_call["args"] 참조에 대응
                return result
            
            except (json.JSONDecodeError, Exception):
                return []
                
        return None
    
    async def converse_mcp(self, session, messages, tools):
        """
        LLM과의 대화를 관리하는 함수. LLM을 호출하고, 도구 사용 요청이 있을 때 도구를 호출하여 결과를 다시 LLM에 전달하는 과정을 반복 수행
        :param session: MCP 세션
        :param messages: 대화 이력
        :param tools: 사용 가능한 MCP 도구 목록
        """
        iteration = 0
        converted_tools = self.convert_tool_format(tools)
        
        logger.warning(f"최대 반복 횟수({MAX_ITERATIONS})까지 TOOL CALL을 실행합니다.")
        while iteration < MAX_ITERATIONS:
            iteration += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[반복 {iteration}] LLM 호출 중...")

            # LLM에 도구를 바인딩하여 호출
            llm_with_tools = self.tc_llm.bind_tools(converted_tools)
            response = llm_with_tools.invoke(messages)
            
            # LLM이 TOOL을 자동으로 인식한 경우 (Native)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"[도구 호출 자동 인식] {len(response.tool_calls)}개의 도구 호출이 자동으로 인식되었습니다.")
            else:
                logger.info("[도구 호출 없음] 응답에서 도구 호출 정보를 찾을 수 없습니다.")
                logger.info("수동으로 도구 호출 정보 추출 합니다.")
                response.tool_calls = self.get_actual_tool_calls(response)
            
            if response.tool_calls and len(response.tool_calls) > 0:
                await self.call_tool(session, response, messages)
                
                ## TODO: 응답결과를 분석하여 추가로 호출할 도구가 있는지 판단하는 로직 추가 가능 (예: 도구 결과에 따라 후속 도구 호출 필요 여부 판단)
                ## 예시: LLM으로 호출하여 법령(조항호) 추가검색 또는 웹검색이 필요한지 판단하는 로직추가 하여 도구호출 반복 수행하도록 개선 가능
                ## 단, 무한 반복 방지 위해 최대 반복 횟수(MAX_ITERATIONS) 도달 시 종료하도록 설정

                break
            else:
                if iteration >= MIN_ITERATIONS :
                    logger.info("호출된 도구가 없어 TOOL CALL을 종료합니다.")
                    break
            
        if iteration >= MAX_ITERATIONS:
            logger.info("최대 반복 횟수에 도달하여 TOOL CALL을 종료합니다.")

        return messages



    async def complete(self, sse_client_addr, sys_message, convrstn_context, original_question, enableExtDocse, human_message):
        """
        SSE 연결을 통해 MCP 서버와 통신하며 전체적인 질의응답 워크플로우를 수행하는 함수
        :param sse_client_addr: MCP 서버 주소
        :param sys_message: 추가적인 시스템 지시사항 (문서 경로 등)
        :param convrstn_context: 요약된 이전 대화 맥락
        :param original_question: 사용자의 원본 질문
        :param human_message: 강화된 질의가 포함된 HumanMessage 객체
        """
        logger.info("세션 시작")
        messages = []

        if not sse_client_addr:
            logger.info("SSE 클라이언트 주소가 설정되지 않았습니다.")
            return messages

        async with sse_client(sse_client_addr) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                try:
                    await session.initialize()
                    # 세션이 초기화될 수 있도록 잠시 대기
                    await asyncio.sleep(SESSION_INIT_WAIT)
                    logger.info("세션 초기화 완료")

                    # 사용 가능한 도구 목록을 가져오고 직렬화 가능한 형식으로 변환
                    tools_result = await session.list_tools()
                    tools_list = [{"name": tool.name, "description": tool.description,
                                "inputSchema": tool.inputSchema} for tool in tools_result.tools]

                    today_str = datetime.now().strftime("%Y-%m-%d")
                    
                    # ensure_ascii=True: 모든 비-ASCII 문자를 \uXXXX 형태로 이스케이프하여 안전한 ASCII 문자열로 변환함
                    tools_list_str = json.dumps(tools_list, ensure_ascii=False)

                    sys_msg_content = (
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
                    
                    messages.append(SystemMessage(content=sys_msg_content))
                    messages.append(human_message)
                    
                except Exception as err:
                    logger.error(f"MCP 초기화 중 오류 발생: {err}")

                await self.converse_mcp(session, messages, tools_result.tools)

                if enableExtDocse:
                    logger.info(f"enableExtDocse: {enableExtDocse}")

                return messages
            

def get_mcp_manager():
    """
    MCPClientManager 인스턴스를 생성하여 반환하는 함수
    """
    return MCPClientManager()


from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def call_tool(sse_client_addr, payload):
    # sse_client와 ClientSession은 async context manager이므로
    # 내부 비동기 로직을 정의한 뒤 asyncio.run으로 동기 실행한다.
    async def _call_tool_async():
        try:
            async with sse_client(sse_client_addr) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    try:
                        await session.initialize()
                        # 세션이 초기화될 수 있도록 잠시 대기
                        await asyncio.sleep(SESSION_INIT_WAIT)
                        logger.info("세션 초기화 완료")

                        tool_name = payload["tool"]
                        tool_args = payload["input"]

                        try:
                            # MCP 세션을 통해 도구 호출
                            tool_response = await session.call_tool(tool_name, tool_args)

                            return {
                                "status": "success",
                                "response": tool_response.content[0].text,
                            }

                        except Exception as err:
                            return {
                                "status": "failed",
                                "error_message": f"도구 호출 중 오류 발생: {str(err)}",
                            }

                    except Exception as err:
                        return {
                            "status": "failed",
                            "error_message": f"MCP 세션 초기화 중 오류 발생: {str(err)}",
                        }
        except Exception as err:
            return {
                "status": "failed",
                "error_message": f"SSE 연결 또는 세션 생성 중 오류 발생: {str(err)}",
            }

    return asyncio.run(_call_tool_async())
    