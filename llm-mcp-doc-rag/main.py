
import mcp.types as types

from mcp import Tool
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from pydantic import BaseModel, Field

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response

from utils import config, loggerUtil

# =========================
# LangChain Tool
# =========================
from langchain_core.tools import tool
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

from tools import listTools, docAnalyze, ragWebsite

logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")
app = Server("mcp-server")
sse = SseServerTransport("/messages")

# SSL 경고 무시 (선택사항)
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


# =====================================================
# 1. @tool 기반 역할/도구별 핸들러 정의
# =====================================================
class qnaWebSearchInput(BaseModel):
    """
    웹 검색 기반 질의응답 도구의 입력 스키마.
    
    이 모델은 MCP 및 LangChain Tool에서
    입력 검증과 자동 문서화를 위해 사용된다.
    """
    question: str = Field(
        ...,
        description="사용자가 질문하는 자연어 질의. ",
        json_schema_extra={
            "examples": ["양자 컴퓨터의 기본 원리는 무엇인가?"],
            "min_length": 1
        }
    )
    allow_search: bool = Field(
        default=True,
        description="인터넷 검색 허용 여부"
    )

@tool(args_schema=qnaWebSearchInput)
async def qna_web_search(**kwargs):
    """웹 검색 기반 질의응답"""

    args = qnaWebSearchInput(**kwargs)

    if not args.question:
        raise ValueError("Missing required argument: question")

    if not args.allow_search:
        return [types.TextContent(
            type="text", 
            text="인터넷 검색이 허용되지 않아 정보를 찾을 수 없습니다. '외부문서 검색 허용'을 활성화해주세요."
        )]
    
    return await ragWebsite.execute(args.question)





class QnADocInput(BaseModel):
    """
    문서 기반 질의응답 도구 입력 스키마
    """

    question: str = Field(
        ...,
        description="문서 내용을 기반으로 분석할 질문"
    )
    fileFullPath: str = Field(
        ...,
        description="분석 대상 문서의 절대 경로"
    )
    allow_search: bool = Field(
        default=False,
        description="외부 검색 허용 여부"
    )

@tool(
    args_schema=QnADocInput,
    description="지정된 문서를 분석하여 질문에 답변하는 도구"
)
async def qna_doc(**kwargs):
    """문서 분석 기반 질의응답"""

    args = QnADocInput(**kwargs)

    return await docAnalyze.execute(args.question, args.fileFullPath, None, args.allow_search)




class QnaAgentInput(BaseModel):
    """
    문서 기반 질의응답 도구 입력 스키마
    """

    question: str = Field(
        ...,
        description="문서 내용을 기반으로 분석할 질문"
    )
    agent_mode: str = Field(
        ...,
        description="전문적인 상담을 지원하는 agent_mode"
    )
    fileFullPath: str = Field(
        default="",
        description="분석 대상 문서의 절대 경로"
    )
    allow_search: bool = Field(
        default=False,
        description="외부 검색 허용 여부"
    )

@tool(
    args_schema=QnaAgentInput,
    description="전문 Agent mode에 따라 기본 참조 문서 내용에 근거한 답변을 생성"
)
async def qna_agent(**kwargs):
    """문서 분석 기반 질의응답"""

    try:
        args = QnaAgentInput(**kwargs)
    except Exception as e:
        logger.error(f"Invalid input for qna_agent: {e}")
        raise ValueError("Invalid input for qna_agent. Please check the provided arguments.")

    return await docAnalyze.execute(args.question, args.fileFullPath, args.agent_mode, args.allow_search)



# =====================================================
# 2. Tool Registry (중앙 매핑)
# =====================================================
TOOL_REGISTRY = {
    "QnA_doc": qna_doc,
    "QnA_agent": qna_agent,
   # "QnA_web_search": qna_web_search,
}




# =====================================================
# 3. MCP call_tool → Router 역할만 수행
# =====================================================
@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

    print(name)
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool '{name}'")

    handler = TOOL_REGISTRY[name]
    
    return await handler.ainvoke(arguments)

# =====================================================
# 4. MCP list_tools
# =====================================================
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return listTools.getList()



# =====================================================
# 5. SSE Endpoints
# =====================================================
async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options()
        )


async def handle_messages(request):
    await sse.handle_post_message(
        request.scope,
        request.receive,
        request._send
    )



# =====================================================
# 6. Starlette App
# =====================================================
starlette_app = Starlette(
    debug=False,
    routes=[
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ],
)



# =====================================================
# 7. Server Run
# =====================================================
if __name__ == "__main__":
    print(f"##################  Server Name = {config.settings.SERVER_NAME}  ##################")
    print(f"##################  Port Number = {config.settings.SERVER_PORT}  ##################")
    
    """LlamaIndex 전역 설정을 config.py의 임베딩 모델과 동기화"""
    # config.py에 정의된 LLM 설정 적용
    Settings.llm = config.settings.get_llm()

    Settings.embed_batch_size = 128  # 혹은 128 (시스템 환경에 맞게 조정)

    # config.py에 정의된 임베딩(문장 수치화) 모델 설정 적용
    Settings.embed_model = config.settings.get_embeddings()

    # 문서를 자르는 단위(Chunk) 설정: 800자 단위로 자르고 100자씩 겹치게 함
    Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)

    agent_infos = []
    agent_infos.append({"mode":"PpsGeneralServiceAgent", "name":"조달청 일반용역 상담", "resource":"./llm-mcp-doc-rag/resource/agent/PpsGeneralServiceAgentGuide.zip;./llm-mcp-doc-rag/resource/agent/PpsGeneralServiceAgent.zip;"})
    agent_infos.append({"mode":"PpsTechnicalServicesAgent", "name":"조달청 기술용역 상담", "resource":"./llm-mcp-doc-rag/resource/agent/PpsTechnicalServicesAgentGuide.zip;./llm-mcp-doc-rag/resource/agent/PpsTechnicalServicesAgent.zip;"})
    agent_infos.append({"mode":"PpsConstructionAgent", "name":"조달청 시설공사 상담", "resource":"./llm-mcp-doc-rag/resource/agent/PpsConstructionAgentGuide.zip;./llm-mcp-doc-rag/resource/agent/PpsConstructionAgent.zip;"})
    agent_infos.append({"mode":"PpsProductsAgent", "name":"조달청 물품 상담", "resource":"./llm-mcp-doc-rag/resource/agent/PpsProductsAgentGuide.zip;./llm-mcp-doc-rag/resource/agent/PpsProductsAgent.zip;"})
    agent_infos.append({"mode":"PpsStockpilingAgent", "name":"조달청 비축 상담", "resource":"./llm-mcp-doc-rag/resource/agent/PpsStockpilingAgentGuide.zip;./llm-mcp-doc-rag/resource/agent/PpsStockpilingAgent.zip;"})

    for agent_info in agent_infos:
        mode = agent_info["mode"]
        resource_paths = agent_info["resource"]
        config.settings.set_query_engine(mode, resource_paths)  # 쿼리 엔진을 미리 생성하여 캐싱해둠

    logger.info("Starting MCP Server...")

    uvicorn.run(
        starlette_app,
        host="127.0.0.1",
        port=config.settings.SERVER_PORT,
        log_level="critical",
        access_log=False,
        timeout_keep_alive=1800
    )