import json
import os

from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage

from utils import config, default_prompt, conversation_context_utils, mcp_utils


from repository import conversation_repository, agent_repository
from agent import pps_assist_agent
from agent.notice_scan_agent import PureLangNoticeScanAgent


# 로거 인스턴스 생성
logger = config.get_logger("./log", "llm-server")


# API 경로는 프론트엔드와의 계약 유지를 위해 변경하지 않음
router = APIRouter(
    prefix="/api/v1/convrstn",
    tags=["qna"],
    responses={404: {"description": "Not found"}},
)

class QuestionRequest(BaseModel):
    agent_id: str
    agent_mode: str
    convrstnId: str
    fileFullPath: str
    question: str
    enableExtDocse: bool

# 엔드포인트 경로 수정 (/stream -> 유지)
@router.post("/question")
async def stream_qna_workflow(request: QuestionRequest, raw_request: Request):
    logger.info(f"[ConversationRouter] 질의 요청 수신 agent_mode={request.agent_mode} conversation_id={request.convrstnId} question={request.question[:50]}")

    agent_info = agent_repository.read_agent(request.agent_id)

    if request.agent_mode == "PpsAssistAgent":
        agent_state = await pps_assist_agent.run(agent_info, request.convrstnId, request.question, request.fileFullPath, request.enableExtDocse)
        logger.info(f"[ConversationRouter] PpsAssistAgent 워크플로우 완료, 최종 답변 스트리밍 시작 conversation_id={request.convrstnId}")

        # tc_llm은 도구 호출용이므로, 일반 답변 생성에는 get_llm()을 사용하는 것이 적절할 수 있음
        # 하지만 일관성을 위해 tc_llm을 유지하되, 스트리밍이 필요한 경우 invoke 대신 stream 사용 고려
        final_llm = config.get_llm().bind(stream=True)

        async def event_generator():
            full_response = ""
            try:
                async for chunk in final_llm.astream(agent_state['rag_answer']):
                    if chunk.content:
                        full_response += chunk.content
                        yield chunk.content
            except Exception as e:
                logger.error(f"Streaming error: {e}")
            finally:
                
                if await raw_request.is_disconnected():
                    try:
                        # 연결이 끊긴 경우 스트리밍을 중단하고 나머지 결과를 한 번에 받아 DB에 저장
                        remaining_response = final_llm.bind(stream=False).invoke(agent_state['rag_answer'])
                        full_response = remaining_response.content
                    except Exception as e:
                        logger.error(f"disconnected error: {e}")
                        
                # 생성이 완료된 후(또는 에러 발생 후) DB에 저장
                conversation_repository.create_conversation_answer(request.convrstnId, {"convrstn_details_id": agent_state['conversation_details_id'], "answer": full_response, "agent_id": request.agent_id, "answer_at": datetime.now()})
                logger.info(f"[ConversationRouter] 최종 답변 스트리밍 및 저장 완료 conversation_id={request.convrstnId} answer_length={len(full_response)}")

        return StreamingResponse(event_generator(), media_type="text/plain")
    elif request.agent_mode == "NoticeScanAgent":
        agent = PureLangNoticeScanAgent()
        logger.info(f"[ConversationRouter] NoticeScanAgent 파이프라인 가동(스트리밍) conversation_id={request.convrstnId} file={request.fileFullPath}")

        async def notice_event_generator():
            final_state = None
            async for event in agent.run_stream(request.fileFullPath):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "result":
                    final_state = event.get("data")

            logger.info(f"[ConversationRouter] NoticeScanAgent 파이프라인 종료 conversation_id={request.convrstnId} status={(final_state or {}).get('status')}")

            if final_state:
                structured = final_state
                topic = os.path.basename(request.fileFullPath)
                extracted = structured.get("extracted_data") or {}
                general = extracted.get("general") or {}
                if general.get("noticeName"):
                    topic = general["noticeName"]
                elif general.get("refNo"):
                    topic = general["refNo"]

                conversation_repository.create_conversation(request.convrstnId, {"topic": topic})
                question_record = conversation_repository.create_conversation_question(
                    request.convrstnId,
                    {"context": "", "question": request.fileFullPath, "enhanced_question": ""},
                )
                conversation_repository.create_conversation_answer(
                    request.convrstnId,
                    {
                        "convrstn_details_id": question_record.convrstn_details_id,
                        "answer": json.dumps(structured, ensure_ascii=False),
                        "agent_id": request.agent_id,
                    },
                )

        return StreamingResponse(notice_event_generator(), media_type="text/event-stream")