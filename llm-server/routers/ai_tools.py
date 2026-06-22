from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils import config

router = APIRouter(prefix="/api/v1/ai-tools", tags=["agent"])


class GenerateQuestion(BaseModel):
    html_content: str
    selected_text: str


@router.post("/generate-question")
def generate_question(request: GenerateQuestion):
    if not request.selected_text:
        raise HTTPException(status_code=400, detail="selected_text is required")
    if not request.html_content:
        raise HTTPException(status_code=400, detail="html_content is required")

    llm = config.get_llm().bind(stream=False)

    categories = ["일반용역", "기술용역", "시설공사", "물품", "비축"]
    matched_category = next((cat for cat in categories if cat in request.selected_text), None)

    prompt = (
        f"다음 HTML 본문에서 사용자가 선택한 텍스트의 의미를 파악하고, "
        f"단어 또는 표현의 정의를 묻는 형태의 질문을 생성해 주세요.\n"
        f"HTML 본문: {request.html_content}\n"
        f"선택한 텍스트: {request.selected_text}\n"
        f"선택한 텍스트에 일반용역, 기술용역, 시설공사, 물품, 비축 중 특정 업무 유형이 파악되면, "
        f"그 업무 유형을 질문에 포함하여 생성하세요.\n"
        f"예: \"비축업무에서 조달청 조달물자 계약관이 하는 역할이 뭐야?\"와 같은 질의문 형태로, "
        f"불필요한 부연설명 없이 간단하고 명료한 질문 한 문장만 출력하세요."
    )

    if matched_category:
        prompt += f"\n업무 유형: {matched_category}"

    generated_question = llm.invoke(prompt)

    return {"generated_question": generated_question.content}
