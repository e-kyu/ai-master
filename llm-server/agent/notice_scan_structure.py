from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from utils import config
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

# 'Y' / 'N' 외 다른 값이 들어올 수 없는 필드에 사용. 정보가 없으면 None 허용.
YNType = Optional[Literal["Y", "N"]]

# 1.1 최종 추출될 조달 데이터 스키마 (Pydantic)
class GeneralInfo(BaseModel):
    """공고 일반 정보 (general)"""
 
    noticeType: str = Field(
        description="공고 종류. 예: 물품, 공사, 용역, 외자, 비축물자 등"
    )
    noticeNo: str = Field(description="공고번호 (예: 20240601-001)")
    refNo: str = Field(default="", description="참조번호/관리번호. 없으면 빈 문자열")
    noticeName: str = Field(description="공고명(제목)")
    postDate: str = Field(
        default="", description="게시(공고)일시. 'YYYY-MM-DDTHH:MM' 형식으로 정규화"
    )
    agency: str = Field(default="", description="공고기관명 (예: 조달청)")
    demandAgency: str = Field(default="", description="수요기관명")
    contractType: str = Field(
        default="", description="계약구분 (예: 일반, 다수공급자계약, 협상계약 등)"
    )
    contractForm: str = Field(
        default="", description="계약형태 (예: 단가, 총액, 회차)"
    )
    bidMethod: str = Field(
        default="", description="입찰방식 (예: 전자입찰, 수기입찰)"
    )
    stockType: str = Field(
        default="",
        description="비축구분 (예: 비축, 해당없음 등). 비축공고가 아니면 '해당없음'",
    )
    contractMethod: str = Field(
        default="",
        description="계약방법 (예: 일반경쟁, 지명경쟁, 제한경쟁, 수의계약)",
    )
    awardMethod: str = Field(
        default="", description="낙찰방법 (예: 최저가, 적격심사, 협상에의한계약)"
    )
    awardDetail: str = Field(
        default="", description="낙찰방법에 대한 상세 설명 (예: 예가 이하 최저가)"
    )
    rebidYn: YNType = Field(default=None, description="재입찰 여부. Y 또는 N")
 
class ExecutionInfo(BaseModel):
    """입찰 집행(일정) 정보 (execution)"""
 
    manager: str = Field(default="", description="담당자 성명")
    bidStartDate: str = Field(default="", description="입찰 시작일시")
    bidEndDate: str = Field(default="", description="입찰 마감일시")
    openDate: str = Field(default="", description="개찰일시")
    openPlace: str = Field(default="", description="개찰장소")
    depositExemptYn: YNType = Field(
        default=None, description="입찰보증금 면제 여부. Y 또는 N"
    )
    depositDate: str = Field(
        default="", description="보증금 납부 마감일. 해당 없으면 빈 문자열"
    )
    relatedNotice: str = Field(
        default="", description="관련(연계) 공고번호. 없으면 빈 문자열"
    )
 
 
class Item(BaseModel):
    """공고 품목 정보 (items 배열의 원소)"""
 
    itemNo: str = Field(description="품목 순번 (1, 2, 3 ...)")
    itemName: str = Field(description="품명")
    standard: str = Field(default="", description="규격")
    unit: str = Field(default="", description="단위 (예: 박스, EA, kg)")
    quantity: Optional[float] = Field(default=None, description="수량")
    unitPrice: Optional[float] = Field(default=None, description="단가")
    amount: Optional[float] = Field(
        default=None, description="금액. 공고문에 명시되어 있지 않으면 수량 x 단가로 계산"
    )
 
    @model_validator(mode="after")
    def _fill_amount(self) -> "Item":
        if (
            self.amount is None
            and self.quantity is not None
            and self.unitPrice is not None
        ):
            self.amount = self.quantity * self.unitPrice
        return self
 
 
class Progress(BaseModel):
    """진행 단계 정보 (progresses 배열의 원소)"""
 
    stepName: str = Field(
        description="진행 단계명 (예: 공고, 입찰등록, 입찰마감, 개찰, 낙찰)"
    )
    stepDate: str = Field(default="", description="해당 단계 진행 일자")
    remark: str = Field(default="", description="비고")
 
 
class Status(BaseModel):
    """투찰/낙찰 현황 정보 (statuses 배열의 원소)"""
 
    bidderName: str = Field(description="투찰업체명")
    bidAmount: Optional[float] = Field(default=None, description="투찰금액")
    result: str = Field(
        default="", description="투찰 결과 (예: 낙찰, 낙찰탈락, 무효)"
    )

class ProcurementData(BaseModel):
    """
    조달청 나라장터 비축공고서 최종 구조체.
 
    LangGraph 노드에서:
        structured_llm = llm.with_structured_output(ProcurementNotice)
        result: ProcurementNotice = structured_llm.invoke(messages)
        payload = result.model_dump()   # 시스템 입력용 dict
    형태로 사용한다.
    """
 
    general: GeneralInfo
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    items: List[Item] = Field(default_factory=list)
    progresses: List[Progress] = Field(default_factory=list)
    statuses: List[Status] = Field(default_factory=list)