from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import re

# 'Y' / 'N' 외 다른 값이 들어올 수 없는 필드에 사용. 정보가 없으면 None 허용.
YNType = Optional[Literal["Y", "N"]]

class GeneralInfo(BaseModel):
    """공고 일반 정보 (general)"""
 
    noticeType: Optional[Literal["실공고(등록공고)", ""]] = Field(
        default="실공고(등록공고)",
        max_length=100,
        examples=["실공고(등록공고)"],
        description=(
            "- 목적: 해당 문서가 테스트용이나 예가 공고가 아닌, 실제 입찰을 진행하기 위해 조달청 시스템에 정식 등록된 공고임을 판별하여 '공고종류'를 '실공고(등록공고)'로 추출합니다.\n"
            "- 추출 근거: \n"
            "  1. 문서 상단에 실제 고유한 조달청 공고번호(예: 제 R26BK01492382-000호)가 명시되어 있는 점\n"
            "  2. '다음과 같이 입찰에 부치고자 공고합니다'라는 명확한 의사표시와 실제 공고 일자(2026/04/28), 전자입찰서 제출 마감일시(2026/05/07 13:00)가 구체적으로 기재되어 있는 점\n"
            "- 결과 처리: 위와 같이 실제 입찰 집행을 목적으로 정식 발송 및 등록된 공고문임이 확인되므로, 해당 필드의 최종 값을 '실공고(등록공고)'로 확정하여 출력하십시오.\n"
        ),
    )
    noticeNo: str = Field(
        max_length=100,
        pattern=r"^[RT]\d{2}BK[\d-]+$",  # 숫자와 하이픈(-)만 허용
        examples=["R26BK01492382-000"],
        description="공고번호 또는 구매결의번호. 본문에서 '공고 제 R...' 또는 '구매결의번호' 뒤의 명시된 문자열을 정확히 추출.",
    )
    noticeName: str = Field(
        max_length=500,
        examples=["알루미늄주괴 비축입찰 공고"],
        description=(
            "공고명(제목)을 생성합니다. 본문 전체나 파일명에서 가장 핵심이 되는 '대표 품목명'(예: 알루미늄, 알루미늄주괴)을 먼저 추출하십시오.\n"
            "그 후, 추출한 품목명을 사용하여 반드시 '가장 단순한 형태의 {품목명} 비축입찰 공고' 형식으로 포맷을 강제 변환하여 출력해야 합니다.\n"
            "규칙 및 주의사항:\n"
            "1. 본문에 '조달(비축)물자 구매 입찰공고안내'와 같은 다른 제목이 있더라도 무시하고, 오직 '{품목명} 비축입찰 공고' 형식만 골격으로 유지하십시오.\n"
            "2. 품목명 뒤에 '구매', '매입', '안내', '조달' 등 임의의 접미사나 수식어를 절대 붙이지 마십시오. (틀린 예: 알루미늄주괴 구매 비축입찰 공고, 알루미늄 조달 비축입찰 공고)\n"
            "3. 가장 명확하고 깔끔한 품목명 하나만 대입하십시오. (올바른 예: '알루미늄주괴 비축입찰 공고')"
        ),
    )
    agency: str = Field(
        default="조달청",
        max_length=200,
        examples=["조달청"],
        description=(
            "- 목적: 해당 입찰을 관할하고 공고를 발행한 최상위 기관명을 '공고기관' 필드로 추출합니다. 하위 조직(예: 전략비축물자과)까지 추출되지 않도록 제어해야 합니다.\n"
            "- 추출 근거: \n"
            "  1. 문서 상단 '조달청 조달(비축)물자 공고 제 R26BK01492382-000호' 및 '조달청 조달물자 계약관' 명의\n"
            "  2. 본문 및 연락처에 명시된 '조달청 공공물자국 전략비축물자과' 문구\n"
            "- 결과 처리: '전략비축물자과'나 '공공물자국'과 같은 하위 부서/조직명은 모두 제외하고, 오직 최상위 발주 기관명인 '조달청'만 최종 값으로 확정하여 출력하십시오."
        ),
    )
    demandAgency: str = Field(default="조달청", max_length=200, description="수요기관명. 비축물자의 경우 통상 '조달청'임.")
    contractType: Optional[Literal["국가계약법", "지방계약법"]] = Field(
        default="국가계약법",
        max_length=100,
        description="계약구분. (예: 일반, 다수공급자계약, 협상계약 등)",
    )
    contractForm: str = Field(
        default="",
        max_length=100,
        examples=["단가", "총액", "회차"],
        description="계약형태 (예: 단가, 총액, 회차)",
    )
    bidMethod: Optional[Literal["전자입찰/직찰", "수기입찰"]] = Field(
        default="전자입찰/직찰",
        description="입찰방식. 본문에 '전자입찰에 의하며' 문구가 있으면 '전자입찰/직찰', 방문/우편만 가능하면 '수기입찰'.",
    )
    stockType: Optional[Literal["직접비축", "해당없음"]] = Field(
        default="직접비축",
        max_length=100,
        examples=["직접비축", "해당없음"],
        description="비축구분. 비축공고문이 확실하므로 '직접비축'을 기본값으로 지정.",
    )
    contractMethod: Optional[Literal["프리미엄계약", "지명경쟁", "제한경쟁", "수의계약"]] = Field(
        default="프리미엄계약",
        description="계약방법. 본문에 '프리미엄계약' 등 별도 표기가 있다면 해당 값을 적고, 기본적으론 '프리미엄계약' 유추.",
    )
    awardMethod: Optional[Literal["희망수량경쟁입찰", ""]] = Field(
        default="희망수량경쟁입찰",
        description=(
            "- 목적: 공고문의 수량 분할 조건과 낙찰자 결정 방식을 종합 분석하여, 최종 '낙찰방법'을 판별하고 추출합니다.\n"
            "- 최우선 판별 규칙 (희망수량경쟁입찰 판정): \n"
            "  본문(예: '1.4 구매 개요' 등)에 '희망수량 입찰 가능', '수량을 분할하여', 또는 '최저가 입찰자의 수량이 구매량에 미달하는 경우 차순위 입찰자를 낙찰자로 선정' 등의 문구가 존재한다면, 이는 가격 평가 방식(최저가 기준)과 관계없이 구조적으로 '희망수량경쟁입찰'에 해당합니다. 이 조건이 감지되면 무조건 '희망수량경쟁입찰'을 선택하십시오.\n"
            "- 추론 및 추출 근거:\n"
            "  1. 희망수량경쟁입찰 근거: 공고문 내 '희망수량 입찰 가능(입찰최소단위 : 500톤)' 명시 및 6.2항의 '최저가 입찰자의 수량이 구매량에 미달 시 차순위 선정' 규정은 전체 발주량을 한 업체가 독점하지 않고 나누어 가져가는 전형적인 '희망수량경쟁'의 특징입니다.\n"
            "  2. 최저가낙찰제 근거: '6. 낙찰자 결정방법'에서 Premium+관세 합산 금액이 가장 낮은(국가에 유리한) 자를 정한다고 되어 있으나, 이는 희망수량경쟁 내에서 우선순위를 정하는 '가격 평가 기준'일 뿐입니다.\n"
            "- 결과 처리 지침: 수량 분할 및 희망수량 투찰 조건이 단 하나라도 매칭된다면, 가격이 최저가 기준이더라도 '최저가낙찰제'가 아닌 '희망수량경쟁입찰'로 최종 값을 정확하게 도출해야 합니다."
        ),
    )
    awardDetail: Optional[Literal["희망수량경쟁(최저가)", ""]] = Field(
        default="",
        max_length=100,
        description=("- 목적: 해당 공고의 낙찰 세부 기준이 '희망수량경쟁(최저가)' 항목에 부합하는지 확인하고 추출합니다.\n"
                     "- 추출 근거:\n"
                     "  1. '1.4 구매 개요'의 '희망수량 입찰 가능(입찰최소단위: 500톤)' 문구\n"
                     "  2. '6. 낙찰자 결정방법' 및 '6.2'의 '최저가 입찰자의 수량이 구매량에 미달하는 경우 차순위 선정' 문구\n"
                     "- 결과 처리: 위의 두 근거를 조합하여 최종 매칭 필드값을 '희망수량경쟁(최저가)'으로 통일하여 출력하십시오.\n"),
        )
    rebidYn: YNType = Field(default="N", description="재입찰 여부. 특별한 언급이 없으면 기본값 'N'")

    @model_validator(mode="after")
    def _validate_general(self) -> "GeneralInfo":
        # 공고명 형식 강제 가동화
        if self.noticeName and "비축입찰 공고" not in self.noticeName:
            clean_name = self.noticeName.replace("구매", "").replace("입찰", "").replace("공고", "").strip()
            self.noticeName = f"{clean_name} 비축입찰 공고"
        return self


class ExecutionInfo(BaseModel):
    """입찰 집행(일정) 정보 (execution)"""
    manager: str = Field(
        default="", 
        max_length=100, 
        description=(
            "담당자 성명 및 연락처. 본문에 '담당자 : 황보철 ☎ 042 724 7204'와 같은 문구가 있다면, "
            "불필요한 수식어(담당자, 문의, ☎, :, 공백)를 모두 정제하고 반드시 '이름 전화번호' 형식으로만 추출해야 함. "
            "전화번호의 국번과 번호 사이에는 하이픈(-)을 삽입할 것. "
            "출력 예시: '황보철 042-724-7204'"
        )
    )
    bidStartDate: str = Field(default="", max_length=50, description="입찰 시작일시. 본문의 공고 발행일자를 찾아'YYYY/MM/DD HH:MI:SS' 형식으로 정규화. 정보가 없으면 공고일시 활용.")
    bidEndDate: str = Field(default="", max_length=50, description="제출/입찰 마감일시. 'YYYY/MM/DD HH:MI:SS' 형식으로 정규화.")
    openDate: str = Field(
        default="", 
        max_length=50, 
        description=(
            "개찰일시. 본문의 '개찰일시 : 2026/05/07 14:00' 문구에서 날짜와 시간 정보를 정확히 파싱해야 함. "
            "텍스트에 분(MM) 단위까지만 나와 있더라도, 반드시 초 단위를 보완하여 'YYYY/MM/DD HH:MI:SS' 형식으로 정규화할 것. "
            "출력 예시: '2026/05/07 14:00:00' (텍스트에 초가 없으면 ':00'을 강제로 붙일 것)."
        )
    )
    openPlace: str = Field(default="", max_length=200, description="개찰장소 및 입찰장소 (예: 조달청 전략비축물자과)")
    depositExemptYn: YNType = Field(
        default=None, description="'입찰보증금 납부를 면제하고 지급각서로 대체' 문구가 보이면 'Y', 무조건 납부 대상자 조건만 나열되어 있으면 텍스트 문맥에 따라 판단."
    )


class Item(BaseModel):
    """공고 품목 정보 (items 배열의 원소)"""
 
    itemNo: str = Field(description="품목 순번 (문자열 형태 '1', '2'...)")
    itemName: str = Field(description="품명 (예: 알루미늄 주괴)")
    standard: str = Field(default="", description="규격 (예: High Grade Primary Aluminium Ingot LME 등록브랜드 AI 99.7% 이상, 비서구산)")
    unit: str = Field(
        default="톤",
        examples=["톤", "kg", "EA"],
        description="단위 (비축 원자재는 주로 '톤' 또는 'kg')",
    )
    quantity: Optional[float] = Field(default=None, description="수량 (숫자만 추출)")
    unitPrice: Optional[float] = Field(default=None, description="단가 (명시되지 않았거나 프리미엄만 입찰할 경우 빈 값 또는 LME 기준가 계산)")
    amount: Optional[float] = Field(
        default=None, description="금액. 공고문에 금액이 직접 명시되어 있지 않으면 수량 x 단가로 자동 계산됨."
    )
 
    @model_validator(mode="after")
    def _fill_amount(self) -> "Item":
        if self.amount is None and self.quantity is not None and self.unitPrice is not None:
            self.amount = self.quantity * self.unitPrice
        return self


class Progress(BaseModel):
    """진행 단계 정보 (progresses 배열의 원소)"""
 
    stepName: str = Field(
        max_length=100,
        description="진행 단계명 (예: 공고, 제출마감, 개찰, 선적기한, 인도기한 등 문맥에서 주요 마일스톤 날짜 추출)",
    )
    stepDate: str = Field(default="", max_length=50, description="해당 단계 진행 일자 (YYYY/MM/DD)")
    remark: str = Field(default="", max_length=300, description="비고 및 특이사항")


class Status(BaseModel):
    """투찰/낙찰 현황 정보 (statuses 배열의 원소) - 공고문 시점엔 주로 빈 배열"""
 
    bidderName: str = Field(max_length=200, description="투찰업체명")
    bidAmount: Optional[float] = Field(default=None, description="투찰금액")
    result: str = Field(
        default="",
        max_length=50,
        description="투찰 결과 (예: 낙찰, 낙찰탈락, 무효)",
    )


class ProcurementData(BaseModel):
    """조달청 나라장터 비축공고서 최종 구조체"""
 
    general: GeneralInfo
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    items: List[Item] = Field(default_factory=list)
    progresses: List[Progress] = Field(default_factory=list)
    statuses: List[Status] = Field(default_factory=list)