### 1. 핵심 구현 내용

이번 PoC 단계에서 실제 코드로 구현된 핵심 기능들을 **동작 원리**와 **사용 기술** 중심으로 기술합니다. (범위 구분은 2장, 실증 결과는 4·5장 참고)

**1.1 에이전트 워크플로우 (Agent Workflow)**

[PpsAssistAgent - 조달 상담 에이전트]

* **구현 기능:** 공공조달 관련 사용자 질문을 분석하여 적절한 법령 DB 또는 웹 검색으로 라우팅 후 답변 생성 (`llm-server/agent/pps_assist_agent.py`)
* **동작 원리:** LangGraph 기반 5-노드 파이프라인. 대화 시작(대화ID 생성) → 대화 히스토리 기반 쿼리 재작성 → 업로드 문서 검색(있을 경우) → 조건 분기(외부문서 사용 여부) → 법령 DB 또는 웹 검색 → LLM 최종 답변 생성. 법령 검색 노드(`qna_law_base_node`)에서는 별도 LLM 호출로 질문 유형을 판단해 5개 전문 에이전트 모드(일반용역/기술용역/시설공사/물품/비축) 중 하나를 자동 선택하고, 실패 시 `PpsStockpilingAgent`로 폴백함.
* **주요 기술:** LangGraph StateGraph, Azure ChatOpenAI (GPT), MCP Tool Calling, SSE Streaming

[NoticeScanAgent - 입찰공고 분석 에이전트]

* **구현 기능:** 입찰공고 문서(PDF/HTML)를 업로드하면 규격화된 메타데이터로 자동 추출 (`llm-server/agent/notice_scan_agent.py`, `notice_scan_structure.py`)
* **동작 원리:** 3-노드 파이프라인. 문서를 Markdown으로 변환 → LLM이 Pydantic 스키마(`ProcurementData` = general/execution/items/progresses/statuses) 기반 구조화 출력(`with_structured_output()`)으로 항목 추출 → 오류 발생 시 에러 핸들러 노드로 graceful degradation. 진행 상황은 SSE 스트리밍으로 프론트엔드에 실시간 전달됨.
* **주요 기술:** LangGraph StateGraph, Pydantic BaseModel, SSE Streaming, pymupdf4llm, MarkItDown

---
**1.2 도구(Tool) 및 함수 연동**

* **구현 기능:** MCP(Model Context Protocol) 기반 RAG 도구 서버를 별도 마이크로서비스(`llm-mcp-doc-rag`)로 분리하여 에이전트에서 호출
* **동작 원리:** `llm-mcp-doc-rag` 서비스가 MCP SSE 서버로 동작하며 4개 도구(`qna_law_base`, `qna_doc`, `qna_web_search`, `convert_to_markdown`)를 노출함. `mcpUtils.py`의 MCP 클라이언트가 LLM에게 도구 스키마(JSON)를 제공하고, LLM이 선택한 도구를 SSE 채널로 호출·결과 수신함. 세션 초기화 타이밍 이슈는 대기 + 재시도 로직으로 처리됨.
* **주요 기술:** MCP SDK 1.25.0, FastAPI SSE, LangChain Tool Definition, Pydantic

---
**1.3 데이터 및 메모리 (RAG & Context)**

* **구현 기능:** 한국 법령 문서의 계층 구조를 인식하는 도메인 특화 청킹 + Parent-Child 노드 기반 벡터 검색
* **동작 원리:**
  * 문서 변환: PDF/HTML → Markdown (pymupdf4llm, MarkItDown) → 불필요 마크다운 기호 정규화
  * 청킹 전략: `LawChunker`가 정규식으로 조문(조)/항(①②)/호(1.2.) /별표/부칙 구조를 탐지하여 계층 노드 생성. 법령 구조가 없으면 `FallbackChunker`(SentenceSplitter)로 자동 전환.
  * 인덱싱: Parent 노드(전체 조문, 문맥용) + Child 노드(세부 항목, 검색용)로 분리. Child 노드를 임베딩하여 FAISS에 저장, 원본 파일 MD5 해시 기반 폴더에 캐싱(`vectorstore/`).
  * 검색: 사용자 질문을 LLM으로 서브쿼리 1~2개로 분해 → 상위 K개 Child 노드 검색 → Parent 메타데이터와 합쳐 컨텍스트 구성.
  * 대화 메모리: 대화 히스토리를 DB(PostgreSQL, `convrstn`/`convrstn_details` 테이블)에 원문 그대로 저장하고 다음 쿼리 재작성 시 활용.
* **주요 기술:** LlamaIndex 0.14.7, FAISS, HuggingFace SBERT (snunlp/KR-SBERT-V40K-klueNLI-augSTS), Azure OpenAI Embeddings (text-embedding-3-small), SQLAlchemy

---

### 2. 구현 완료 범위와 미구현 범위

문서 신뢰도를 위해 "동작이 로그/코드로 확인된 것"과 "설계 의도만 있는 것"을 명확히 분리합니다.

**2.1 완전 구현 — 실행 로그 또는 코드로 동작이 확인된 모듈**

| 모듈 | 확인 근거 |
| :--- | :--- |
| PpsAssistAgent 5-노드 파이프라인 (라우팅→검색→답변) | `log/llm-server_2026-07-03.log` 12:56~12:57 구간에 대화 시작→쿼리 재작성→MCP 세션 초기화→도구 호출→답변 생성까지 전 구간 로그로 확인 |
| MCP 4개 도구(SSE) 연동 및 세션 초기화 재시도 | 전 로그 파일에 `MCP 세션 초기화 완료`, `HTTP ... 202 Accepted` 다수 확인 |
| `agent_mode` 자동 선택(질문 유형 → 5개 전문 에이전트 매핑) | `log/llm-server_2026-06-30.log:137` `호출할 도구 목록: [...{'agent_mode': 'PpsConstructionAgent', ...}]` — "공사 적격심사" 질문이 실제로 `PpsConstructionAgent`로 정확히 라우팅됨을 확인 |
| LawChunker/FallbackChunker 2단계 청킹 + FAISS 캐싱 | `log/llm-mcp-doc-rag_*.log`에 `기존 벡터스토어를 로드합니다: ./vectorstore/base_resource/<md5>` 반복 확인, `vectorstore/` 하위에 실제 캐시 폴더 다수 존재 |
| NoticeScanAgent 문서→Markdown 변환 노드 | `log/llm-mcp-doc-rag_2026-07-03.log:855` `마크다운 변환 중... (...비축물자대금 신용카드...)` |
| React 프론트엔드 결과 표시(요약 카드 4종 + 상세정보/품목/진행현황/원문/자동입력항목 5탭, JSON 클립보드 복사) | `SummaryMetrics.tsx`, `ResultTabs.tsx` 코드 확인 |

**2.2 부분 구현 — 동작은 하나 한계·불일치가 확인된 모듈**

| 모듈 | 한계 내용 | 확인 근거 |
| :--- | :--- | :--- |
| 법령 RAG 검색 리콜 | 동일 질문을 반복 호출해도 절반 이상이 관련 청크를 찾지 못해 `Empty Response`를 반환. 재시도 시 답변 자체가 달라져 재현성도 낮음 | 5.4절 로그 통계, 4장 시나리오 1 |
| `ProcurementData` 스키마의 계약방법(Literal) 필드 | `contractMethod`(일반경쟁/지명경쟁/제한경쟁/수의계약), `awardMethod`(희망수량경쟁입찰/최저가낙찰제)이 공사·용역 입찰 기준으로 설계되어 있어, 실제 비축물자 공고의 "프리미엄 경쟁입찰" 같은 값은 스키마에 없어 근사 매핑되거나 공백 처리됨 | 5.1절 샘플(실제 업로드 문서 `R26BK01538675-000_알루미늄주괴`) |
| 대화 메모리 | DB 저장은 되지만 요약/압축 없이 원문이 그대로 누적되는 구조라, 대화가 길어지면 컨텍스트 크기가 선형으로 증가 | `db/models.py: ConvrstnDetails`, `convrstnContextUtils` 코드 확인 |

**2.3 설계만 되어 있고 미구현 — 프롬프트/주석 수준에만 존재**

| 항목 | 현재 상태 |
| :--- | :--- |
| **입찰공고 규정 위반 자동 감지** | `notice_scan_agent.py`의 시스템 프롬프트에는 "규정 저촉 조항이 의심되면 `regulatory_review_notice`에 상세 경고를 기재하라"는 지시문이 있으나, 실제 `ProcurementData` Pydantic 모델(`general/execution/items/progresses/statuses`)에는 해당 필드가 정의되어 있지 않음. 코드의 `is_violating = bool(data_dict.get("regulatory_review_notice"))`는 딕셔너리에 그 키가 없어 **항상 `False`**를 반환하며, React 쪽에도 위반 표시 UI가 없음(`ResultTabs.tsx`, `SummaryMetrics.tsx`에 violation 관련 코드 없음). 즉 "위반 감지" 기능은 초안 문서에 서술되었던 것과 달리 **현재 실질적으로 미구현 상태**임 — 5.2절에서 상세 설명 |
| 대화 히스토리 요약(장기 컨텍스트 압축) | 저장 로직만 존재, 요약 로직 없음 |
| Streamlit → React 전체 이관 | 진행 중 (`llm-app/react/CLAUDE.md`에 마이그레이션 배경 명시), 일부 화면 이관 여부 미확인 |

---

### 3. 주요 문제 해결 및 기술 리서치

구현 과정에서 마주친 기술적 문제와 이를 해결하기 위해 찾아본 자료(리서치) 및 적용한 방법을 기록합니다. (2장에서 다룬, 아직 해결되지 않은 한계는 포함하지 않음)

| 이슈 구분 | 문제 상황 및 원인 | 리서치 및 해결 과정 (Reference & Solution) |
| :--- | :--- | :--- |
| **청킹** | PDF → Markdown 변환 후 `##`, `-`, 이미지 플레이스홀더 등 불필요 기호가 잔류하여 법령 구조 정규식에 오탐 발생 | • **리서치:** `pymupdf4llm` 변환 출력 패턴 분석<br>• **적용:** `_normalize_text()` 전처리 함수로 변환 아티팩트를 제거한 뒤 청킹 수행 |
| **청킹** | 일부 문서에 법령 계층 구조가 없어 `LawChunker` 적용 시 빈 노드 생성 또는 파싱 실패 | • **리서치:** LlamaIndex `SentenceSplitter` 문서 검토<br>• **적용:** 샘플 페이지에서 구조 여부 사전 탐지 후, 실패 시 `FallbackChunker`로 전환하는 2단계 전략 적용 (commit `3dc6120`) |
| **에이전트** | 상담사 에이전트가 법령 외 가이드 문서도 검색 대상에 포함하여 불필요한 컨텍스트 혼입 | • **리서치:** LangGraph 노드별 분기 처리 패턴 검토<br>• **적용:** `agent_mode` 기반 예외 처리로 가이드 문서를 쿼리 엔진 대상에서 명시적으로 제외 (commit `1ae763e`) |
| **구조화 출력** | `NoticeScanAgent`의 Pydantic 스키마 필드 제약 미비로 LLM 출력값이 비표준 형식으로 반환되어 파싱 오류 발생 | • **리서치:** Pydantic Field constraints, LangChain `with_structured_output()` 동작 방식 분석<br>• **적용:** description 강화 + Literal 타입으로 허용값 명시, 오류 시 Error Handler 노드로 분기 (commit `a7b561b`, `f9f425e`) |
| **도구 연동** | MCP 클라이언트가 서버 SSE 엔드포인트 연결 전에 세션을 초기화하려 해 Race Condition 발생 | • **리서치:** MCP SSE 프로토콜 핸드셰이크 순서 분석<br>• **적용:** 세션 초기화 대기 + 재시도 로직 추가 |
| **성능/기타** | Windows 환경에서 NumPy/scikit-learn OpenMP 라이브러리 중복 로딩으로 경고 및 프로세스 불안정 발생 | • **리서치:** Intel OpenMP + MS OpenMP 충돌 이슈 확인 (Stack Overflow, NumPy GitHub)<br>• **적용:** `os.environ["KMP_DUPLICATE_LIB_OK"] = "True"` 환경변수 설정 |

---

### 4. 검증 시나리오별 결과

실제 운영 로그(`log/llm-server_*.log`, `log/llm-mcp-doc-rag_*.log`)와 코드/스키마 정적 검토를 근거로, 아래 표는 **재구성이 아닌 실측 결과**입니다.

| 시나리오 | 입력 | 기대 결과 | 실제 결과 | 판정 |
| :--- | :--- | :--- | :--- | :--- |
| **① 법령 질의 – 반복 재현성** | "추정가격이 30억인 공사는 어떤 적격심사를 적용 받아?" (동일 질문을 00:01~00:09 사이 6회 연속 질의, `llm-server_2026-07-02.log:192-257`) | 매번 동일하거나 최소한 일관된 근거의 답변 | 6회 중 3회는 `Empty Response`(00:04:49, 00:05:50, 00:07:35), 1회는 무관한 가이드 PDF에서 "30억 이상이면 적격심사 대상"이라는 답변(00:03:12), 마지막 1회(00:09:10)는 정반대로 "30억은 적격심사 대상이 아닌 소규모 공사로 제외 가능성 높음"이라는 답변이 실제 법령 조문(제1조·제2조)을 근거로 생성됨 | **부분 실패** — 파이프라인은 완주하나 검색 결과가 비어 있거나(50%), 같은 질문에 서로 모순되는 답변이 나와 재현성·신뢰도 이슈 확인 |
| **② 법령 질의 → agent_mode 자동 라우팅** | "공사 적격심사의 역할과 기능을 설명해 주세요." | 시설공사 관련 질문이므로 `agent_mode=PpsConstructionAgent`로 라우팅 | 로그(`llm-server_2026-06-30.log:137`)에서 실제로 `qna_law_base` 도구가 `agent_mode: 'PpsConstructionAgent'`로 호출됨을 확인 | **성공** |
| **③ 법령 질의 – 단건 검색 실패** | "비축업무의 낙찰자선정방법에 어떤게 있는지 알려줘" (`llm-server_2026-07-03.log`, convrstn_id=`4f24a0d5...`) | 하위질문 2건 모두에 대해 관련 조문 검색 및 답변 생성 | 질문 분해는 정상 수행(`['비축업무 낙찰자 선정방법은 무엇인가요?', '비축업무 낙찰자 선정 절차는 어떻게 되나요?']`)되었으나, 두 하위질문 모두 `Empty Response` 반환(`llm-mcp-doc-rag_2026-07-03.log:2283`) → 최종 답변은 근거 문서 없이 LLM 사전지식으로만 생성 | **실패** (근거 문서 미포함) |
| **④ 입찰공고 → 구조화 데이터 추출** | 실제 업로드된 나라장터 비축입찰 공고 `R26BK01538675-000_알루미늄주괴.html` (`uploadFile/20260626/`) | `ProcurementData` 스키마(general/execution/items)에 맞춰 공고번호, 품목, 일정 등 추출 | 해당 파이프라인의 **실행 로그가 현재 남아있지 않아 라이브 실행 결과는 확인 불가**. 문서 원문과 Pydantic 스키마를 수기 대조한 결과 대부분 필드는 매핑 가능하나, `contractMethod`(실제 값 "프리미엄 경쟁입찰")가 Literal 허용값(일반경쟁/지명경쟁/제한경쟁/수의계약)에 없어 강제 근사 또는 파싱 오류 가능성 확인 | **부분 성공** (스키마 확장 필요, 실행 검증은 로그 부재로 보류) |
| **⑤ 입찰공고 → 규정 위반 감지** | 임의의 입찰공고 문서 (특정 브랜드 지정 등 위반 소지 문구 포함 가정) | `regulatory_review_notice` 등에 위반 경고 표시 | `ProcurementData`에 위반 관련 필드가 아예 없어 `is_violating`이 로직상 항상 `False`. 프론트엔드에도 관련 UI 없음 | **실패 (미구현)** — 2.3절 참고 |

---

### 5. 샘플 출력 증거

**5.1 구조화 추출 JSON 예시**

`uploadFile/20260626/..._R26BK01538675-000_20260521200240_알루미늄주괴.html` (실제 업로드된 조달청 비축입찰 공고 원문)을 `ProcurementData` 스키마 필드에 맞춰 수기로 매핑한 예시입니다. (④번 시나리오 참고 — 라이브 파이프라인 로그는 없어 코드 기준 재구성)

```json
{
  "general": {
    "noticeType": "비축물자",
    "noticeNo": "R26BK01538675-000",
    "refNo": "",
    "noticeName": "알루미늄주괴 비축입찰 공고",
    "postDate": "2026/05/22",
    "agency": "조달청",
    "demandAgency": "",
    "contractType": "",
    "contractForm": "",
    "bidMethod": "전자입찰",
    "stockType": "비축",
    "contractMethod": "일반경쟁",
    "awardMethod": "희망수량경쟁입찰",
    "awardDetail": "",
    "rebidYn": null
  },
  "execution": {
    "manager": "황보철",
    "bidStartDate": "",
    "bidEndDate": "2026/06/02 13:00:00",
    "openDate": "2026/06/02 14:00:00",
    "openPlace": "조달청 전략비축물자과(입찰집행관 PC)",
    "depositExemptYn": "Y",
    "depositDate": "",
    "relatedNotice": ""
  },
  "items": [
    {
      "itemNo": "1",
      "itemName": "알루미늄(비서구산) High Grade Primary Aluminium Ingot LME 등록브랜드",
      "standard": "Al 99.7% 이상, 비서구산",
      "unit": "톤",
      "quantity": 1000,
      "unitPrice": null,
      "amount": null
    }
  ],
  "progresses": [],
  "statuses": []
}
```

> ⚠️ `contractMethod`는 원문상 실제 계약방법이 "프리미엄 경쟁입찰"이지만 Literal 허용값에 없어 가장 가까운 값("일반경쟁")으로 근사 표기했습니다. 이는 2.2절에서 지적한 스키마 한계이며, `unitPrice`/`amount`는 원문에 가격지정기간 내 LME 시세로 사후 확정된다고 명시되어 있어 추론하지 않고 `null` 처리했습니다(미추론 원칙).

**5.2 위반 탐지 결과 예시**

현재 코드 기준 실제 반환값과, 설계 의도상 목표했던 형태를 함께 제시합니다.

```json
// 현재 실제 반환값 (extract_metadata_node의 실제 동작)
{
  "extracted_data": { "general": { "...": "..." } },
  "is_violating": false
}
```

```json
// 설계 단계 목표였던 형태 — 미구현, ProcurementData에 필드가 없어 동작하지 않음
{
  "regulatory_review_notice": [
    {
      "clause": "제17조 특정규격 명시 금지",
      "issue": "특정 제조사 모델명 명시 (브랜드 제한 의심)",
      "severity": "warning"
    }
  ],
  "is_violating": true
}
```

**5.3 법령 질의응답 결과 예시 (실 로그 기반)**

* 질문: "추정가격이 30억인 공사는 어떤 적격심사를 적용 받아?" (`log/llm-server_2026-07-02.log:257`, 2026-07-03 00:08:50)
* 처리시간: 13.37초 (`log/llm-mcp-doc-rag_2026-07-03.log:97`)
* 실제 답변:

  > "추정가격 30억 원인 공사는 「조달청 공사계약 종합심사낙찰제 심사세부기준」에 따른 적격심사 적용 대상인 일반공사, 간이형공사, 실적제한공사, 고난도공사 등의 기준에 해당하지 않는 규모로 분류됩니다. 따라서 이 기준에 따르면 추정가격 30억 원인 공사는 적격심사 대상이 아닌 소규모 공사로 분류되어 별도의 종합심사낙찰제 적용 대상에서 제외될 가능성이 높습니다."

* 근거 메타데이터(citation): `law_title: "01 조달청 공사계약 종합심사낙찰제 심사세부기준(개정전문)"`, `article_no: "제1조(목적)"`, `제2조(정의)` (Parent 노드 기준 조문 전체 복원)
* 참고: 같은 질문을 8분 사이 6회 반복 질의한 로그 전체를 보면 이 답변 이전 시도들에서는 다른 문서(수요기관 퀵가이드)를 근거로 정반대 결론을 내리거나 빈 응답이 반환됨(4장 시나리오 ① 참고) — 즉 이 답변은 "성공 사례"이지만 재현성은 낮음을 함께 밝혀둠.

**5.4 실행 로그 요약**

`log/llm-mcp-doc-rag_2026-06-05.log` ~ `2026-07-05.log`(18개 파일)의 `Final Response` 라인을 전수 집계한 결과입니다.

| 항목 | 값 |
| :--- | :--- |
| 전체 RAG 호출(`Final Response` 로그) 건수 | 126건 |
| 하위질문 중 1개 이상이 `Empty Response`인 호출 | 71건 (약 56%) |
| 완전 성공(모든 하위질문에 근거 확보) | 55건 (약 44%) |

> 표본에는 개발 중 반복 재시도가 다수 포함되어 있어 실제 사용자 체감 성공률과는 차이가 있을 수 있으나, 검색 리콜이 안정적이지 않다는 정황 근거로는 충분합니다. 개선은 6장 로드맵 참고.

---

### 6. 향후 고도화 예정 (PoC 이후 과제)

아래는 이번 PoC 범위에서 설계 단계로만 남겨두고, 구현은 다음 단계로 넘기는 항목입니다.

* 입찰공고 규정 위반 자동 감지: `ProcurementData`에 `violations` 필드 추가, 위반 유형 사전 정의, React 쪽 경고 UI 추가
* 법령 RAG 검색 리콜/재현성 개선: 하이브리드 검색(BM25 + Dense), 청크 크기·top-K 튜닝, 동일 질문 재시도 시 결과 일관성 확보
* 비축물자 전용 계약방법 Enum 확장(`contractMethod`/`awardMethod`에 "프리미엄 경쟁입찰" 등 추가)
* NoticeScanAgent 실행 로그 표준화(현재 별도 로그가 남지 않아 라이브 검증이 어려움)
* 대화 히스토리 요약/압축 로직(장기 대화 시 컨텍스트 비대화 방지)
* Streamlit 레거시 화면의 React 전체 이관 마무리
