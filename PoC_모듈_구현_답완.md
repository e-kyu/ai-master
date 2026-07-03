 핵심 구현 내용

이번 PoC 단계에서 실제 코드로 구현된 핵심 기능들을 동작 원리와 사용 기술 중심으로 상세히 기술합니다.

1.1 에이전트 워크플로우 (Agent Workflow)

[NoticeScanAgent - 입찰공고 분석 에이전트]

- 구현 기능: 입찰공고 문서(PDF/HTML)를 업로드하면 규격화된 메타데이터로 자동 추출 및 법령 위반 여부 감지
- 동작 원리: 3-노드 파이프라인으로 구성됨. 문서를 Markdown으로 변환 → LLM이 Pydantic 스키마 기반 구조화 출력(with_structured_output())으로 항목 추출 → 오류 발생 시 에러 핸들러 노드로 graceful degradation 처리. 진행 상황은 SSE 스트리밍으로 프론트엔드에 실시간 전달됨.
- 주요 기술: LangGraph StateGraph, Pydantic BaseModel, SSE Streaming, pymupdf4llm, MarkItDown

[PpsAssistAgent - 조달 상담 에이전트]

- 구현 기능: 공공조달 관련 사용자 질문을 분석하여 적절한 법령 DB 또는 웹 검색으로 라우팅 후 답변 생성
- 동작 원리: LangGraph 기반 5-노드 파이프라인으로 구성됨. 사용자 질문 접수 → 대화 히스토리 기반 쿼리 재작성 → 업로드 문서 검색(있을 경우) → 조건 분기(외부문서 사용 여부) → 법령 DB 또는 웹 검색 → LLM 최종 답변 생성 순으로 처리. 법령 검색 시 LLM이 질문 유형을 판단해 전문 에이전트 모드(일반용역/기술용역 등)를 자동 선택함.
- 주요 기술: LangGraph StateGraph, Azure ChatOpenAI (GPT), MCP Tool Calling, SSE Streaming

---
1.2 도구(Tool) 및 함수 연동

- 구현 기능: MCP(Model Context Protocol) 기반 RAG 도구 서버를 별도 마이크로서비스로 분리하여 에이전트에서 호출
- 동작 원리: llm-mcp-doc-rag 서비스가 MCP SSE 서버로 동작하며 4개 도구(qna_law_base, qna_doc, qna_web_search, convert_to_markdown)를 노출함. mcpUtils.py의 MCP 클라이언트가 LLM에게 도구 스키마(JSON)를 제공하고, LLM이 선택한 도구를 SSE 채널로 호출·결과 수신함. 세션 초기화 타이밍 이슈는 1초 대기 + 재시도 로직으로 처리됨.
- 주요 기술: MCP SDK 1.25.0, FastAPI SSE, LangChain Tool Definition, Pydantic


---
1.3 데이터 및 메모리 (RAG & Context)

- 구현 기능: 한국 법령 문서의 계층 구조를 인식하는 도메인 특화 청킹 + Parent-Child 노드 기반 벡터 검색
- 동작 원리:
  - 문서 변환: PDF/HTML → Markdown (pymupdf4llm, MarkItDown) → 불필요 마크다운 기호 정규화
  - 청킹 전략: LawChunker가 정규식으로 조문(조)/항(①②)/호(1.2.) /별표/부칙 구조를 탐지하여 계층 노드 생성. 법령 구조가 없으면 FallbackChunker(SentenceSplitter)로 자동 전환.
  - 인덱싱: Parent 노드(전체 조문, 문맥용) + Child 노드(세부 항목, 검색용)로 분리. Child 노드를 임베딩하여 FAISS에 저장, MD5 해시로 캐싱.
  - 검색: 사용자 질문을 LLM으로 서브쿼리 분해 → 쿼리 최적화(연도·도메인 키워드 추가) → 상위 20개 Child 노드 검색 → Parent 메타데이터와 합쳐 컨텍스트 구성.
  - 대화 메모리: 대화 히스토리를 DB(PostgreSQL/SQLite)에 저장하고 다음 쿼리 재작성 시 활용.
- 주요 기술: LlamaIndex 0.14.7, FAISS, HuggingFace SBERT (snunlp/KR-SBERT-V40K-klueNLI-augSTS), Azure OpenAI Embeddings (text-embedding-3-small), SQLAlchemy

---
주요 문제 해결 및 기술 리서치

구현 과정에서 마주친 기술적 문제와 이를 해결하기 위해 찾아본 자료(리서치) 및 적용한 방법을 기록합니다.

| 이슈 구분 | 문제 상황 및 원인 | 리서치 및 해결 과정 (Reference & Solution) |
| :--- | :--- | :--- |
| **청킹** | PDF → Markdown 변환 후 `##`, `-`, 이미지 플레이스홀더 등 불필요 기호가 잔류하여 법령 구조 정규식에 오탐 발생 | • **리서치:** `pymupdf4llm` 변환 출력 패턴 분석<br>• **적용:** `_normalize_text()` 전처리 함수로 변환 아티팩트를 제거한 뒤 청킹 수행 |
| **청킹** | 일부 문서에 법령 계층 구조가 없어 `LawChunker` 적용 시 빈 노드 생성 또는 파싱 실패 | • **리서치:** LlamaIndex `SentenceSplitter` 문서 검토<br>• **적용:** 샘플 페이지에서 구조 여부 사전 탐지 후, 실패 시 `FallbackChunker`로 전환하는 2단계 전략 적용 (commit `3dc6120`) |
| **에이전트** | 상담사 에이전트가 법령 외 가이드 문서도 검색 대상에 포함하여 불필요한 컨텍스트 혼입 | • **리서치:** LangGraph 노드별 분기 처리 패턴 검토<br>• **적용:** `agent_mode` 기반 예외 처리로 가이드 문서를 쿼리 엔진 대상에서 명시적으로 제외 (commit `1ae763e`) |
| **구조화 출력** | `NoticeScanAgent`의 Pydantic 스키마 필드 제약 미비로 LLM 출력값이 비표준 형식으로 반환되어 파싱 오류 발생 | • **리서치:** Pydantic Field constraints, LangChain `with_structured_output()` 동작 방식 분석<br>• **적용:** description 강화 + Literal 타입으로 허용값 명시, 오류 시 Error Handler 노드로 분기 (commit `a7b561b`, `f9f425e`) |
| **도구 연동** | MCP 클라이언트가 서버 SSE 엔드포인트 연결 전에 세션을 초기화하려 해 Race Condition 발생 | • **리서치:** MCP SSE 프로토콜 핸드셰이크 순서 분석<br>• **적용:** `SESSION_INIT_WAIT = 1` 초 대기 + 재시도 로직 추가 |
| **성능/기타** | Windows 환경에서 NumPy/scikit-learn OpenMP 라이브러리 중복 로딩으로 경고 및 프로세스 불안정 발생 | • **리서치:** Intel OpenMP + MS OpenMP 충돌 이슈 확인 (Stack Overflow, NumPy GitHub)<br>• **적용:** `os.environ["KMP_DUPLICATE_LIB_OK"] = "True"` 환경변수 설정 |


---
핵심 동작 검증

위에서 구현한 기능이 의도대로 동작하는지 보여주는 대표적인 실행 결과를 첨부합니다.

[검증 시나리오: 법령 기반 조달 질문 → 에이전트 자동 라우팅 및 답변 생성]

- 입력: "소프트웨어 유지보수 용역 발주 시 하도급 제한 기준이 어떻게 되나요?"
- 에이전트 동작:

  a. QueryRewrite 노드: 대화 히스토리 없음 → 원문 그대로 사용
  b. DocumentSearch 노드: 업로드 파일 없음 → 스킵
  c. 조건 분기: enableExtDocse=false → LawBase 경로 선택
  d. LLM이 질문 유형 판단 → agent_mode = "PpsTechnicalServicesAgent" 자동 선택
  e. qna_law_base MCP 도구 호출: "소프트웨어 유지보수 하도급 제한" 쿼리 최적화 → FAISS에서 관련 조문 상위 20개 검색
  f. LawChunker Parent 노드로 조문 전문 컨텍스트 복원 → LLM 최종 답변 생성
  g. 답변 SSE 스트리밍 출력 + DB 저장
- 최종 결과:

▎ [에이전트 답변 예시] "소프트웨어 유지보수 용역의 경우 「소프트웨어 진흥법」 제20조에 따라 원칙적으로 하도급이 제한됩니다. 다만, 발주기관의 사전 승인을 받은 경우 또는 전문 분야별 업무 특성상 불가피한 경우에 한해 계약금액의 50% 미만 범위에서 허용됩니다..."

---
[검증 시나리오: 입찰공고 PDF 업로드 → 구조화 데이터 자동 추출 및 위반 감지]

- 입력: 입찰공고 PDF 파일 업로드 (NoticeScanAgent)
- 에이전트 동작:

  a. ConvertToMarkdown 노드: pymupdf4llm으로 PDF → Markdown 변환 (표·레이아웃 보존), 진행률 SSE 전송
  b. ExtractMetadata 노드: LLM이 ProcurementData Pydantic 스키마에 맞춰 공고명, 수요기관, 납품기한, 자격조건, 계약방식 등 20여 개 필드 추출
  c. 규정 위반 항목 감지 (예: 특정 업체 제품 사양 명시 여부)
  d. 추출 실패 필드는 null + 사유 기록 (미추론 원칙 준수)
- 최종 결과:

{
  "generalInfo": {
    "noticeName": "사무용 소프트웨어 비축입찰 공고",
    "demandOrganization": "조달청",
    "contractMethod": "일반경쟁"
  },
  "eligibility": {
    "qualification": "소프트웨어 사업자 신고업체",
    "restrictedItems": null
  },
  "violations": ["특정 제조사 모델명 명시 (브랜드 제한 의심)"]
}
