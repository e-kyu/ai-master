import mcp.types as types

def get_list() -> list[types.Tool]:
    return [
        # ─────────────────────────────────────
        # 문서 기반 QnA
        # ─────────────────────────────────────
        types.Tool(
            name="qna_doc",
            description=(
                "사용자가 첨부한 문서가 존재 하는 경우 문서를 분석하여 문서 내용에 근거한 답변을 생성한다."
                "'파일', '경로', '문서 기반' 언급이 있는 경우"
            ),
            inputSchema={
                "type": "object",
                "required": ["question", "fileFullPath"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "문서를 기반으로 한 질문 문장"
                    },
                    "fileFullPath": {
                        "type": "string",
                        "description": "분석할 문서의 전체 파일 경로"
                    },
                    "allow_search": {
                        "type": "boolean",
                        "description": "인터넷 검색 허용 여부. False일 경우 도구 사용 불가.",
                        "default": True
                    }
                }
            }
        ),

        # ─────────────────────────────────────
        # 전문 Agent의 QnA
        # ─────────────────────────────────────
        types.Tool(
            name="qna_law_base",
            description=(
                "전문 Agent mode에 따라 기본 참조 문서 내용에 근거한 답변을 생성한다."
            ),
            inputSchema={
                "type": "object",
                "required": ["question", "agent_mode"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "문서를 기반으로 한 질문 문장"
                    },
                    "agent_mode": {
                        "type": "string",
                        "description": ("사용자 질문에 적합한 에이전트를 아래 목록에서 '하나만' 선택:\n"
                                        "- PpsGeneralServiceAgent: 조달청 일반용역 전문 상담\n"
                                        "- PpsTechnicalServicesAgent: 조달청 기술용역 전문 상담\n"
                                        "- PpsConstructionAgent: 조달청 시설공사 전문 상담\n"
                                        "- PpsProductsAgent: 조달청 물품 관련 전문 상담\n"
                                        "- PpsStockpilingAgent: 조달청 비축 관련 전문 상담")
                    },
                    "fileFullPath": {
                        "type": "string",
                        "description": "분석할 문서의 전체 파일 경로"
                    },
                    "allow_search": {
                        "type": "boolean",
                        "description": "인터넷 검색 허용 여부. False일 경우 도구 사용 불가.",
                        "default": True
                    }
                }
            }
        ),

        # ─────────────────────────────────────
        # 웹 검색(RAG) 기반 QnA
        # ─────────────────────────────────────
        types.Tool(
            name="qna_web_search",
            description=(
                "인터넷 검색 도구"
                "  - 최신 정보, 일반 지식, 뉴스, 법령 변경, 외부 정보가 필요한 경우 사용합니다.\n"
                "  - 단, 'allow_search' 파라미터가 False인 경우에는 이 도구를 호출해서는 안 됩니다."
            ),
            inputSchema={
                "type": "object",
                "required": ["question", "allow_search"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "사용자가 입력한 질문 문장"
                    },
                    "allow_search": {
                        "type": "boolean",
                        "description": "인터넷 검색 허용 여부. False일 경우 도구 사용 불가.",
                        "default": True
                    }
                }
            }
        ),

        # ─────────────────────────────────────
        # 문서 변환 (HTML/PDF -> Markdown)
        # ─────────────────────────────────────
        types.Tool(
            name="convert_to_markdown",
            description=(
                "HTML 또는 PDF 파일을 Markdown 텍스트로 변환하는 도구입니다.\n"
                "  - 사용자가 업로드한 공고문, 문서 파일(.html, .htm, .pdf)의 내용을 분석하기 전,"
                " 텍스트 형태로 추출해야 할 때 사용합니다.\n"
                "  - 'file_path'는 서버에서 접근 가능한 절대/상대 경로여야 합니다.\n"
                "  - 지원하지 않는 확장자인 경우 실패 상태(status='failed')와 에러 메시지를 반환합니다."
            ),
            inputSchema={
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "변환할 파일의 경로 (.html, .htm, .pdf 지원)"
                    }
                }
            }
        ),
    ]