import re
import os

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode, TextNode
from utils import loggerUtil


# ============================================================
# 고정 정규식
#
# 예전에는 조문/항/별표/부칙 표기마다 후보 정규식을 8~10개씩 나열해두고
# 문서 표본에서 가장 매칭이 많은 것을 "동적으로" 골라 쓰는 방식이었다.
# 하지만 이 파이프라인이 실제로 다루는 문서는 전부 조달청 규정/훈령류 한국어
# 법령이고, 표기 형식은 두 갈래뿐이다.
#   1) SimpleDirectoryReader로 읽은 PDF 원문 텍스트 (마크다운 장식 없음)
#   2) tools/convertToMarkdown.py(ConvertToMarkdownAgent)로 변환한 마크다운
#      - pymupdf4llm/markitdown이 문단마다 헤딩('#')이나 목록 기호('-')를 붙여서
#        내보낸다. 즉 "## [별표1-1] 제목", "- 제1조(목적) 본문..." 처럼 원문에는
#        없던 장식이 앞에 붙는다.
# 두 형식을 각각 분기 처리하면 로직이 두 배로 늘어나므로, _normalize_text()에서
# 장식만 한 번에 제거해 두 입력을 완전히 같은 모양으로 맞추고, 이후의 정규식/
# 상태머신은 입력 형식을 몰라도 되도록 만든다. "Article", "Chapter" 같은 국제
# 문서용 후보나 후보 나열 + 점수 계산(pick_best) 로직은 애초에 불필요해 제거했다.
# ============================================================
_ARTICLE_RE = r'제\d+조(?:의\d+)?\([^\n)]*\)'                       # 제N조(제목) / 제N조의M(제목)
_PARAGRAPH_RE = r'[①-⑳]'                                          # 항: 원문자 ①②③...
_ITEM_RE = r'^\s*\d{1,2}\.\s+'                                     # 호: 줄 시작의 "1. " "2. " 등
_TABLE_HEADER_RE = r'^\[?별표\s*[\d\-]+\]?'                         # 별표 헤딩 (예: "[별표1-1] 제목", "별표 1")
_BEOPYO_SUB_RE = r'^\d+\.\s*[^\n(]+?(?:\(\s*[\d,]+\s*점\s*\))?\s*$'  # 별표 소제목 (예: "1.수행능력평가(30점)")
_ADDENDUM_RE = r'^부\s*칙\s*<제?\s*(\S+?)\s*호,?\s*([\d.]+)>'        # 부칙 <제N호, 날짜> (변환 시 "부  칙"처럼 사이가 벌어지기도 함)
_CROSS_REF_RE = r'(별표\s*\d+|별지\s*제?\s*\d+\s*호?|제\d+조(?:의\d+)?(?:\s*[①-⑳])?)'  # 상호참조 토큰

# 마크다운 변환본에서 문단 앞에 붙는 장식들. _normalize_text()가 줄마다 이 장식만
# 제거하므로, 위 정규식들은 '#'나 '-' 접두어를 신경 쓸 필요가 없다.
_MD_HEADING_RE = re.compile(r'^\s*#{1,6}\s*')          # "## 제목" -> "제목"
_MD_BULLET_RE = re.compile(r'^\s*[-*•]\s+')            # "- 제1조(목적)..." -> "제1조(목적)..."
_MD_PICTURE_PLACEHOLDER_RE = re.compile(r'^\s*\*\*==>.*<==\*\*\s*$')  # 이미지 생략 표시줄(잡음)

# ============================================================
# 청크 크기 상한
#
# 실제 법령 문서는 조문 하나(①②③... 안에 1./2. 호, 가./나. 목, 1)/2) 사이목,
# 나-1./나-2. 세부항목까지 중첩되기도 하고, 별표는 소수점 번호(4.1, 4.1.1 ...)를
# 쓰는 다단 채점표가 여러 페이지에 걸치기도 한다. 이 모든 중첩 표기 방식을 정규식
# 으로 일일이 받아내려 하면 로직이 끝없이 늘어나고 문서마다 또 예외가 생긴다.
# 대신 "자연스러운 구분자(항/소제목)로 한 번 쪼갠 뒤, 그래도 크면 글자 수 기준
# 으로 한 번 더 쪼갠다"는 안전장치 하나로 문서 형식과 무관하게 부모/자식 노드
# 크기를 모두 통제한다. 예전에는 조문/별표 "전체"를 통째로 parent 하나로 만들어서
# 부모 노드가 페이지 단위로 비대해지고, 그 결과 임베딩 품질과 검색 속도가 함께
# 나빠졌다 — 부모 크기 상한이 이 문제의 핵심 수정 지점이다.
# ============================================================
_MAX_PARENT_CHARS = 1200  # 부모 노드 목표 최대 글자수 - 넘으면 항/소제목 단위로 쪼개 별도 parent로 승격
_MAX_CHILD_CHARS = 400    # 자식 노드 목표 최대 글자수 - 넘으면 문장 단위로 추가 분할


class LawChunker:
    """
    한국 법령/훈령류 문서(조문-항-호 / 별표 / 부칙 구조)를 Parent-Child 계층으로 청킹한다.

    사용 흐름:
        chunker = LawChunker()
        stats = chunker.detect_structure(llama_docs)   # 이 문서에 조문/별표가 실제로 있는지만 확인
        if stats["has_article"] or stats["has_byeolpyo"]:
            all_nodes, node_dict = chunker.parse_to_hierarchical_nodes(llama_docs)
        else:
            fallback_chunker = FallbackChunker()
            all_nodes, node_dict = fallback_chunker.create_fallback_nodes(llama_docs)
    """

    def __init__(self, sample_size=10):
        self.logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")
        self.sample_size = sample_size
        # 항/소제목으로 쪼갠 뒤에도 _MAX_CHILD_CHARS를 넘는 조각을 추가로 잘라내는 안전장치.
        self._size_splitter = SentenceSplitter(chunk_size=_MAX_CHILD_CHARS, chunk_overlap=40)

    def _cap_text(self, text: str, max_chars: int) -> list:
        """
        text가 max_chars 이내면 그대로 1개, 넘으면 문장 단위로 추가 분할해 여러 조각으로
        반환한다. 항/호/목/사이목처럼 문서마다 제각각인 중첩 표기를 전부 정규식으로
        받아내는 대신, 이 크기 기준 안전장치 하나로 어떤 문서든 청크 크기를 보장한다.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]
        pieces = [p.strip() for p in self._size_splitter.split_text(text) if p.strip()]
        return pieces or [text]

    # ============================================================
    # 1. 문서 구조 탐지
    #    정규식이 고정값이므로 "탐지"라는 이름과 달리 실제로 하는 일은
    #    표본 텍스트에서 각 패턴이 몇 번 매칭되는지 세는 것뿐이다.
    #    이 결과로 "조문 기반 파싱을 시도할지 / 기본 SentenceSplitter로
    #    폴백할지"만 판단하면 충분하다.
    # ============================================================
    def detect_structure(self, llama_docs):
        sample = self._sample_text(llama_docs)

        article_matches = len(re.findall(_ARTICLE_RE, sample))
        table_matches = len(re.findall(_TABLE_HEADER_RE, sample, re.MULTILINE))
        addendum_matches = len(re.findall(_ADDENDUM_RE, sample, re.MULTILINE))

        self.logger.info(
            f"[LawChunker] 구조 탐지 결과 - 조문:{article_matches} "
            f"별표:{table_matches} 부칙:{addendum_matches}"
        )

        return {
            "has_article": article_matches > 0,
            "has_byeolpyo": table_matches > 0,
            "has_addendum": addendum_matches > 0,
            "article_matches": article_matches,
            "table_matches": table_matches,
            "addendum_matches": addendum_matches,
        }

    def _sample_text(self, llama_docs):
        """문서 페이지가 많으면 균등 간격으로 sample_size개만 뽑아 탐지 비용을 줄인다."""
        if not llama_docs:
            return ""
        if len(llama_docs) <= self.sample_size:
            texts = [doc.text for doc in llama_docs]
        else:
            step = len(llama_docs) / self.sample_size
            indices = [min(int(i * step), len(llama_docs) - 1) for i in range(self.sample_size)]
            texts = [llama_docs[idx].text for idx in indices]
        return self._normalize_text("\n".join(texts))

    # ============================================================
    # 0-1. PDF 원문 텍스트 / 마크다운 변환본을 같은 모양으로 맞추는 전처리
    #      (조문/별표/부칙 탐지·분리 로직 전체가 이 정규화된 텍스트만 보고 동작한다)
    # ============================================================
    def _normalize_text(self, full_text):
        """
        pymupdf4llm/markitdown으로 변환된 마크다운은 문단마다 헤딩('#') 또는 목록
        기호('-')가 붙어 나오고, 이미지가 있던 자리에는 "**==> picture ... <==**"
        같은 잡음 줄이 끼어든다. 반면 SimpleDirectoryReader가 읽은 PDF 원문 텍스트는
        이런 장식이 전혀 없다.

        조문/별표/부칙 탐지는 전부 "줄 시작이 어떤 패턴인가"를 기준으로 동작하므로,
        입력이 어느 쪽이든 여기서 장식만 제거해 두 형식을 완전히 같은 모양으로
        맞춘다. 이렇게 하면 뒤따르는 _split_into_sections/_parse_* 로직은 입력이
        PDF인지 마크다운인지 전혀 신경 쓸 필요가 없어진다.
        """
        lines = []
        dropped_pictures = 0
        for line in full_text.split("\n"):
            if _MD_PICTURE_PLACEHOLDER_RE.match(line):
                dropped_pictures += 1
                continue
            line = _MD_HEADING_RE.sub('', line)
            line = _MD_BULLET_RE.sub('', line)
            lines.append(line)

        if dropped_pictures:
            self.logger.debug(f"[LawChunker] 이미지 생략 표시줄 {dropped_pictures}개 제거")

        return "\n".join(lines)

    # ============================================================
    # 2. 섹션 분류 상태머신
    #    줄 단위로 훑으며 조문 / 별표 / 부칙 / intro(서두) 섹션으로 나눈다.
    #
    #    re.split을 한 번에 돌리는 대신 상태머신을 쓰는 이유:
    #    부칙 문구 안에는 "제1조(시행일)", "제2조(...)"처럼 본문 조문과 똑같은
    #    형식의 하위 조항이 들어있다. 현재 상태가 "부칙"인 동안 조문 헤더를
    #    만나도 새 섹션으로 취급하지 않고 부칙 본문에 그대로 흡수시켜야, 부칙의
    #    "제1조"가 본문의 진짜 "제1조(목적)"와 뒤섞여 별개 조문으로 잘못
    #    쪼개지는 것을 막을 수 있다.
    # ============================================================
    def _split_into_sections(self, full_text: str):
        # PDF 원문/마크다운 변환본을 같은 모양으로 맞춘 뒤 한 줄씩 훑는다.
        full_text = self._normalize_text(full_text)

        article_re = re.compile(_ARTICLE_RE)
        table_re = re.compile(_TABLE_HEADER_RE, re.MULTILINE)
        addendum_re = re.compile(_ADDENDUM_RE, re.MULTILINE)

        # 별표/부칙 번호 추출용 (헤더 정규식이 그룹을 못 잡는 경우를 대비한 보강 정규식).
        # 별표는 "별표1-1"처럼 하이픈 붙은 하위 번호까지 그대로 잡아야 서로 다른
        # 하위 표(1-1, 1-2 ...)가 같은 table_no로 뭉개지지 않는다.
        table_no_re = re.compile(r'별표\s*([\d\-]+)')
        addendum_no_re = re.compile(r'제?\s*(\S+?)\s*호,?\s*([\d.]+)')

        lines = full_text.split("\n")
        sections = []
        current = {"type": "intro", "header": "", "body": [], "meta": {}}

        def flush():
            if current["body"] or current["header"]:
                current["body"] = "\n".join(current["body"]).strip()
                sections.append(current.copy())

        for line in lines:
            stripped = line.strip()

            # 정규화(_normalize_text)로 '#'/'-' 장식은 이미 제거됐지만 일반적인
            # 들여쓰기 공백은 남아있을 수 있어, 앞뒤 공백을 걷어낸 stripped를
            # 기준으로 매칭한다.
            m_table = table_re.match(stripped)
            m_addendum = addendum_re.match(stripped)
            # "제"로 시작하는 줄만 조문 정규식을 검사해 불필요한 매칭 시도를 줄인다.
            # search가 아닌 match를 써서 줄 맨 앞에서부터 정확히 조문 제목과 일치할
            # 때만 헤더로 인정한다(예: "제1조 관련 제2조(목적)..." 같은 본문 중간의
            # 조문 참조가 새 섹션으로 잘못 끊기지 않도록).
            m_article = article_re.match(stripped) if stripped.startswith("제") else None

            if m_table:
                flush()
                no_match = table_no_re.search(line)
                current = {
                    "type": "별표",
                    "header": stripped,
                    "body": [],
                    "meta": {"table_no": f"별표{no_match.group(1)}" if no_match else stripped},
                }
                continue

            if m_addendum:
                flush()
                no_match = addendum_no_re.search(line)
                current = {
                    "type": "부칙",
                    "header": stripped,
                    "body": [],
                    "meta": {
                        "addendum_no": no_match.group(1) if no_match else None,
                        "addendum_date": no_match.group(2) if no_match else None,
                    },
                }
                continue

            # 별표/부칙 내부에서는 조문과 같은 표기가 나와도 새 섹션을 만들지 않는다.
            if m_article and current["type"] not in ("별표", "부칙"):
                flush()
                # PDF 원문/마크다운 변환본 모두 조문 제목과 첫 문장이 같은 줄에
                # 붙어 나온다(예: "제1조(목적) 이 규정은 ...을 목적으로 한다.").
                # 제목만 header로 뽑아내고, 뒤에 남은 문장은 버리지 않고 본문의
                # 첫 줄로 편입시킨다. 그렇지 않으면 짧은 조문은 항/호 분할 대상인
                # content가 비어버려 자식 노드가 아예 만들어지지 않는다.
                header_text = m_article.group(0)
                remainder = stripped[m_article.end():].strip()
                current = {
                    "type": "조문",
                    "header": header_text,
                    "body": [remainder] if remainder else [],
                    "meta": {
                        "article_no": header_text,
                        "article_title": header_text,
                    },
                }
                continue

            current["body"].append(line)

        flush()

        # 부칙이 여러 개면 마지막 부칙만 "현재 유효한 개정"으로 표시한다.
        addendum_indices = [i for i, s in enumerate(sections) if s["type"] == "부칙"]
        for i in addendum_indices:
            sections[i]["meta"]["is_latest"] = (i == addendum_indices[-1])

        self.logger.debug(
            f"[LawChunker] 섹션 분리 완료 - 총 {len(sections)}개 "
            f"(조문:{sum(1 for s in sections if s['type'] == '조문')} "
            f"별표:{sum(1 for s in sections if s['type'] == '별표')} "
            f"부칙:{len(addendum_indices)})"
        )
        return sections

    # ============================================================
    # 3. cross-ref 추출: 본문 안에서 "별표 N", "제N조" 같은 상호참조 토큰을 모은다.
    # ============================================================
    def _extract_cross_refs(self, text: str):
        refs = set()
        for m in re.finditer(_CROSS_REF_RE, text):
            token = m.group(0).strip()
            if not token:
                continue
            nums = re.findall(r'별표\s*(\d+)', token)
            if nums:
                for n in nums:
                    refs.add(f"별표{n}")
            else:
                refs.add(re.sub(r'\s+', '', token))
        return sorted(refs)

    # ============================================================
    # 4. 별표 섹션 파서
    #    별표 전체(표+세부규정)가 이미 작으면 기존처럼 별표 전체를 하나의 parent로
    #    쓴다. 하지만 [별표 1]처럼 여러 페이지에 걸친 채점표는 별표 전체를 parent
    #    하나로 만들면 부모 노드가 지나치게 비대해져 임베딩/검색 속도가 나빠진다.
    #    이 경우 감지된 소제목(예: "1. 수행능력 평가(30점)") 단위로 parent를 여러
    #    개로 쪼개고, 소제목이 아예 없으면 크기 기준으로만 잘라 여러 parent를
    #    만든다 — 표 구조를 보존하려던 기존 "단일 child로 전체 보존" 전략은
    #    다단 채점표에서는 오히려 검색을 느리게 만들므로 더 이상 쓰지 않는다.
    #    반환값은 (parent_node, child_nodes) 쌍의 리스트다.
    # ============================================================
    def _parse_byeolpyo_section(self, section, law_title):
        table_no = section["meta"]["table_no"]
        header = section["header"]
        body = section["body"]
        full_text = f"{header}\n{body}".strip()

        sub_re = re.compile(_BEOPYO_SUB_RE, re.MULTILINE)
        matches = list(sub_re.finditer(body))

        if len(full_text) <= _MAX_PARENT_CHARS or not matches:
            return [self._build_byeolpyo_parent(table_no, header, body, law_title)]

        # 소제목 단위로 쪼개 각 소제목을 독립된(더 작은) parent로 승격시킨다.
        results = []
        preface = body[: matches[0].start()].strip() if matches[0].start() > 0 else ""

        for idx, m in enumerate(matches):
            sub_title = m.group(0).strip()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
            sub_content = body[start:end].strip()
            if preface:
                sub_content = f"{preface}\n{sub_content}".strip()
                preface = ""  # 서두 문구는 첫 소제목에만 한 번 붙인다
            results.append(
                self._build_byeolpyo_parent(
                    table_no, f"{header} - {sub_title}", sub_content, law_title, section_title=sub_title
                )
            )

        self.logger.debug(f"[LawChunker] 별표 '{table_no}' 크기 초과로 소제목 {len(results)}개 parent로 분할")
        return results

    def _build_byeolpyo_parent(self, table_no, header, content, law_title, section_title=None):
        """별표(또는 별표 소제목) 하나를 parent 노드 1개 + 크기 상한을 지킨 child 노드들로 만든다."""
        full_text = f"{header}\n{content}".strip()
        parent_node = TextNode(
            text=full_text,
            metadata={
                "law_title": law_title,
                "doc_type": "별표",
                "table_no": table_no,
                "table_title": header,
                "section": section_title,
                "type": "parent",
                "cross_refs": self._extract_cross_refs(full_text),
            },
        )

        pieces = self._cap_text(content, _MAX_CHILD_CHARS)
        child_nodes = []
        for idx, piece in enumerate(pieces):
            label = section_title or header
            if len(pieces) > 1:
                label = f"{label} ({idx + 1}/{len(pieces)})"
            contextualized_text = (
                f"별표: {table_no} ({header})\n"
                f"구분: {label}\n"
                f"내용:\n{piece}"
            )
            child_node = TextNode(
                text=contextualized_text,
                metadata={
                    "law_title": law_title,
                    "doc_type": "별표",
                    "table_no": table_no,
                    "table_title": header,
                    "section": label,
                    "type": "child",
                    "contains_table": "|" in piece,
                    "cross_refs": self._extract_cross_refs(piece),
                },
            )
            i_node = IndexNode.from_text_node(child_node, index_id=parent_node.node_id)
            child_nodes.append(i_node)

        self.logger.debug(f"[LawChunker] 별표 '{table_no}' ({header}) 파싱 완료 - child {len(child_nodes)}개")
        return parent_node, child_nodes

    # ============================================================
    # 5. 조문 섹션 파서
    #    조문 전체(제목+①~⑨ 전부)가 이미 작으면 기존처럼 조문 전체를 하나의
    #    parent로 쓴다. 하지만 항이 여러 개거나 항 하나에 호/목/사이목까지 깊게
    #    중첩된 조문(예: 제2조③이 1.~7.호, 그 안에 가.~아.목까지 갖는 경우)은
    #    조문 전체를 parent 하나로 만들면 부모 노드가 지나치게 비대해진다.
    #    이 경우 항(①②③...) 단위로 parent를 여러 개로 쪼갠다. 항조차 없는
    #    조문은 기존처럼 조문 전체가 parent다. 각 parent 안에서는 호(1. 2. 3.)로
    #    한 번 더 나누고, 그래도 크면 크기 상한(_cap_text)으로 추가 분할한다 —
    #    가/나/목이나 사이목, 소수점 번호 같은 세부 표기까지 정규식으로 쫓아가는
    #    대신 이 크기 상한이 안전망 역할을 한다.
    #    반환값은 (parent_node, child_nodes) 쌍의 리스트다.
    # ============================================================
    def _parse_article_section(self, section, law_title):
        article_no = section["meta"].get("article_no")
        title = section["meta"]["article_title"]
        header = section["header"]
        content = section["body"]
        full_text = f"{header}\n{content}".strip()

        if len(full_text) <= _MAX_PARENT_CHARS or not re.search(_PARAGRAPH_RE, content):
            return [self._build_article_parent(header, title, article_no, content, law_title)]

        # 항(①②③...) 단위로 쪼개 각 항을 독립된(더 작은) parent로 승격시킨다.
        paragraph_re = re.compile(f'({_PARAGRAPH_RE})')
        parts = paragraph_re.split(content)

        preface = ""
        if parts and not paragraph_re.match(parts[0].strip()):
            preface = parts.pop(0).strip()

        results = []
        for j in range(0, len(parts), 2):
            if j + 1 >= len(parts):
                continue
            marker = parts[j].strip()
            para_content = parts[j + 1].strip()
            if preface:
                para_content = f"{preface}\n{para_content}".strip()
                preface = ""  # 서두 문구는 첫 항에만 한 번 붙인다
            results.append(
                self._build_article_parent(
                    f"{header} {marker}", f"{title} {marker}", article_no, para_content,
                    law_title, paragraph_no=marker,
                )
            )

        self.logger.debug(f"[LawChunker] {title} 크기 초과로 항 {len(results)}개 parent로 분할")
        return results

    def _build_article_parent(self, header, title, article_no, content, law_title, paragraph_no=None):
        """조문(또는 조문의 항 하나) 을 parent 노드 1개 + 크기 상한을 지킨 child 노드들로 만든다."""
        full_text = f"{header}\n{content}".strip()
        parent_node = TextNode(
            text=full_text,
            metadata={
                "law_title": law_title,
                "doc_type": "조문",
                "article_no": article_no,
                "article_title": title,
                "paragraph_no": paragraph_no,
                "type": "parent",
                "cross_refs": self._extract_cross_refs(full_text),
            },
        )

        if re.search(_ITEM_RE, content, re.MULTILINE):
            item_re = re.compile(f'({_ITEM_RE})', re.MULTILINE)
            parts = item_re.split(content)
            child_chunks = []
            if parts and not item_re.match(parts[0].strip()):
                first_text = parts.pop(0).strip()
                if first_text:
                    child_chunks.append(("", first_text))
            for j in range(0, len(parts), 2):
                if j + 1 < len(parts):
                    child_chunks.append((parts[j].strip(), parts[j + 1].strip()))
        else:
            # 호 구분이 없으면 본문 전체를 하나의 조각으로 두고, 크기 상한이 필요시 나눈다.
            child_chunks = [("", content.strip())] if content.strip() else []

        child_nodes = []
        for marker, chunk in child_chunks:
            if not chunk.strip():
                continue
            pieces = self._cap_text(chunk, _MAX_CHILD_CHARS)
            for idx, piece in enumerate(pieces):
                breadcrumb = title + (f" {marker}" if marker else "")
                if len(pieces) > 1:
                    breadcrumb = f"{breadcrumb} ({idx + 1}/{len(pieces)})"
                # 자식 검색 시 상위 맥락 유실을 막기 위해 법률명/조항 정보를 본문 앞에 주입한다.
                contextualized_text = (
                    f"법률명: {law_title}\n"
                    f"조항: {breadcrumb}\n"
                    f"내용: {piece}"
                )
                child_node = TextNode(
                    text=contextualized_text,
                    metadata={
                        "law_title": law_title,
                        "doc_type": "조문",
                        "article_no": article_no,
                        "article_title": title,
                        "paragraph_no": paragraph_no,
                        "item_no": marker,
                        "breadcrumb": breadcrumb,
                        "type": "child",
                        "cross_refs": self._extract_cross_refs(piece),
                    },
                )
                i_node = IndexNode.from_text_node(child_node, index_id=parent_node.node_id)
                child_nodes.append(i_node)

        self.logger.debug(f"[LawChunker] {title} 파싱 완료 - child {len(child_nodes)}개")
        return parent_node, child_nodes

    # ============================================================
    # 6. 메인 파서: 문서 전체를 섹션으로 나눈 뒤 섹션 타입별로 위임한다.
    # ============================================================
    def parse_to_hierarchical_nodes(self, llama_docs):
        all_nodes = []
        node_dict = {}

        # 페이지 경계에서 조문이 끊기지 않도록 전체 문서를 하나의 텍스트로 합친다.
        full_text = "\n".join([doc.text for doc in llama_docs])

        law_title = "답변자료"
        if llama_docs:
            first_meta = getattr(llama_docs[0], "metadata", {})
            if isinstance(first_meta, dict):
                fname = first_meta.get("file_name")
                if fname:
                    law_title = os.path.splitext(fname)[0]

        sections = self._split_into_sections(full_text)
        all_table_nos = [s["meta"]["table_no"] for s in sections if s["type"] == "별표"]

        for section in sections:
            if section["type"] == "intro":
                # 서두 문구도 크기 상한을 넘으면 여러 parent로 나눠 통제한다.
                for piece in self._cap_text(section["body"], _MAX_PARENT_CHARS):
                    p_node = TextNode(
                        text=piece,
                        metadata={"law_title": law_title, "doc_type": "intro", "type": "parent"},
                    )
                    all_nodes.append(p_node)
                    node_dict[p_node.node_id] = p_node

            elif section["type"] == "조문":
                for parent_node, child_nodes in self._parse_article_section(section, law_title):
                    node_dict[parent_node.node_id] = parent_node
                    all_nodes.append(parent_node)
                    all_nodes.extend(child_nodes)

            elif section["type"] == "별표":
                for parent_node, child_nodes in self._parse_byeolpyo_section(section, law_title):
                    # 다른 별표들과 서로 참조할 수 있도록 형제 별표 목록을 메타데이터로 남긴다.
                    parent_node.metadata["peer_tables"] = [
                        t for t in all_table_nos if t != parent_node.metadata["table_no"]
                    ]
                    node_dict[parent_node.node_id] = parent_node
                    all_nodes.append(parent_node)
                    all_nodes.extend(child_nodes)

            elif section["type"] == "부칙":
                text = f"{section['header']}\n{section['body']}"
                node = TextNode(
                    text=text,
                    metadata={
                        "law_title": law_title,
                        "doc_type": "부칙",
                        "type": "parent",
                        "is_latest": section["meta"].get("is_latest", False),
                    },
                )
                node_dict[node.node_id] = node
                all_nodes.append(node)

        self.logger.info(f"[LawChunker] 계층 노드 생성 완료 - 총 {len(all_nodes)}개 (parent+child)")
        return all_nodes, node_dict

