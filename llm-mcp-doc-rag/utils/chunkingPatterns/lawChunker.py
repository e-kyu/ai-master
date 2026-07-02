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
    # 4. 별표 섹션 파서 (표는 쪼개지 않고, 소제목 단위로만 child를 나눈다)
    # ============================================================
    def _parse_byeolpyo_section(self, section, law_title):
        table_no = section["meta"]["table_no"]
        header = section["header"]
        body = section["body"]
        full_text = f"{header}\n{body}"

        parent_node = TextNode(
            text=full_text,
            metadata={
                "law_title": law_title,
                "doc_type": "별표",
                "table_no": table_no,
                "table_title": header,
                "type": "parent",
                "cross_refs": self._extract_cross_refs(full_text),
            },
        )

        sub_re = re.compile(_BEOPYO_SUB_RE, re.MULTILINE)
        matches = list(sub_re.finditer(body))
        child_chunks = []

        if not matches:
            # 소제목이 없으면 표 전체 보존을 우선해 단일 child로 처리한다.
            if body.strip():
                child_chunks.append((header, body.strip()))
        else:
            if matches[0].start() > 0:
                preface = body[: matches[0].start()].strip()
                if preface:
                    child_chunks.append(("서문", preface))
            for idx, m in enumerate(matches):
                sub_title = m.group(0).strip()
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
                content = body[start:end].strip()
                child_chunks.append((sub_title, content))

        child_nodes = []
        for sub_title, content in child_chunks:
            if not content.strip():
                continue
            contextualized_text = (
                f"별표: {table_no} ({header})\n"
                f"구분: {sub_title}\n"
                f"내용:\n{content}"
            )
            child_node = TextNode(
                text=contextualized_text,
                metadata={
                    "law_title": law_title,
                    "doc_type": "별표",
                    "table_no": table_no,
                    "table_title": header,
                    "section": sub_title,
                    "type": "child",
                    "contains_table": "|" in content,
                    "cross_refs": self._extract_cross_refs(content),
                },
            )
            i_node = IndexNode.from_text_node(child_node, index_id=parent_node.node_id)
            child_nodes.append(i_node)

        self.logger.debug(f"[LawChunker] 별표 '{table_no}' 파싱 완료 - child {len(child_nodes)}개")
        return parent_node, child_nodes

    # ============================================================
    # 5. 조문 섹션 파서
    #    항(①②③)이 있으면 항 단위로 분할하고, 없으면 호(1. 2. 3.) 단위로
    #    분할한다. 한 조문 안에 항/호 표기가 동시에 등장하는 경우는 실무상
    #    거의 없어(예: "정의" 조문은 호만, 대부분의 조문은 항만 사용) 이
    #    2단 우선순위만으로 충분하며 후보 스코어링 같은 추가 로직은 불필요하다.
    # ============================================================
    def _parse_article_section(self, section, law_title):
        article_no = section["meta"].get("article_no")
        title = section["meta"]["article_title"]
        header = section["header"]
        content = section["body"]
        full_text = f"{header}\n{content}"

        parent_node = TextNode(
            text=full_text,
            metadata={
                "law_title": law_title,
                "doc_type": "조문",
                "article_no": article_no,
                "article_title": title,
                "type": "parent",
                "cross_refs": self._extract_cross_refs(full_text),
            },
        )

        if re.search(_PARAGRAPH_RE, content):
            split_re = re.compile(f'({_PARAGRAPH_RE})')
        elif re.search(_ITEM_RE, content, re.MULTILINE):
            split_re = re.compile(f'({_ITEM_RE})', re.MULTILINE)
        else:
            split_re = None

        child_chunks = []
        if split_re is None:
            # 항/호 구분이 전혀 없는 짧은 조문은 본문 전체를 하나의 child로 둔다.
            if content.strip():
                child_chunks.append(("", content.strip()))
        else:
            parts = split_re.split(content)
            if parts and not split_re.match(parts[0].strip()):
                first_text = parts.pop(0).strip()
                if first_text:
                    child_chunks.append(("", first_text))
            for j in range(0, len(parts), 2):
                if j + 1 < len(parts):
                    marker = parts[j].strip()
                    body_text = parts[j + 1].strip()
                    child_chunks.append((marker, body_text))

        child_nodes = []
        for marker, chunk in child_chunks:
            if not chunk.strip():
                continue
            breadcrumb = f"{title}" + (f" {marker}" if marker else "")
            # 자식 검색 시 상위 맥락 유실을 막기 위해 법률명/조항 정보를 본문 앞에 주입한다.
            contextualized_text = (
                f"법률명: {law_title}\n"
                f"조항: {breadcrumb}\n"
                f"내용: {chunk}"
            )
            child_node = TextNode(
                text=contextualized_text,
                metadata={
                    "law_title": law_title,
                    "doc_type": "조문",
                    "article_no": article_no,
                    "article_title": title,
                    "paragraph_no": marker,
                    "breadcrumb": breadcrumb,
                    "type": "child",
                    "cross_refs": self._extract_cross_refs(chunk),
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
                if section["body"].strip():
                    p_node = TextNode(
                        text=section["body"],
                        metadata={"law_title": law_title, "doc_type": "intro", "type": "parent"},
                    )
                    all_nodes.append(p_node)
                    node_dict[p_node.node_id] = p_node

            elif section["type"] == "조문":
                parent_node, child_nodes = self._parse_article_section(section, law_title)
                node_dict[parent_node.node_id] = parent_node
                all_nodes.append(parent_node)
                all_nodes.extend(child_nodes)

            elif section["type"] == "별표":
                parent_node, child_nodes = self._parse_byeolpyo_section(section, law_title)
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

