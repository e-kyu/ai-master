import re
import os
import difflib
import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from llama_index.core.schema import IndexNode, TextNode
from utils import loggerUtil


# ============================================================
# 경계탐지(boundary-first) 정규식
#
# 상태머신으로 줄 단위를 훑는 대신, 전체 텍스트에서 헤더 위치만 먼저
# finditer로 모두 찾고 그 위치 기준으로 슬라이싱한다. 부칙 -> 별표 -> 조문
# 순서로 영역을 먼저 통째로 잘라내므로, 부칙 안에 조문과 동일한 표기
# ("제1조(시행일)")가 나와도 애초에 PAT_JOMUN을 그 영역에 적용하지 않아
# 오분류가 구조적으로 발생하지 않는다.
# ============================================================
PAT_JOMUN = re.compile(r'^제(\d+)조(?:의(\d+))?\(([^\n)]*)\)', re.M)
PAT_BYEOLPYO = re.compile(r'^\[?별표\s*([\d\-]+)\]?', re.M)
PAT_BUCHIL = re.compile(r'^부\s*칙\s*[<〈]\s*제?\s*(\S+?)\s*호\s*,?\s*([\d.]+)\s*[>〉]', re.M)

# cross-ref "언급" 추출 전용 (경계탐지용 PAT_JOMUN과 달리 제목이 없어도 매칭)
REF_PAT = re.compile(r'(별표\s*[\d\-]+|별지\s*제?\s*\d+\s*호?|제\d+조(?:의\d+)?(?:\s*[①-⑳])?(?:제\d+호)?)')
_TOK_ARTICLE_RE = re.compile(r'^제(\d+)조(?:의(\d+))?\s*([①-⑳])?')
_TOK_BYEOLPYO_RE = re.compile(r'^별표\s*([\d\-]+)')

# 부칙 시행일 문구
PAT_EFFECTIVE_LITERAL = re.compile(r'(\d{4})[.\s년]\s*(\d{1,2})[.\s월]\s*(\d{1,2})일?\s*부터\s*시행')
PAT_EFFECTIVE_ON_PROMULGATION = re.compile(r'공포한\s*날부터\s*시행')
PAT_EFFECTIVE_AFTER_DAYS = re.compile(r'공포\s*후\s*(\d+)일이?\s*경과한\s*날부터\s*시행')

# 마크다운 변환본 장식 제거용 (v1과 동일)
_MD_HEADING_RE = re.compile(r'^\s*#{1,6}\s*')
_MD_BULLET_RE = re.compile(r'^\s*[-*•]\s+')
_MD_PICTURE_PLACEHOLDER_RE = re.compile(r'^\s*\*\*==>.*<==\*\*\s*$')

# 표/산식 라인 판별 — 헤더 패턴 매칭보다 먼저 검사해서 "가. 50%미만 22.0" 같은
# 등급행이 목(가.나.다.) 헤더로 오분류되는 것을 막는다.
_MD_TABLE_ROW_RE = re.compile(r'^\|.*\|$')
_GRADE_ROW_RE = re.compile(r'^[A-Za-z가-힣]\s*[.\)]?\s*.{0,20}\d+(?:\.\d+)?\s*(?:점|%)')
_FORMULA_TOKEN_RE = re.compile(r'[×÷±≤≥∑√]|(?<=\d)\s*[/*]\s*(?=\d)')

# 조문 레벨 정규식 (항 -> 호 -> 목 -> 세목). 별표도 같은 스택 규약(_push/_absorb)을
# 공유하며, "목" 패턴은 별표에서도 그대로 재사용한다.
LEVEL_PATTERNS_JOMUN = [
    ("clause", re.compile(r'^([①-⑳])')),
    ("item", re.compile(r'^(\d+)\.\s')),
    ("sub", re.compile(r'^([가나다라마바사아자차카타파하])\.\s')),
    ("subsub", re.compile(r'^(\d+)\)\s')),
]
_LEVEL_SUFFIX = {"clause": "항", "item": "호", "sub": "목", "subsub": "세목"}

_DOTTED_RE = re.compile(r'^(\d+(?:\.\d+)+)\.?\s')   # "4.1", "4.1.1" (점이 1개 이상 있어야 함)
_PLAIN_ITEM_RE = re.compile(r'^(\d+)\.\s')          # "1. 수행능력평가(30점)" 형식 별표 소제목
_SUB_RE = LEVEL_PATTERNS_JOMUN[2][1]                # 가.나.다. 재사용

DEFAULT_CONFIG = {
    "max_parent_tokens": 300,
    "max_child_tokens": 100,
    "child_overlap_tokens": 15,
    "parallel_char_threshold": 20000,
    "reconstruction_min_ratio": 0.995,
}

CHILD_TEMPLATE = "{law_title} > {breadcrumb}\n내용: {content}"


# ============================================================
# 토큰 카운터 (tiktoken cl100k_base 근사치)
#
# 실제 임베딩 모델(KR-SBERT 등)의 토크나이저와는 다르지만, 모델에 무관하게
# 쓸 수 있는 가벼운 범용 근사치로 사용한다. buchil 헤더의 "<제3621호, ...>"
# 처럼 꺾쇠가 포함된 텍스트를 tiktoken이 특수토큰으로 오인해 예외를 던지지
# 않도록 disallowed_special=()로 비활성화한다.
# ============================================================
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_tokenizer().encode(text, disallowed_special=()))


def tiktoken_window_split(text: str, max_tokens: int, overlap_tokens: int) -> list:
    """토큰 슬라이딩 윈도우로 강제 분할한다. cl100k_base는 바이트 레벨 BPE라
    어느 지점에서 잘라 디코딩해도 깨진 UTF-8이 나오지 않는다."""
    enc = _get_tokenizer()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return [text]
    step = max(max_tokens - overlap_tokens, 1)
    pieces, start = [], 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        pieces.append(enc.decode(ids[start:end]))
        if end == len(ids):
            break
        start += step
    return pieces


# ============================================================
# 입력 정규화 (PDF 원문 / 마크다운 변환본을 같은 모양으로)
# ============================================================
def _normalize_text(full_text: str) -> str:
    lines = []
    for line in full_text.split("\n"):
        if _MD_PICTURE_PLACEHOLDER_RE.match(line):
            continue
        line = _MD_HEADING_RE.sub('', line)
        line = _MD_BULLET_RE.sub('', line)
        lines.append(line)
    return "\n".join(lines)


def _normalize_date(date_str: str) -> str:
    cleaned = date_str.strip().rstrip(".")
    parts = re.split(r'[.\-]', cleaned)
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        y, m, d = parts
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return date_str


def _add_days(date_str: str, days: int) -> str:
    cleaned = date_str.strip().rstrip(".")
    try:
        dt = datetime.strptime(cleaned, "%Y.%m.%d")
    except ValueError:
        return _normalize_date(date_str)
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")


# ============================================================
# 표 / 산식 라인 판별
# ============================================================
def is_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _MD_TABLE_ROW_RE.match(s) or _GRADE_ROW_RE.match(s):
        return True
    fields = re.split(r'\s{2,}|\t', s)
    return len(fields) >= 3 and any(re.search(r'\d', f) for f in fields)


def is_formula_line(line: str) -> bool:
    s = line.strip()
    tokens = len(_FORMULA_TOKEN_RE.findall(s))
    pct = len(re.findall(r'\d+(?:\.\d+)?\s*%', s))
    return (tokens + pct) >= 2


# ============================================================
# LawNode — 조문/별표 공용 아웃라인 트리
# ============================================================
@dataclass
class LawNode:
    level: str
    number: Optional[str] = None
    text: str = ""
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentGroup:
    parent_node: LawNode
    breadcrumb: list
    child_source_nodes: list


def render_full_text(node: LawNode) -> str:
    parts = [node.text] if node.text else []
    parts += [render_full_text(c) for c in node.children]
    return "\n".join(p for p in parts if p)


def breadcrumb_label(node: LawNode) -> str:
    if node.level in _LEVEL_SUFFIX and node.number:
        marker = node.number.strip()
        suffix = _LEVEL_SUFFIX[node.level]
        return marker if marker.endswith(suffix) else f"{marker}{suffix}"
    first_line = (node.text or "").split("\n", 1)[0].strip()
    return first_line


def _push(stack: list, depth: int, **kwargs) -> LawNode:
    while len(stack) > depth:
        stack.pop()
    while len(stack) < depth:
        filler = LawNode(level=f"_gap{len(stack)}")
        stack[-1].children.append(filler)
        stack.append(filler)
    node = LawNode(**kwargs)
    stack[-1].children.append(node)
    stack.append(node)
    return node


def _absorb(node: LawNode, line: str, is_table: bool, is_formula: bool) -> None:
    node.text = f"{node.text}\n{line}" if node.text else line
    if is_table:
        node.metadata["is_table_block"] = True
    if is_formula:
        node.metadata["is_formula_block"] = True


def build_outline_jomun(lines: list) -> LawNode:
    root = LawNode(level="article_body")
    stack = [root]
    for raw in lines:
        t, f = is_table_line(raw), is_formula_line(raw)
        if t or f:
            _absorb(stack[-1], raw, t, f)
            continue
        line = raw.strip()
        matched = False
        for depth, (level_name, pat) in enumerate(LEVEL_PATTERNS_JOMUN, start=1):
            m = pat.match(line)
            if m:
                _push(stack, depth, level=level_name, number=m.group(1), text=raw)
                matched = True
                break
        if not matched:
            _absorb(stack[-1], raw, False, False)
    return root


def dotted_depth(number: str) -> int:
    return number.count(".") + 1


def build_outline_byeolpyo(lines: list) -> LawNode:
    root = LawNode(level="byeolpyo_root")
    stack = [root]
    for raw in lines:
        t, f = is_table_line(raw), is_formula_line(raw)
        if t or f:
            _absorb(stack[-1], raw, t, f)
            continue
        line = raw.strip()
        m = _DOTTED_RE.match(line)
        if m:
            depth = dotted_depth(m.group(1))
            _push(stack, depth, level=f"L{depth}", number=m.group(1), text=raw)
            continue
        m = _PLAIN_ITEM_RE.match(line)
        if m:
            _push(stack, 1, level="byeolpyo_item", number=m.group(1), text=raw)
            continue
        m = _SUB_RE.match(line)
        if m:
            _push(stack, 2, level="byeolpyo_sub", number=m.group(1), text=raw)
            continue
        _absorb(stack[-1], raw, False, False)
    return root


# ============================================================
# 경계탐지 우선 분리
# ============================================================
def split_top_level(full_text: str) -> dict:
    buchil_hits = list(PAT_BUCHIL.finditer(full_text))
    buchil_start = buchil_hits[0].start() if buchil_hits else len(full_text)
    body_text = full_text[:buchil_start]
    buchil_text = full_text[buchil_start:]

    byeolpyo_hits = list(PAT_BYEOLPYO.finditer(body_text))
    byeolpyo_start = byeolpyo_hits[0].start() if byeolpyo_hits else len(body_text)
    jomun_text = body_text[:byeolpyo_start]
    byeolpyo_text = body_text[byeolpyo_start:]

    return {"jomun": jomun_text, "byeolpyo": byeolpyo_text, "buchil": buchil_text}


def _slice_by_hits(text: str, hits: list) -> list:
    spans = []
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        spans.append((h, text[h.start():end]))
    return spans


def split_jomun_articles(jomun_text: str):
    hits = list(PAT_JOMUN.finditer(jomun_text))
    if not hits:
        return jomun_text.strip(), []
    intro = jomun_text[:hits[0].start()].strip()
    return intro, _slice_by_hits(jomun_text, hits)


def split_byeolpyo_tables(byeolpyo_text: str) -> list:
    return _slice_by_hits(byeolpyo_text, list(PAT_BYEOLPYO.finditer(byeolpyo_text)))


def split_buchil_entries(buchil_text: str) -> list:
    return _slice_by_hits(buchil_text, list(PAT_BUCHIL.finditer(buchil_text)))


# ============================================================
# 조문 / 별표 루트 빌드 (헤더 분리 + 아웃라인 빌더 호출)
# ============================================================
def _build_article_root(m: re.Match, span_text: str):
    header_text = m.group(0)
    title = (m.group(3) or "").strip()
    num, sub = m.group(1), m.group(2)
    remainder = span_text[len(header_text):]
    first_newline = remainder.find("\n")
    if first_newline == -1:
        body_lines = [remainder.strip()] if remainder.strip() else []
    else:
        first_line_remainder = remainder[:first_newline].strip()
        rest_lines = remainder[first_newline + 1:].split("\n")
        body_lines = ([first_line_remainder] if first_line_remainder else []) + rest_lines

    root = build_outline_jomun(body_lines)
    root.text = f"{header_text}\n{root.text}" if root.text else header_text
    root.level = "article"
    root.number = header_text
    article_no = f"제{num}조" + (f"의{sub}" if sub else "")
    meta = {"article_no": article_no, "article_title": title, "doc_type": "조문"}
    return root, meta


def _build_byeolpyo_root(m: re.Match, span_text: str):
    header_line_end = span_text.find("\n")
    if header_line_end == -1:
        header_text, body_lines = span_text.strip(), []
    else:
        header_text = span_text[:header_line_end].strip()
        body_lines = span_text[header_line_end + 1:].split("\n")

    root = build_outline_byeolpyo(body_lines)
    root.text = f"{header_text}\n{root.text}" if root.text else header_text
    root.level = "byeolpyo_root"
    root.number = header_text
    table_no = f"별표{m.group(1)}"
    meta = {"table_no": table_no, "table_title": header_text, "doc_type": "별표"}
    return root, meta


# ============================================================
# 부칙 시행일 파싱
# ============================================================
def parse_buchil_dates(entry_text: str, published_date_raw: str) -> dict:
    published_date = _normalize_date(published_date_raw)
    sentences = re.split(r'(?<=[.다])\s*\n|(?<=시행한다\.)\s*', entry_text)
    primary, exceptions = None, []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        eff = None
        if PAT_EFFECTIVE_ON_PROMULGATION.search(sent):
            eff = published_date
        else:
            m = PAT_EFFECTIVE_AFTER_DAYS.search(sent)
            if m:
                eff = _add_days(published_date_raw, int(m.group(1)))
            else:
                m = PAT_EFFECTIVE_LITERAL.search(sent)
                if m:
                    eff = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        if not eff:
            continue
        if primary is None:
            primary = eff
        else:
            exceptions.append({"scope": sent, "effective_date": eff})
    return {
        "published_date": published_date,
        "effective_date": primary or published_date,
        "effective_date_exceptions": exceptions,
    }


# ============================================================
# Parent/Child 변환 — 토큰 기준 재귀 승격
# ============================================================
def to_parent_child(node: LawNode, breadcrumb: list, cfg: dict) -> list:
    full_text = render_full_text(node)
    protected = bool(node.metadata.get("is_table_block") or node.metadata.get("is_formula_block"))
    if not node.children or protected or count_tokens(full_text) <= cfg["max_parent_tokens"]:
        child_sources = node.children if node.children else [
            LawNode(level=f"{node.level}_leaf", text=node.text, metadata=dict(node.metadata))
        ]
        return [ParentGroup(node, list(breadcrumb), child_sources)]

    groups = []
    preface = node.text
    for child in node.children:
        sub_groups = to_parent_child(child, breadcrumb + [breadcrumb_label(child)], cfg)
        if preface and sub_groups:
            first = sub_groups[0].parent_node
            first.text = f"{preface}\n{first.text}" if first.text else preface
            preface = ""
        groups.extend(sub_groups)
    return groups


def materialize_parent_group(group: ParentGroup, law_title: str, doc_type: str, cfg: dict):
    parent_text = render_full_text(group.parent_node)
    parent_metadata = {
        "law_title": law_title,
        "doc_type": doc_type,
        "type": "parent",
        "breadcrumb": " > ".join(group.breadcrumb),
    }
    parent_metadata.update(group.parent_node.metadata)
    if group.parent_node.level == "clause" and group.parent_node.number:
        parent_metadata["clause_marker"] = group.parent_node.number

    parent_tn = TextNode(text=parent_text, metadata=parent_metadata)
    parent_tn.metadata["_cross_ref_surface_tokens"] = [m.group(0) for m in REF_PAT.finditer(parent_text)]

    children = []
    for src in group.child_source_nodes:
        text = render_full_text(src)
        if not text.strip():
            continue
        protected = bool(src.metadata.get("is_table_block") or src.metadata.get("is_formula_block"))
        if protected or count_tokens(text) <= cfg["max_child_tokens"]:
            pieces = [text]
        else:
            pieces = tiktoken_window_split(text, cfg["max_child_tokens"], cfg["child_overlap_tokens"])

        for idx, piece in enumerate(pieces):
            label = breadcrumb_label(src)
            if len(pieces) > 1:
                label = f"{label} ({idx + 1}/{len(pieces)})"
            breadcrumb_str = " > ".join(group.breadcrumb + [label])
            child_metadata = {
                "law_title": law_title,
                "doc_type": doc_type,
                "type": "child",
                "breadcrumb": breadcrumb_str,
            }
            child_metadata.update(src.metadata)
            child_metadata["_cross_ref_surface_tokens"] = [m.group(0) for m in REF_PAT.finditer(piece)]
            child_tn = TextNode(
                text=CHILD_TEMPLATE.format(law_title=law_title, breadcrumb=breadcrumb_str, content=piece),
                metadata=child_metadata,
            )
            children.append(IndexNode.from_text_node(child_tn, index_id=parent_tn.node_id))
    return parent_tn, children


# ============================================================
# Cross-reference 2-pass 해결
# ============================================================
def _resolve_token(tok: str, article_index: dict, article_paragraph_index: dict, byeolpyo_index: dict) -> list:
    tok = tok.strip()
    m = _TOK_BYEOLPYO_RE.match(tok)
    if m:
        return list(byeolpyo_index.get(f"별표{m.group(1)}", []))
    m = _TOK_ARTICLE_RE.match(tok)
    if m:
        article_no = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        clause = m.group(3)
        if clause:
            node_id = article_paragraph_index.get((article_no, clause))
            if node_id:
                return [node_id]
        return list(article_index.get(article_no, []))
    return []


def resolve_cross_refs(parent_nodes: list, article_index: dict, article_paragraph_index: dict, byeolpyo_index: dict) -> None:
    for node in parent_nodes:
        tokens = node.metadata.pop("_cross_ref_surface_tokens", [])
        resolved, unresolved = [], []
        for tok in tokens:
            ids = _resolve_token(tok, article_index, article_paragraph_index, byeolpyo_index)
            if ids:
                resolved.extend(ids)
            else:
                unresolved.append(tok)
        node.metadata["cross_refs"] = sorted(set(resolved) - {node.node_id})
        node.metadata["cross_refs_unresolved"] = sorted(set(unresolved))


# ============================================================
# 영역별 워커 — ProcessPoolExecutor로 보낼 수 있도록 모듈 최상위 함수로 두고
# 원시 자료형(str/dict)과 LawNode/ParentGroup 같은 순수 dataclass만 주고받는다.
# self.logger(파일 핸들을 쥔 객체)는 pickling이 안 되므로 워커 안에서는 절대
# 로깅하지 않고, 경고 메시지를 문자열로 모아 반환해 부모 프로세스가 로깅한다.
# ============================================================
def _process_region_worker(region_name: str, region_text: str, law_title: str, cfg: dict) -> dict:
    normalized = _normalize_text(region_text)
    warnings = []
    result = {"region": region_name, "warnings": warnings}

    if region_name == "jomun":
        intro, spans = split_jomun_articles(normalized)
        reconstructed_parts = [intro] if intro else []
        articles = []
        for m, span_text in spans:
            root, meta = _build_article_root(m, span_text)
            reconstructed_parts.append(render_full_text(root))
            groups = to_parent_child(root, [breadcrumb_label(root)], cfg)
            for g in groups:
                g.parent_node.metadata = {**meta, **g.parent_node.metadata}
            articles.append({"groups": groups})
        ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), normalized.strip()).ratio()
        ok = ratio >= cfg["reconstruction_min_ratio"]
        if not ok:
            warnings.append(f"[{law_title}] 조문 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")
        result.update({"intro_text": intro, "articles": articles, "ratio": ratio, "ok": ok})

    elif region_name == "byeolpyo":
        spans = split_byeolpyo_tables(normalized)
        reconstructed_parts = []
        tables = []
        for m, span_text in spans:
            root, meta = _build_byeolpyo_root(m, span_text)
            reconstructed_parts.append(render_full_text(root))
            groups = to_parent_child(root, [breadcrumb_label(root)], cfg)
            for g in groups:
                g.parent_node.metadata = {**meta, **g.parent_node.metadata}
            tables.append({"groups": groups})
        ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), normalized.strip()).ratio()
        ok = ratio >= cfg["reconstruction_min_ratio"]
        if not ok:
            warnings.append(f"[{law_title}] 별표 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")
        result.update({"tables": tables, "ratio": ratio, "ok": ok})

    else:  # buchil
        spans = split_buchil_entries(normalized)
        reconstructed_parts = []
        entries = []
        for idx, (m, span_text) in enumerate(spans):
            law_no, published_date_raw = m.group(1), m.group(2)
            text = span_text.strip()
            reconstructed_parts.append(text)
            dates = parse_buchil_dates(span_text, published_date_raw)
            entries.append({
                "text": text,
                "law_no": law_no,
                "is_latest": idx == len(spans) - 1,
                **dates,
            })
        ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), normalized.strip()).ratio()
        ok = ratio >= cfg["reconstruction_min_ratio"]
        if not ok:
            warnings.append(f"[{law_title}] 부칙 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")
        result.update({"entries": entries, "ratio": ratio, "ok": ok})

    return result


class LawChunker:
    """
    한국 법령/훈령류 문서(조문-항-호-목-세목 / 별표 / 부칙 구조)를 Parent-Child
    계층으로 청킹한다. 경계탐지(boundary-first) 분리 + 조문/별표 공용 아웃라인
    빌더 + 토큰 기준 크기 제어로 동작한다(v2).

    사용 흐름:
        chunker = LawChunker()
        stats = chunker.detect_structure(llama_docs)
        if stats["has_article"] or stats["has_byeolpyo"]:
            all_nodes, node_dict = chunker.parse_to_hierarchical_nodes(llama_docs)
        else:
            fallback_chunker = FallbackChunker()
            all_nodes, node_dict = fallback_chunker.create_fallback_nodes(llama_docs)
    """

    def __init__(self, cfg: dict = None):
        self.logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    # ============================================================
    # 1. 문서 구조 탐지 — 표본 없이 전문(全文) 1-pass 스캔
    # ============================================================
    def detect_structure(self, llama_docs) -> dict:
        normalized = _normalize_text("\n".join(d.text for d in llama_docs))
        article_matches = len(list(PAT_JOMUN.finditer(normalized)))
        table_matches = len(list(PAT_BYEOLPYO.finditer(normalized)))
        addendum_matches = len(list(PAT_BUCHIL.finditer(normalized)))

        self.logger.info(
            f"[LawChunker] 구조 탐지 결과(전문 스캔) - 조문:{article_matches} "
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

    def _extract_law_title(self, llama_docs):
        file_name, law_title = None, "답변자료"
        if llama_docs:
            first_meta = getattr(llama_docs[0], "metadata", {}) or {}
            if isinstance(first_meta, dict):
                file_name = first_meta.get("file_name")
                if file_name:
                    law_title = os.path.splitext(file_name)[0]
        return file_name, law_title

    # ============================================================
    # 2. 메인 파서
    #
    # llama_docs를 하나로 합쳐서 처리하면 여러 파일(zip/디렉토리 입력)이 섞였을 때
    # A 파일의 부칙 뒤에 B 파일의 조문/별표가 붙어버려 경계탐지가 통째로 깨진다
    # (split_top_level은 "첫 부칙 등장 지점 이후 전부"를 부칙 영역으로 간주하므로).
    # 이를 막기 위해 doc(파일) 단위로 독립적으로 split_top_level ~ materialize까지
    # 끝내고, cross-ref 해결과 peer_tables만 전체 문서를 모은 뒤 한 번에 계산한다.
    # ============================================================
    def parse_to_hierarchical_nodes(self, llama_docs):
        all_nodes, node_dict = [], {}
        article_index = defaultdict(list)
        article_paragraph_index = {}
        byeolpyo_index = defaultdict(list)
        parent_nodes = []

        for doc in llama_docs:
            file_name, law_title = self._extract_law_title([doc])
            normalized = _normalize_text(doc.text)
            regions = split_top_level(normalized)

            tasks = [(name, text) for name, text in regions.items() if text.strip()]
            if not tasks:
                continue

            total_chars = sum(len(t) for _, t in tasks)
            use_mp = (
                total_chars >= self.cfg["parallel_char_threshold"]
                and len(tasks) > 1
                and os.environ.get("LAWCHUNKER_DISABLE_MP") != "1"
            )

            try:
                if use_mp:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=min(3, len(tasks))) as ex:
                        futures = [ex.submit(_process_region_worker, name, text, law_title, self.cfg) for name, text in tasks]
                        results = [f.result() for f in futures]
                else:
                    results = [_process_region_worker(name, text, law_title, self.cfg) for name, text in tasks]
            except Exception as e:
                self.logger.warning(f"[LawChunker] ({file_name or law_title}) 병렬 처리 실패, 순차 처리로 재시도합니다: {e}")
                results = [_process_region_worker(name, text, law_title, self.cfg) for name, text in tasks]

            for r in results:
                for w in r.get("warnings", []):
                    self.logger.warning(f"[LawChunker] {w}")

            results_by_region = {r["region"]: r for r in results}
            doc_nodes = []

            # --- 조문 (intro 포함) ---
            jomun_result = results_by_region.get("jomun")
            if jomun_result:
                ratio, ok = jomun_result.get("ratio", 1.0), jomun_result.get("ok", True)
                intro_text = (jomun_result.get("intro_text") or "").strip()
                if intro_text:
                    pieces = [intro_text] if count_tokens(intro_text) <= self.cfg["max_parent_tokens"] \
                        else tiktoken_window_split(intro_text, self.cfg["max_parent_tokens"], self.cfg["child_overlap_tokens"])
                    for piece in pieces:
                        p_node = TextNode(text=piece, metadata={
                            "law_title": law_title, "doc_type": "intro", "type": "parent",
                            "reconstruction_ratio": ratio, "reconstruction_ok": ok,
                        })
                        doc_nodes.append(p_node)
                        node_dict[p_node.node_id] = p_node

                for article_payload in jomun_result.get("articles", []):
                    for g in article_payload["groups"]:
                        parent_tn, child_nodes = materialize_parent_group(g, law_title, "조문", self.cfg)
                        parent_tn.metadata["reconstruction_ratio"] = ratio
                        parent_tn.metadata["reconstruction_ok"] = ok
                        for c in child_nodes:
                            c.metadata["reconstruction_ratio"] = ratio
                            c.metadata["reconstruction_ok"] = ok
                        node_dict[parent_tn.node_id] = parent_tn
                        doc_nodes.append(parent_tn)
                        doc_nodes.extend(child_nodes)
                        parent_nodes.append(parent_tn)

                        article_no = parent_tn.metadata.get("article_no")
                        if article_no:
                            article_index[article_no].append(parent_tn.node_id)
                            clause_marker = parent_tn.metadata.get("clause_marker")
                            if clause_marker:
                                article_paragraph_index[(article_no, clause_marker)] = parent_tn.node_id

            # --- 별표 ---
            byeolpyo_result = results_by_region.get("byeolpyo")
            if byeolpyo_result:
                ratio, ok = byeolpyo_result.get("ratio", 1.0), byeolpyo_result.get("ok", True)
                for table_payload in byeolpyo_result.get("tables", []):
                    for g in table_payload["groups"]:
                        parent_tn, child_nodes = materialize_parent_group(g, law_title, "별표", self.cfg)
                        parent_tn.metadata["reconstruction_ratio"] = ratio
                        parent_tn.metadata["reconstruction_ok"] = ok
                        for c in child_nodes:
                            c.metadata["reconstruction_ratio"] = ratio
                            c.metadata["reconstruction_ok"] = ok
                        node_dict[parent_tn.node_id] = parent_tn
                        doc_nodes.append(parent_tn)
                        doc_nodes.extend(child_nodes)
                        parent_nodes.append(parent_tn)

                        table_no = parent_tn.metadata.get("table_no")
                        if table_no:
                            byeolpyo_index[table_no].append(parent_tn.node_id)

            # --- 부칙 ---
            buchil_result = results_by_region.get("buchil")
            if buchil_result:
                ratio, ok = buchil_result.get("ratio", 1.0), buchil_result.get("ok", True)
                for entry in buchil_result.get("entries", []):
                    node = TextNode(text=entry["text"], metadata={
                        "law_title": law_title, "doc_type": "부칙", "type": "parent",
                        "law_no": entry["law_no"],
                        "published_date": entry["published_date"],
                        "effective_date": entry["effective_date"],
                        "effective_date_exceptions": entry["effective_date_exceptions"],
                        "is_latest": entry["is_latest"],
                        "reconstruction_ratio": ratio, "reconstruction_ok": ok,
                    })
                    node_dict[node.node_id] = node
                    doc_nodes.append(node)

            # --- 파일별 top_k 검색을 위한 소스 파일명 태깅 (doc마다 자기 파일명으로) ---
            if file_name:
                for n in doc_nodes:
                    n.metadata["file_name"] = file_name

            all_nodes.extend(doc_nodes)

        # --- peer_tables: 전체 문서를 모은 별표 인덱스 기준으로 한 번에 계산 ---
        all_table_nos = list(byeolpyo_index.keys())
        for table_no in all_table_nos:
            peers = [nid for other in all_table_nos if other != table_no for nid in byeolpyo_index[other]]
            for nid in byeolpyo_index[table_no]:
                node_dict[nid].metadata["peer_tables"] = peers

        # --- cross-ref 2-pass 해결 (전체 문서를 모은 인덱스 기준) ---
        resolve_cross_refs(parent_nodes, article_index, article_paragraph_index, byeolpyo_index)

        self.logger.info(f"[LawChunker] 계층 노드 생성 완료 - 총 {len(all_nodes)}개 (parent+child)")
        return all_nodes, node_dict
