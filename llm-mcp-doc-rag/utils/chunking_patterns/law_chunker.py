"""
LawChunker - LangGraph 기반 파이프라인 (v3, 상세 주석 버전)

■ 이 파일이 하는 일 (한 줄 요약)
   한국 법령/훈령 문서(예: "제1조(목적) ...", "별표1", "부칙 <제123호, 2024.1.1>" 같은
   구조를 가진 텍스트)를 읽어서, 검색(RAG)에 쓰기 좋은 "부모-자식" 형태의 조각(chunk)
   들로 잘게 나눠 주는 도구다.

■ 왜 "부모-자식(Parent-Child)"인가?
   - "자식(child)" 노드는 아주 잘게 쪼갠 짧은 텍스트라서 임베딩 검색 정확도가 높다.
   - 하지만 검색 결과로 짧은 조각만 보여주면 문맥이 부족하다.
   - 그래서 검색은 "자식"으로 하되, 실제로 사용자에게 보여줄 때는 그 자식이 속한
     "부모(더 큰 문맥 단위, 예: 조문 전체)"를 함께 보여준다. 이 패턴을
     LlamaIndex에서는 IndexNode(자식) + TextNode(부모) 조합으로 구현한다.

■ 이 파일의 구성 순서
   1) 정규식(패턴) 정의 - "이 텍스트가 제몇조인지, 별표인지, 부칙인지" 등을 찾는 규칙
   2) 자잘한 유틸 함수들 - 토큰 개수 세기, 텍스트 정규화, 표/산식 라인 판별 등
   3) LawNode / ParentGroup - 문서를 트리 구조로 표현하기 위한 자료구조
   4) 문서를 조문/별표/부칙 영역으로 잘라내는 함수들
   5) 조문/별표 각각을 "항-호-목-세목" 트리로 만드는 함수들
   6) 트리를 실제 부모/자식 TextNode로 변환하는 함수들
   7) "제3조를 참고하라" 같은 상호참조(cross-reference)를 실제 노드 id로 연결하는 함수들
   8) 위 모든 걸 하나의 흐름으로 실행하는 LangGraph 파이프라인 (노드 + 조건부 분기)

■ LangGraph를 처음 보는 분을 위한 아주 간단한 설명
   - LangGraph는 "상태(state)를 들고 이 노드에서 저 노드로 이동하며 값을 갱신하는
     순서도(flowchart)"를 코드로 표현하게 해주는 라이브러리다.
   - "노드(node)"는 그냥 "state(딕셔너리)를 입력받아 일부를 갱신해서 돌려주는 함수"다.
   - "엣지(edge)"는 노드 A가 끝나면 노드 B로 간다는 화살표다.
   - "조건부 엣지(conditional edge)"는 화살표가 하나가 아니라 "state 값에 따라 어느
     쪽으로 갈지 결정"하는 분기점이다. (if/elif 를 그래프로 그린 것이라고 생각하면 된다)
   - "Send"는 "지금 상태를 N개로 복제해서 같은 노드를 N번 병렬로 실행시켜라"는 특수
     명령이다. 이 파일에서는 "문서가 여러 개면 문서 개수만큼 process_document를
     병렬로 돌려라"는 용도로 쓰인다.

그래프 흐름 그림
============
START
  └─ detect_structure                구조 탐지(조문/별표/부칙 존재 여부, 전문 1-pass 스캔)
       │
       ├─(조문도 별표도 없음)──► fallback_chunk ──► END
       │
       └─(구조 있음)──► prepare_documents
                            │  (조건부 엣지 = Send 팬아웃: 문서 1건당 1회 분기)
                            ▼
                        process_document  ×N   (문서 단위 독립 처리, 병렬 실행 가능)
                            │   내부에서 다시 3영역(조문/별표/부칙)으로 조건부 분기
                            │   → 텍스트가 없는 영역은 아예 실행하지 않음
                            ▼
                        merge_documents         (Send로 흩어진 문서별 결과를 합류)
                            ▼
                        compute_peer_tables      (별표 간 peer_tables 계산)
                            ▼
                        resolve_cross_refs       (조문/별표 상호참조 2-pass 해결)
                            ▼
                        finalize ──► END

원본과의 차이점
================
- 원본은 "영역(조문/별표/부칙) 단위" ProcessPoolExecutor 병렬 처리를 문서 내부에서
  수행했다. 본 리팩토링에서는 병렬 단위를 "문서(doc) 단위"로 올려 LangGraph의
  Send 팬아웃이 곧 병렬 처리 단위가 되도록 했다. 문서 수가 많은 배치 처리에서는
  이 편이 그래프 상에서 병렬 처리 진행 상황을 추적하기 쉽다.
- LawChunker.detect_structure() / parse_to_hierarchical_nodes() 는 기존과 동일한
  시그니처로 유지되며, 내부적으로 컴파일된 그래프를 실행하는 얇은 래퍼로 바뀌었다.
"""

import re
import os
import difflib
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, TypedDict, Annotated

from llama_index.core.schema import IndexNode, TextNode
from utils import logger_util

from langgraph.graph import StateGraph, START, END
try:
    from langgraph.types import Send
except ImportError:  # 구버전 langgraph 호환
    from langgraph.constants import Send


# 이 파이프라인 전용 로거. 파일 어디서든 logger.info(...) / logger.warning(...) 로 로그를 남긴다.
logger = logger_util.get_logger("./log", "llm-mcp-doc-rag")


# ============================================================
# 1) 정규식(패턴) 정의
#
# 여기 있는 정규식들은 "이 줄이 조문 시작인지 / 별표 시작인지 / 부칙 시작인지"를
# 판별하는 데 쓰인다. re.M(MULTILINE) 옵션 때문에 '^'가 "문자열 전체의 시작"이
# 아니라 "각 줄의 시작"을 의미하게 되어, 여러 줄짜리 긴 텍스트에서도
# "줄 맨 앞에 이 패턴이 나오는 모든 위치"를 한 번에 찾을 수 있다.
#
# ※ 설계 철학: "경계탐지(boundary-first)" 방식
#   전통적인 방법은 텍스트를 한 줄씩 순회하면서 "지금 조문 안인지 부칙 안인지"를
#   상태(state)로 들고 다니는 상태머신(state machine) 방식이다. 하지만 이 방식은
#   부칙 안에 "제1조(시행일)"처럼 조문과 똑같이 생긴 문장이 나오면 헷갈리기 쉽다.
#   그래서 이 코드는 반대로 접근한다:
#     1. 먼저 전체 텍스트에서 "부칙이 시작되는 위치"를 찾아 그 뒤는 통째로
#        "부칙 영역"으로 잘라낸다.
#     2. 남은 앞부분에서 "별표가 시작되는 위치"를 찾아 그 뒤를 "별표 영역"으로 자른다.
#     3. 그러고 나서야 "조문 영역"에만 조문 패턴(PAT_JOMUN)을 적용한다.
#   이렇게 하면 부칙 영역 안에 있는 "제1조(시행일)" 같은 문구는 애초에 조문 패턴이
#   적용되는 범위 밖에 있으므로 조문으로 잘못 인식될 수가 없다(구조적으로 안전).
# ============================================================

# 조문 시작 패턴: "제1조(목적)", "제2조의2(정의)" 같은 줄을 찾는다.
#   - 제(\d+)조            → "제" + 숫자 + "조" (예: 제1조).  그룹1 = 조 번호("1")
#   - (?:의(\d+))?         → "의2"처럼 가지번호가 있을 수도(선택) 있음. 그룹2 = 가지번호("2")
#   - \(([^\n)]*)\)        → 소괄호 안의 제목. 그룹3 = 조문 제목(예: "목적")
#   예) "제3조의2(적용범위)" → 그룹1="3", 그룹2="2", 그룹3="적용범위"
PAT_JOMUN = re.compile(r'^제(\d+)조(?:의(\d+))?\(([^\n)]*)\)', re.M)

# 별표 시작 패턴: "별표1", "[별표 2-1]" 같은 줄을 찾는다.
#   - \[? ... \]?          → 대괄호는 있어도 없어도 매칭(선택)
#   - 별표\s*([\d\-]+)     → "별표" 뒤에 숫자나 하이픈(2-1 같은 가지번호). 그룹1 = 별표 번호
PAT_BYEOLPYO = re.compile(r'^\[?별표\s*([\d\-]+)\]?', re.M)

# 부칙 시작 패턴: "부칙 <제3621호, 2024.1.1>" 같은 줄을 찾는다.
#   - 부\s*칙              → "부칙" (사이에 공백이 끼어도 허용)
#   - [<〈] ... [>〉]      → 꺾쇠괄호(반각 < > 또는 전각 〈 〉 둘 다 허용)
#   - 제?\s*(\S+?)\s*호    → "제3621호"에서 숫자/문자 부분만. 그룹1 = 법령(개정)번호
#   - ([\d.]+)             → "2024.1.1" 같은 날짜. 그룹2 = 공포일자(원문 그대로의 문자열)
PAT_BUCHIL = re.compile(r'^부\s*칙\s*[<〈]\s*제?\s*(\S+?)\s*호\s*,?\s*([\d.]+)\s*[>〉]', re.M)

# --- 아래는 "본문 중간에 등장하는 언급(mention)"을 찾는 패턴들이다 ---
# 위의 PAT_JOMUN 등이 "조문/별표/부칙의 시작 지점"을 찾는 것과 달리,
# 이 REF_PAT 등은 본문 문장 속에서 "별표3을 참고하라", "제5조제2항에 따라" 처럼
# 다른 조문/별표를 "언급"하는 부분을 찾아내는 데 쓰인다(상호참조 해결용).
# 그래서 PAT_JOMUN과 달리 제목(괄호) 없이도 매칭되도록 좀 더 느슨하게 만들어졌다.
REF_PAT = re.compile(r'(별표\s*[\d\-]+|별지\s*제?\s*\d+\s*호?|제\d+조(?:의\d+)?(?:\s*[①-⑳])?(?:제\d+호)?)')

# REF_PAT으로 찾아낸 문자열("제5조제2항" 등) 하나를 다시 세분해서
# "몇 조인지 / 몇 항인지"를 뽑아내기 위한 보조 패턴.
#   - ([①-⑳])? → 항을 나타내는 동그라미 숫자(①②③...). 그룹3 = 항 기호
_TOK_ARTICLE_RE = re.compile(r'^제(\d+)조(?:의(\d+))?\s*([①-⑳])?')
# REF_PAT으로 찾아낸 문자열이 "별표N" 형태일 때 별표 번호만 뽑아내는 보조 패턴.
_TOK_BYEOLPYO_RE = re.compile(r'^별표\s*([\d\-]+)')

# --- 부칙(附則) 안에서 "시행일"을 뽑아내기 위한 패턴들 ---
# 부칙에는 보통 "이 법은 2024년 1월 1일부터 시행한다" 같은 문장이 있다.
# 시행일을 표현하는 방식이 여러 가지라서(날짜 직접 명시 / 공포한 날 / 공포 후 며칠) 패턴을 3개로 나눴다.

# 패턴1: "2024.1.1부터 시행" / "2024년 1월 1일부터 시행" 처럼 날짜가 직접 적힌 경우.
#   그룹1=연도, 그룹2=월, 그룹3=일
PAT_EFFECTIVE_LITERAL = re.compile(r'(\d{4})[.\s년]\s*(\d{1,2})[.\s월]\s*(\d{1,2})일?\s*부터\s*시행')
# 패턴2: "공포한 날부터 시행" → 이 경우 시행일 = 공포일(published_date)과 같다.
PAT_EFFECTIVE_ON_PROMULGATION = re.compile(r'공포한\s*날부터\s*시행')
# 패턴3: "공포 후 30일이 경과한 날부터 시행" → 공포일 + N일 로 계산해야 한다. 그룹1 = N(일수)
PAT_EFFECTIVE_AFTER_DAYS = re.compile(r'공포\s*후\s*(\d+)일이?\s*경과한\s*날부터\s*시행')

# --- 마크다운(.md) 변환본에 남아있는 장식 문자를 제거하기 위한 패턴들 ---
# PDF를 마크다운으로 변환한 문서를 입력으로 받는 경우, "## 제목", "- 항목" 처럼
# 마크다운 문법 기호가 붙어 있을 수 있다. 이런 기호가 붙어 있으면 위의 PAT_JOMUN 같은
# "줄 맨 앞부터 매칭" 패턴들이 실패하므로, 파싱 전에 미리 제거해준다.
_MD_HEADING_RE = re.compile(r'^\s*#{1,6}\s*')          # 줄 맨 앞의 "#", "##" ... "######" 제거
_MD_BULLET_RE = re.compile(r'^\s*[-*•]\s+')            # 줄 맨 앞의 "-", "*", "•" 불릿 기호 제거
# "**==>그림 설명<==**" 처럼 이미지 자리에 들어가는 플레이스홀더 줄은 통째로 제거 대상.
_MD_PICTURE_PLACEHOLDER_RE = re.compile(r'^\s*\*\*==>.*<==\*\*\s*$')

# --- "이 줄은 표/산식이라서 조문 하위구조로 쪼개면 안 된다"를 판별하기 위한 패턴들 ---
_MD_TABLE_ROW_RE = re.compile(r'^\|.*\|$')   # 마크다운 표 행: "| 항목 | 값 |" 형태
# "가. 50%미만 22.0" 처럼 등급표의 한 행인데, 얼핏 보면 "가." 로 시작해서
# LEVEL_PATTERNS_JOMUN의 "sub"(목) 패턴과 헷갈릴 수 있다. 그래서 표/등급행 판별을
# "목" 패턴 매칭보다 먼저 수행해서, 이런 줄이 조문 하위목차로 오분류되지 않게 막는다.
_GRADE_ROW_RE = re.compile(r'^[A-Za-z가-힣]\s*[.\)]?\s*.{0,20}\d+(?:\.\d+)?\s*(?:점|%)')
# 산식(수식) 줄 판별용 토큰: ×÷±≤≥∑√ 같은 수학 기호, 또는 "3/4"처럼 숫자 사이의 / * 기호.
_FORMULA_TOKEN_RE = re.compile(r'[×÷±≤≥∑√]|(?<=\d)\s*[/*]\s*(?=\d)')

# --- 조문 내부 계층(항 → 호 → 목 → 세목) 판별 패턴들 ---
# 한국 법령 조문은 보통 아래처럼 4단계 깊이로 중첩된다.
#   ① (항, clause)  →  1. (호, item)  →  가. (목, sub)  →  1) (세목, subsub)
# 리스트의 순서가 곧 "깊이(depth)" 순서다: 앞에 있을수록 상위 계층.
LEVEL_PATTERNS_JOMUN = [
    ("clause", re.compile(r'^([①-⑳])')),                              # 항: ①②③...
    ("item", re.compile(r'^(\d+)\.\s')),                               # 호: "1. ", "2. "
    ("sub", re.compile(r'^([가나다라마바사아자차카타파하])\.\s')),      # 목: "가. ", "나. "
    ("subsub", re.compile(r'^(\d+)\)\s')),                             # 세목: "1) ", "2) "
]
# 각 계층 이름(level)을 한글 명칭(항/호/목/세목)으로 바꿀 때 쓰는 매핑.
_LEVEL_SUFFIX = {"clause": "항", "item": "호", "sub": "목", "subsub": "세목"}

# --- 별표(표) 내부 계층을 판별하기 위한 패턴들 ---
# 별표는 조문과 달리 "4.1", "4.1.1" 같은 점(.)으로 구분된 번호 체계를 쓰는 경우가 많다.
_DOTTED_RE = re.compile(r'^(\d+(?:\.\d+)+)\.?\s')   # "4.1 ", "4.1.1 " 처럼 점이 1개 이상 있어야 매칭
_PLAIN_ITEM_RE = re.compile(r'^(\d+)\.\s')          # "1. 수행능력평가(30점)" 처럼 점이 없는 단순 번호
_SUB_RE = LEVEL_PATTERNS_JOMUN[2][1]                # "가.나.다." 패턴은 조문의 "목" 패턴을 그대로 재사용


# 파이프라인 전체에서 사용할 기본 설정값들. LawChunker(cfg={...}) 생성 시 이 값들을
# 덮어쓸 수 있다(뒤에서 {**DEFAULT_CONFIG, **cfg} 형태로 병합).
DEFAULT_CONFIG = {
    "max_parent_tokens": 300,        # "부모" 청크 하나가 가질 수 있는 최대 토큰 수
    "max_child_tokens": 100,         # "자식" 청크 하나가 가질 수 있는 최대 토큰 수
    "child_overlap_tokens": 15,      # 자식 청크를 강제로 더 쪼갤 때, 앞뒤 조각이 겹치는 토큰 수(문맥 유지용)
    "parallel_char_threshold": 20000,  # (레거시 설정값. 현재 그래프 버전에서는 문서 단위 Send 병렬만 사용)
    "reconstruction_min_ratio": 0.995,  # "쪼갠 조각들을 다시 합쳤을 때 원본과 얼마나 똑같아야 하는가" 최소 기준(0~1)
}

# 자식(child) 노드의 실제 텍스트를 만들 때 쓰는 템플릿.
# 임베딩 모델이 "이 문장이 어느 법령의 어느 조항에서 왔는지" 맥락을 알 수 있도록
# breadcrumb(예: "제5조(적용범위) > ②항 > 2호")를 텍스트 안에 함께 넣어준다.
CHILD_TEMPLATE = "{law_title} > {breadcrumb}\n내용: {content}"


# ============================================================
# 2) 토큰 카운터
#
# "토큰(token)"이란 LLM/임베딩 모델이 텍스트를 처리하는 최소 단위다(글자 수와는 다름).
# 이 모듈에서는 실제 임베딩 모델의 토크나이저 대신, OpenAI 계열에서 널리 쓰이는
# tiktoken의 cl100k_base 인코딩을 "모델에 상관없이 쓸 수 있는 대략적인 근사치"로 사용한다.
# ============================================================

# 토크나이저 객체는 생성 비용이 있으므로, 전역 변수에 한 번만 만들어두고 재사용한다
# (파이썬의 "지연 초기화(lazy initialization)" 패턴 - 실제로 필요해지는 시점에 생성).
_TOKENIZER = None


def _get_tokenizer():
    """cl100k_base 토크나이저를 최초 1회만 로드하고, 이후에는 캐시된 것을 재사용한다."""
    global _TOKENIZER
    if _TOKENIZER is None:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """주어진 텍스트가 토큰 몇 개인지 센다.

    disallowed_special=() 로 특수토큰 검사를 꺼두는 이유:
    부칙 헤더에 나오는 "<제3621호, ...>" 같은 꺾쇠(<, >) 문자를 tiktoken이
    "이건 GPT의 특수 제어토큰이다"라고 착각해서 예외(exception)를 던지는 경우가 있다.
    이를 방지하기 위해 특수토큰 판정을 비활성화한다.
    """
    if not text:
        return 0
    return len(_get_tokenizer().encode(text, disallowed_special=()))


def tiktoken_window_split(text: str, max_tokens: int, overlap_tokens: int) -> list:
    """토큰 개수 기준 "슬라이딩 윈도우"로 긴 텍스트를 강제로 여러 조각으로 나눈다.

    동작 방식(비유): 창문(window) 하나가 텍스트를 max_tokens 만큼씩 훑고 지나가는데,
    창문이 완전히 겹치지 않고 overlap_tokens 만큼씩만 겹치면서 앞으로 이동한다.
    이렇게 겹치는 구간을 두는 이유는, 조각의 경계에서 문맥이 뚝 끊기지 않게 하기 위해서다.

    예) max_tokens=100, overlap_tokens=15 라면
        1번 조각: 토큰 [0:100]
        2번 조각: 토큰 [85:185]   (앞 조각과 15개 겹침)
        3번 조각: 토큰 [170:270]  ...
    """
    enc = _get_tokenizer()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        # 애초에 짧아서 나눌 필요가 없으면 그대로 반환.
        return [text]
    step = max(max_tokens - overlap_tokens, 1)  # 창문이 한 번에 전진하는 토큰 수 (0 이하로 내려가지 않게 방어)
    pieces, start = [], 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        # cl100k_base는 "바이트 레벨 BPE"라서, 토큰 경계 어디서 잘라 다시 텍스트로
        # decode 해도 깨진(잘못된) UTF-8 문자가 나오지 않는다는 성질을 이용한다.
        pieces.append(enc.decode(ids[start:end]))
        if end == len(ids):
            break  # 마지막 조각까지 다 만들었으면 종료
        start += step
    return pieces


# ============================================================
# 3) 입력 정규화 (PDF 원문 / 마크다운 변환본을 같은 모양으로 맞추기)
# ============================================================
def _normalize_text(full_text: str) -> str:
    """줄 단위로 훑으면서 마크다운 장식(헤딩 #, 불릿 -, 이미지 플레이스홀더)을 제거한다.

    이렇게 정규화를 미리 해두면, 뒤에서 사용하는 PAT_JOMUN 같은 "줄 맨 앞부터 매칭"
    정규식들이 "## 제1조(목적)"처럼 앞에 # 이 붙어있어도 실패하지 않고 잘 매칭된다.
    """
    lines = []
    for line in full_text.split("\n"):
        if _MD_PICTURE_PLACEHOLDER_RE.match(line):
            continue  # 이미지 플레이스홀더 줄은 아예 버린다(다음 줄로 넘어감)
        line = _MD_HEADING_RE.sub('', line)   # 줄 맨 앞의 #, ##, ... 제거
        line = _MD_BULLET_RE.sub('', line)    # 줄 맨 앞의 -, *, • 제거
        lines.append(line)
    return "\n".join(lines)


def _normalize_date(date_str: str) -> str:
    """"2024.1.1" / "2024-01-01" 같은 다양한 표기를 "2024-01-01" 형태로 통일한다."""
    cleaned = date_str.strip().rstrip(".")            # 앞뒤 공백, 끝에 붙은 마침표 제거
    parts = re.split(r'[.\-]', cleaned)                # "." 또는 "-" 기준으로 연/월/일 분리
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        y, m, d = parts
        # 자리수를 4자리(연도)/2자리(월,일)로 0채움(zero-padding)해서 통일된 포맷을 만든다.
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return date_str  # 형식이 예상과 다르면 원본을 그대로 반환(안전한 폴백)


def _add_days(date_str: str, days: int) -> str:
    """공포일(date_str)에 days일을 더한 날짜를 "YYYY-MM-DD" 문자열로 계산한다.

    "공포 후 30일이 경과한 날부터 시행" 같은 문장에서 실제 시행일을 계산할 때 쓰인다.
    """
    cleaned = date_str.strip().rstrip(".")
    try:
        dt = datetime.strptime(cleaned, "%Y.%m.%d")   # "2024.1.1" 형식으로 파싱 시도
    except ValueError:
        # 날짜 형식이 다르면 날짜 계산 자체가 불가능하므로, 최소한 표기라도 통일해서 반환.
        return _normalize_date(date_str)
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")


# ============================================================
# 4) 표 / 산식 라인 판별
#
# 아래 두 함수는 "이 한 줄이 일반 문장이 아니라 표의 한 행이거나 수식이다"를 판별한다.
# 표/수식으로 판정된 줄은 조문 하위구조(항-호-목-세목)로 쪼개지 않고, "보호 구역"으로
# 표시해서(is_table_block / is_formula_block) 원문 그대로 통째로 유지한다.
# 표를 억지로 항/호/목으로 쪼개면 표의 의미(행과 열의 관계)가 깨지기 때문이다.
# ============================================================
def is_table_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _MD_TABLE_ROW_RE.match(s) or _GRADE_ROW_RE.match(s):
        return True
    # 마크다운 표 기호(|)가 없어도, 탭이나 공백 2칸 이상으로 나뉜 "필드"가 3개 이상이고
    # 그중 숫자가 포함된 필드가 하나라도 있으면 "표처럼 생긴 줄"로 간주한다.
    # (예: "구분   최소값   최대값   비고" 처럼 공백으로 정렬된 표)
    fields = re.split(r'\s{2,}|\t', s)
    return len(fields) >= 3 and any(re.search(r'\d', f) for f in fields)


def is_formula_line(line: str) -> bool:
    s = line.strip()
    # 수식 기호(×÷±≤≥∑√, 숫자 사이의 /나 *) 개수 + "50%"처럼 퍼센트 표기 개수를 합쳐서
    # 2개 이상이면 "이 줄은 산식(계산식)이다"라고 판단한다. (기호 1개만으로는 오탐이
    # 많아서 "2개 이상"이라는 문턱값을 둔 것)
    tokens = len(_FORMULA_TOKEN_RE.findall(s))
    pct = len(re.findall(r'\d+(?:\.\d+)?\s*%', s))
    return (tokens + pct) >= 2


# ============================================================
# 5) LawNode — 조문/별표 공용 아웃라인 트리
#
# LawNode는 "법령 문서의 계층 구조(트리)"를 표현하기 위한 범용 자료구조다.
# 예를 들어 아래와 같은 텍스트가 있다면:
#
#   제5조(적용범위)
#   ① 이 규정은 다음 각 호에 적용한다.
#     1. 정규직 직원
#     2. 계약직 직원
#       가. 6개월 이상 근무자
#
# 이걸 LawNode 트리로 표현하면:
#   article(제5조) ── clause(①) ── item(1. 정규직 직원)
#                                └─ item(2. 계약직 직원) ── sub(가. 6개월 이상 근무자)
#
# 즉 "level"은 이 노드가 무슨 계층인지(조/항/호/목/세목 등), "number"는 그 계층의
# 번호(①, 1, 가 등), "text"는 그 노드 자신에게 직접 속한 줄들의 원문,
# "children"은 그 아래로 파고드는 하위 계층 노드 리스트다.
# ============================================================
@dataclass
class LawNode:
    level: str                                  # 이 노드의 계층 종류 (예: "article", "clause", "item", "sub", "subsub")
    number: Optional[str] = None                # 이 계층의 번호/기호 (예: "①", "1", "가")
    text: str = ""                               # 이 노드에 직접 속한 원문 텍스트(여러 줄이면 개행으로 이어붙임)
    children: list = field(default_factory=list)  # 하위 계층 LawNode 리스트
    metadata: dict = field(default_factory=dict)   # 표/산식 여부 등 부가 정보를 담는 자유 형식 딕셔너리


@dataclass
class ParentGroup:
    """"부모로 만들 노드 1개 + 그 부모의 위치를 설명하는 breadcrumb(경로) +
    그 부모 밑에서 자식(child) 청크로 쪼갤 원본 노드들의 목록"을 한 데 묶은 자료구조.

    to_parent_child() 함수가 LawNode 트리를 순회하면서 "여기서 부모를 하나 끊자"고
    판단한 지점마다 ParentGroup을 하나씩 만들어낸다.
    """
    parent_node: LawNode      # 부모로 취급할 LawNode (이 노드부터 그 아래 전체가 부모의 내용이 됨)
    breadcrumb: list          # 최상위부터 이 부모까지의 경로 (예: ["제5조(적용범위)", "①항"])
    child_source_nodes: list  # 이 부모 밑에서 각각 별도의 자식 청크로 만들 LawNode들


def render_full_text(node: LawNode) -> str:
    """이 노드와 그 모든 하위 노드(children)의 text를 순서대로 이어붙여
    "이 노드가 커버하는 원문 전체"를 재구성한다. (재귀 함수)
    """
    parts = [node.text] if node.text else []
    parts += [render_full_text(c) for c in node.children]  # 자식들도 재귀적으로 이어붙임
    return "\n".join(p for p in parts if p)


def breadcrumb_label(node: LawNode) -> str:
    """이 노드를 breadcrumb(경로) 표시용 짧은 라벨로 바꾼다.

    예) level="item", number="2"  →  "2호"  (숫자 뒤에 "호"라는 한글 접미사를 붙임)
        level="article" 처럼 특별한 접미사가 없는 경우 → 그냥 첫 줄 텍스트를 라벨로 사용
        (예: "제5조(적용범위)" 자체가 라벨이 됨)
    """
    if node.level in _LEVEL_SUFFIX and node.number:
        marker = node.number.strip()
        suffix = _LEVEL_SUFFIX[node.level]
        # 이미 "2호"처럼 접미사가 붙어있는 문자열이면 중복으로 더 붙이지 않는다.
        return marker if marker.endswith(suffix) else f"{marker}{suffix}"
    # 항/호/목/세목이 아닌 노드(예: article, byeolpyo_root)는 첫 줄 그대로를 라벨로 사용.
    first_line = (node.text or "").split("\n", 1)[0].strip()
    return first_line


def _push(stack: list, depth: int, **kwargs) -> LawNode:
    """새 계층 노드를 트리에 끼워넣는 헬퍼 함수. "스택(stack)" 기반 트리 빌드 기법이다.

    stack은 "현재까지 내려온 경로"를 나타낸다. 예를 들어 stack = [root, article, clause]
    라면 지금 우리는 "clause(항)" 밑에 뭔가를 추가하려는 상황이다.

    depth 매개변수는 "새 노드가 몇 번째 깊이에 들어가야 하는가"를 의미한다.
    - stack이 depth보다 깊으면(더 깊은 곳까지 내려가 있으면) → 그 초과분을 pop해서 제거
      (예: 이전에 "가.나.다" 목까지 내려갔다가, 새로 "2." 같은 호가 나오면 목 계층은 닫아야 함)
    - stack이 depth보다 얕으면(중간 계층 없이 건너뛴 경우) → "_gap" 이라는 빈 채움 노드로
      메꿔서 트리 깊이를 맞춘다. (예: 항 없이 바로 "1. ..." 호가 나오는 문서 대응)
    """
    while len(stack) > depth:
        stack.pop()
    while len(stack) < depth:
        # 중간 계층이 비어있으면 "_gap0", "_gap1" 같은 이름의 빈 노드로 채워
        # 트리 깊이(depth)를 일치시킨다. 이 gap 노드는 실제 조문 번호가 없는
        # "구조상 자리만 차지하는" 노드다.
        filler = LawNode(level=f"_gap{len(stack)}")
        stack[-1].children.append(filler)
        stack.append(filler)
    node = LawNode(**kwargs)
    stack[-1].children.append(node)  # 스택의 맨 위(현재 부모) 밑에 새 노드를 자식으로 추가
    stack.append(node)               # 이제 이 새 노드가 "현재 위치"가 됨
    return node


def _absorb(node: LawNode, line: str, is_table: bool, is_formula: bool) -> None:
    """새 계층을 시작하지 않는 일반 텍스트 줄(또는 표/산식 줄)을, 현재 가장 안쪽에 있는
    노드(스택 맨 위)의 text에 그냥 이어붙인다("흡수"시킨다는 의미).
    """
    node.text = f"{node.text}\n{line}" if node.text else line
    if is_table:
        # 이 노드 안에 표가 섞여 있다는 표시를 남긴다. 뒤에서 이 표시가 있으면
        # 토큰 기준으로 더 잘게 쪼개지 않고 통째로 보존한다(to_parent_child에서 사용).
        node.metadata["is_table_block"] = True
    if is_formula:
        node.metadata["is_formula_block"] = True


def build_outline_jomun(lines: list) -> LawNode:
    """조문 본문(줄 리스트)을 항(①)-호(1.)-목(가.)-세목(1)) 4단계 트리로 변환한다.

    동작 원리: 줄을 하나씩 보면서
      1) 이 줄이 표/산식이면 → 그냥 현재 노드에 흡수(_absorb)
      2) 아니면 LEVEL_PATTERNS_JOMUN을 순서대로 검사해서, 어느 계층 패턴과 일치하는지 찾는다.
         일치하면 그 계층의 새 노드를 만들어 트리에 추가(_push)하고, 이후 줄들은
         "다음 계층 헤더가 나올 때까지" 이 새 노드에 흡수된다.
      3) 어떤 패턴과도 일치하지 않으면 → 지금까지 내려온 가장 안쪽 노드에 흡수.
    """
    root = LawNode(level="article_body")
    stack = [root]  # 스택의 시작은 root 하나뿐 (아직 항/호/목/세목 어디에도 안 들어간 상태)
    for raw in lines:
        t, f = is_table_line(raw), is_formula_line(raw)
        if t or f:
            # 표/산식 줄은 계층 판별 없이 무조건 "현재 가장 안쪽 노드"에 흡수한다.
            _absorb(stack[-1], raw, t, f)
            continue
        line = raw.strip()
        matched = False
        # depth=1(항) → depth=2(호) → depth=3(목) → depth=4(세목) 순서로 검사.
        for depth, (level_name, pat) in enumerate(LEVEL_PATTERNS_JOMUN, start=1):
            m = pat.match(line)
            if m:
                _push(stack, depth, level=level_name, number=m.group(1), text=raw)
                matched = True
                break  # 계층 하나가 매칭됐으면 나머지 계층 패턴은 더 검사할 필요 없음
        if not matched:
            # 어떤 계층 패턴에도 안 걸리는 평범한 문장(설명, 부연 등) → 그냥 흡수.
            _absorb(stack[-1], raw, False, False)
    return root


def dotted_depth(number: str) -> int:
    """"4.1.2" 같은 점 표기 번호의 깊이를 센다. 점(.) 개수 + 1이 곧 깊이다.
    예) "4" → 깊이 1, "4.1" → 깊이 2, "4.1.2" → 깊이 3
    """
    return number.count(".") + 1


def build_outline_byeolpyo(lines: list) -> LawNode:
    """별표(표 형식 첨부문서) 본문을 트리로 변환한다. build_outline_jomun과 원리는
    같지만, 별표는 조문과 번호 체계가 달라서(점으로 구분된 번호, 단순 번호, 가/나/다)
    아래 3가지 패턴을 우선순위대로 검사한다.
    """
    root = LawNode(level="byeolpyo_root")
    stack = [root]
    for raw in lines:
        t, f = is_table_line(raw), is_formula_line(raw)
        if t or f:
            _absorb(stack[-1], raw, t, f)
            continue
        line = raw.strip()
        # 1순위: "4.1.2 " 처럼 점으로 구분된 번호 → dotted_depth()로 계산한 깊이만큼 push
        m = _DOTTED_RE.match(line)
        if m:
            depth = dotted_depth(m.group(1))
            _push(stack, depth, level=f"L{depth}", number=m.group(1), text=raw)
            continue
        # 2순위: "1. 수행능력평가(30점)" 처럼 점 없는 단순 번호 → 항상 depth=1(최상위 항목)
        m = _PLAIN_ITEM_RE.match(line)
        if m:
            _push(stack, 1, level="byeolpyo_item", number=m.group(1), text=raw)
            continue
        # 3순위: "가. ..." 처럼 한글 순번 → depth=2 (앞의 숫자 항목 밑에 딸린 하위 항목으로 취급)
        m = _SUB_RE.match(line)
        if m:
            _push(stack, 2, level="byeolpyo_sub", number=m.group(1), text=raw)
            continue
        # 어느 패턴에도 안 맞으면 표/설명문 등으로 보고 그냥 흡수.
        _absorb(stack[-1], raw, False, False)
    return root


# ============================================================
# 6) 경계탐지 우선 분리
#
# 여기서부터가 위에서 설명한 "boundary-first" 전략의 실제 구현이다.
# split_top_level()이 문서 전체를 [조문][별표][부칙] 3구역으로 먼저 잘라내고,
# 그 다음 함수들이 각 구역 안에서 다시 개별 조문/별표/부칙 항목 단위로 잘게 자른다.
# ============================================================
def split_top_level(full_text: str) -> dict:
    """문서 전체 텍스트를 "조문 영역 / 별표 영역 / 부칙 영역" 3덩어리로 나눈다.

    순서가 중요하다: 먼저 "부칙이 처음 시작되는 위치"를 찾아서 그 뒤 전부를
    부칙 영역으로 떼어낸 다음, 남은 앞부분(body_text)에서 다시 "별표가 처음
    시작되는 위치"를 찾아 별표 영역을 떼어낸다. 이렇게 부칙 → 별표 → 조문 순서로
    "뒤에서부터" 잘라내기 때문에, 조문 패턴 검사는 이미 부칙/별표가 제거된
    순수한 조문 영역에서만 이뤄져서 오분류가 원천적으로 방지된다.
    """
    # --- 1단계: 부칙 떼어내기 ---
    buchil_hits = list(PAT_BUCHIL.finditer(full_text))
    # 부칙이 아예 없는 문서라면 buchil_start를 텍스트 끝으로 둬서 "부칙 영역이 빈 문자열"이 되게 한다.
    buchil_start = buchil_hits[0].start() if buchil_hits else len(full_text)
    body_text = full_text[:buchil_start]     # 부칙 시작 전까지 = 조문+별표가 있을 영역
    buchil_text = full_text[buchil_start:]   # 부칙 시작 이후 전부 = 부칙 영역

    # --- 2단계: 남은 body_text에서 별표 떼어내기 ---
    byeolpyo_hits = list(PAT_BYEOLPYO.finditer(body_text))
    byeolpyo_start = byeolpyo_hits[0].start() if byeolpyo_hits else len(body_text)
    jomun_text = body_text[:byeolpyo_start]      # 별표 시작 전까지 = 순수 조문 영역
    byeolpyo_text = body_text[byeolpyo_start:]   # 별표 시작 이후 = 별표 영역

    return {"jomun": jomun_text, "byeolpyo": byeolpyo_text, "buchil": buchil_text}


def _slice_by_hits(text: str, hits: list) -> list:
    """정규식으로 찾은 "매칭 위치들(hits)"을 기준으로 텍스트를 조각조각 자른다.

    각 hit은 "다음 hit이 시작되기 직전까지"를 자신의 구간으로 갖는다.
    (예: 제1조 매칭 위치 ~ 제2조 매칭 직전까지가 "제1조 전체 내용"이 되는 방식)
    마지막 hit은 텍스트 끝까지를 자신의 구간으로 갖는다.
    반환값은 (매칭객체, 그 구간의 텍스트) 튜플의 리스트.
    """
    spans = []
    for i, h in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        spans.append((h, text[h.start():end]))
    return spans


def split_jomun_articles(jomun_text: str):
    """조문 영역을 "조문이 시작되기 전 머리말(intro)"과 "조문별 (매치, 원문) 리스트"로 나눈다.

    intro는 예를 들어 "○○에 관한 규정"처럼 제1조가 나오기 전에 있는 문서 제목/전문 같은
    텍스트다. 조문이 하나도 없으면(hits가 비어있으면) 전체를 intro로 취급하고 빈 리스트를 반환.
    """
    hits = list(PAT_JOMUN.finditer(jomun_text))
    if not hits:
        return jomun_text.strip(), []
    intro = jomun_text[:hits[0].start()].strip()   # 첫 조문이 시작되기 전까지의 텍스트
    return intro, _slice_by_hits(jomun_text, hits)


def split_byeolpyo_tables(byeolpyo_text: str) -> list:
    """별표 영역을 "별표 하나씩"으로 자른다. (별표1, 별표2, ... 각각의 구간)"""
    return _slice_by_hits(byeolpyo_text, list(PAT_BYEOLPYO.finditer(byeolpyo_text)))


def split_buchil_entries(buchil_text: str) -> list:
    """부칙 영역을 "부칙 항목 하나씩"으로 자른다. (개정 이력이 여러 번이면 여러 개로 나뉨)"""
    return _slice_by_hits(buchil_text, list(PAT_BUCHIL.finditer(buchil_text)))


# ============================================================
# 7) 조문 / 별표 루트 빌드 (헤더 분리 + 아웃라인 빌더 호출)
# ============================================================
def _build_article_root(m: re.Match, span_text: str):
    """"조문 하나의 구간 텍스트(span_text)"를 받아서 실제 LawNode 트리로 만든다.

    span_text는 "제5조(적용범위)\n① ... \n1. ..." 처럼 헤더 줄 + 본문 줄들로 이뤄져 있다.
    먼저 헤더(제목) 부분을 떼어내고, 나머지 본문(remainder)을 줄 단위로 쪼갠 뒤
    build_outline_jomun()에 넘겨서 항-호-목-세목 트리를 만든다.
    """
    header_text = m.group(0)                     # 정규식이 실제로 매칭한 전체 문자열 (예: "제5조(적용범위)")
    title = (m.group(3) or "").strip()            # 괄호 안 제목 (예: "적용범위")
    num, sub = m.group(1), m.group(2)             # 조 번호, 가지번호 (예: "5", None)
    remainder = span_text[len(header_text):]      # 헤더를 제외한 나머지 본문 부분

    # 헤더 바로 뒤에 줄바꿈 없이 본문이 이어붙는 경우와, 줄바꿈 후 본문이 시작되는
    # 경우를 모두 다루기 위해 첫 줄바꿈 위치를 찾는다.
    first_newline = remainder.find("\n")
    if first_newline == -1:
        # 줄바꿈이 아예 없으면(한 줄짜리 아주 짧은 조문) 그 전체를 본문 한 줄로 취급.
        body_lines = [remainder.strip()] if remainder.strip() else []
    else:
        # 헤더와 같은 줄에 붙어 이어지는 텍스트(first_line_remainder)와
        # 그 이후의 줄들(rest_lines)을 합쳐서 본문 줄 목록을 만든다.
        first_line_remainder = remainder[:first_newline].strip()
        rest_lines = remainder[first_newline + 1:].split("\n")
        body_lines = ([first_line_remainder] if first_line_remainder else []) + rest_lines

    root = build_outline_jomun(body_lines)                      # 본문을 항-호-목-세목 트리로 변환
    root.text = f"{header_text}\n{root.text}" if root.text else header_text  # 트리의 최상단에 헤더 줄을 다시 붙여줌
    root.level = "article"
    root.number = header_text
    article_no = f"제{num}조" + (f"의{sub}" if sub else "")      # cross-ref 조회 키로 쓸 표준화된 조문 번호(예: "제5조", "제5조의2")
    meta = {"article_no": article_no, "article_title": title, "doc_type": "조문"}
    return root, meta


def _build_byeolpyo_root(m: re.Match, span_text: str):
    """별표 하나의 구간 텍스트를 받아서 LawNode 트리로 만든다. _build_article_root와
    거의 같은 로직이지만, 별표는 헤더가 항상 "첫 줄 전체"이므로 좀 더 단순하다.
    """
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
    table_no = f"별표{m.group(1)}"   # cross-ref 조회 키로 쓸 표준화된 별표 번호(예: "별표3")
    meta = {"table_no": table_no, "table_title": header_text, "doc_type": "별표"}
    return root, meta


# ============================================================
# 8) 부칙 시행일 파싱
# ============================================================
def parse_buchil_dates(entry_text: str, published_date_raw: str) -> dict:
    """부칙 항목 텍스트 하나에서 "공포일"과 "시행일(및 예외 규정들)"을 뽑아낸다.

    부칙에는 보통 여러 문장이 있는데, 그중 "~부터 시행한다"는 문장이 하나 이상 있을 수
    있다(주된 시행일 + "다만 제3조는 2025년부터 시행한다" 같은 예외 규정).
    그래서 문장을 하나씩 훑으면서 시행일 표현이 있는 문장을 전부 찾고,
    맨 처음 찾은 것을 "주 시행일(primary)"로, 그 이후에 찾은 것들은 "예외(exceptions)"로 분류한다.
    """
    published_date = _normalize_date(published_date_raw)
    # 문장을 나누는 기준: 마침표/'다' 뒤의 줄바꿈, 또는 "시행한다." 라는 문구 뒤.
    # (법령 문장이 항상 깔끔하게 마침표로 안 끝나는 경우가 있어서 두 가지 기준을 OR로 사용)
    sentences = re.split(r'(?<=[.다])\s*\n|(?<=시행한다\.)\s*', entry_text)
    primary, exceptions = None, []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        eff = None  # 이 문장에서 계산된 시행일(없으면 None)
        if PAT_EFFECTIVE_ON_PROMULGATION.search(sent):
            # "공포한 날부터 시행" → 시행일 = 공포일 그 자체
            eff = published_date
        else:
            m = PAT_EFFECTIVE_AFTER_DAYS.search(sent)
            if m:
                # "공포 후 N일이 경과한 날부터 시행" → 공포일 + N일
                eff = _add_days(published_date_raw, int(m.group(1)))
            else:
                m = PAT_EFFECTIVE_LITERAL.search(sent)
                if m:
                    # "YYYY.MM.DD부터 시행" → 그 날짜를 그대로 사용
                    eff = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        if not eff:
            continue  # 이 문장에는 시행일 정보가 없음 → 건너뜀
        if primary is None:
            primary = eff  # 맨 처음 발견한 시행일 = 주 시행일
        else:
            # 두 번째 이후로 발견된 시행일들은 "특정 조항에만 적용되는 예외 시행일"로 간주.
            exceptions.append({"scope": sent, "effective_date": eff})
    return {
        "published_date": published_date,
        "effective_date": primary or published_date,  # 시행일 문장을 아예 못 찾았으면 공포일로 대체(안전한 폴백)
        "effective_date_exceptions": exceptions,
    }


# ============================================================
# 9) Parent/Child 변환 — 토큰 기준 재귀 승격
#
# 지금까지 만든 LawNode 트리는 "완전한 계층 구조"이지만, 이대로 그냥 하나하나를
# 다 별도의 부모로 만들면 부모가 너무 잘게 쪼개져서 문맥이 부족해질 수 있고,
# 반대로 조문 전체를 통째로 부모 하나로 만들면 너무 커서 임베딩/검색 효율이 떨어진다.
#
# 그래서 to_parent_child()는 "이 노드(및 하위 전체)의 텍스트가 max_parent_tokens
# 이하로 충분히 작다면 여기서 부모를 끊고, 그렇지 않다면 더 아래(자식들)로 내려가서
# 각각을 다시 검사한다"는 재귀적인 규칙으로 "적당한 크기의 부모 경계"를 자동으로 찾는다.
# ============================================================
def to_parent_child(node: LawNode, breadcrumb: list, cfg: dict) -> list:
    """LawNode 트리를 순회하면서 "부모로 삼을 만한 크기의 경계"를 찾아 ParentGroup
    리스트로 변환한다. (재귀 함수)

    종료 조건(부모를 여기서 끊는 경우) 3가지:
      1) 이 노드에 더 이상 자식이 없다 (말단 노드)
      2) 표/산식이 섞여 있어서 "보호 구역"으로 표시된 노드다 (쪼개면 표가 깨지므로 통째로 유지)
      3) 이 노드+하위 전체를 합친 텍스트가 max_parent_tokens 이하로 이미 충분히 작다
    """
    full_text = render_full_text(node)
    protected = bool(node.metadata.get("is_table_block") or node.metadata.get("is_formula_block"))
    if not node.children or protected or count_tokens(full_text) <= cfg["max_parent_tokens"]:
        # 여기서 부모 하나를 확정한다. 자식 청크로 쪼갤 대상(child_source_nodes)은
        # 이 노드의 children이 있으면 그것들, 없으면(말단 노드) 이 노드 자신 하나를
        # "leaf(잎)" 노드로 감싸서 사용한다. (자식이 하나도 없으면 부모=자식이 되는 셈)
        child_sources = node.children if node.children else [
            LawNode(level=f"{node.level}_leaf", text=node.text, metadata=dict(node.metadata))
        ]
        return [ParentGroup(node, list(breadcrumb), child_sources)]

    # 아직 너무 크다 → 부모를 여기서 끊지 않고, 자식들 각각에 대해 재귀적으로 더 내려가서
    # "적당히 작은" 경계를 찾는다.
    groups = []
    preface = node.text  # 이 노드 자신에게 직접 달려있던 텍스트(예: "① 이 규정은 다음 각 호에 적용한다." 라는 항의 도입부)
    for child in node.children:
        sub_groups = to_parent_child(child, breadcrumb + [breadcrumb_label(child)], cfg)
        if preface and sub_groups:
            # 이 노드 자신의 텍스트(preface, 예: 항의 도입 문구)를 잃어버리지 않도록,
            # 첫 번째로 만들어진 하위 ParentGroup의 맨 앞에 붙여준다.
            # (한 번 붙이고 나면 preface를 비워서 두 번 중복으로 안 붙게 함)
            first = sub_groups[0].parent_node
            first.text = f"{preface}\n{first.text}" if first.text else preface
            preface = ""
        groups.extend(sub_groups)
    return groups


def materialize_parent_group(group: ParentGroup, law_title: str, doc_type: str, cfg: dict):
    """ParentGroup(설계도)을 실제 LlamaIndex TextNode/IndexNode 객체로 "실체화"한다.

    - 부모(parent_tn): TextNode 하나. 실제 검색 결과로 사용자에게 보여줄 문맥 단위.
    - 자식(children): IndexNode 여러 개. 임베딩되어 실제 벡터 검색의 대상이 되는 단위.
      IndexNode.from_text_node(..., index_id=parent_tn.node_id) 로 만들면, 이 자식이
      검색됐을 때 "이 자식의 부모는 parent_tn이다"라는 연결정보가 함께 저장되어,
      나중에 LlamaIndex의 RecursiveRetriever 등이 부모를 자동으로 함께 가져올 수 있다.
    """
    parent_text = render_full_text(group.parent_node)
    parent_metadata = {
        "law_title": law_title,                              # 법령/문서 제목
        "doc_type": doc_type,                                 # "조문" / "별표" 등 문서 종류
        "type": "parent",                                     # 부모 노드임을 표시
        "breadcrumb": " > ".join(group.breadcrumb),           # 최상위부터 여기까지의 경로 문자열
    }
    parent_metadata.update(group.parent_node.metadata)  # is_table_block 같은 LawNode 자체 메타데이터도 합침
    if group.parent_node.level == "clause" and group.parent_node.number:
        # 이 부모가 "항(①②③...)" 단위라면, 나중에 "제5조제2항"처럼 항 단위로
        # cross-ref를 정확히 찾을 수 있도록 항 기호를 별도 메타데이터로 저장해둔다.
        parent_metadata["clause_marker"] = group.parent_node.number

    parent_tn = TextNode(text=parent_text, metadata=parent_metadata)
    # 이 부모 텍스트 안에 "제3조", "별표2" 같은 다른 조항/별표를 언급하는 부분이 있는지
    # 미리 찾아서 임시로 저장해둔다. 이 시점에는 아직 "제3조"가 실제로 어떤 node_id인지
    # 모르기 때문에(아직 문서 전체를 다 안 만들었으므로) 일단 "표면 텍스트"만 저장해두고,
    # 문서 전체 처리가 끝난 뒤 resolve_cross_refs()에서 실제 node_id로 2차 변환한다(2-pass).
    parent_tn.metadata["_cross_ref_surface_tokens"] = [m.group(0) for m in REF_PAT.finditer(parent_text)]

    children = []
    for src in group.child_source_nodes:
        text = render_full_text(src)
        if not text.strip():
            continue  # 내용이 아예 없는 노드는 자식으로 만들 필요 없음(예: 빈 _gap 노드)
        protected = bool(src.metadata.get("is_table_block") or src.metadata.get("is_formula_block"))
        if protected or count_tokens(text) <= cfg["max_child_tokens"]:
            # 표/산식이거나 이미 충분히 짧으면 그대로 자식 조각 1개로 사용.
            pieces = [text]
        else:
            # 너무 길면 토큰 슬라이딩 윈도우로 강제 분할(문맥 겹침 유지).
            pieces = tiktoken_window_split(text, cfg["max_child_tokens"], cfg["child_overlap_tokens"])

        for idx, piece in enumerate(pieces):
            label = breadcrumb_label(src)
            if len(pieces) > 1:
                # 한 노드가 여러 조각으로 더 쪼개진 경우, "(1/3)"처럼 몇 번째 조각인지 라벨에 표시.
                label = f"{label} ({idx + 1}/{len(pieces)})"
            breadcrumb_str = " > ".join(group.breadcrumb + [label])
            child_metadata = {
                "law_title": law_title,
                "doc_type": doc_type,
                "type": "child",
                "breadcrumb": breadcrumb_str,
            }
            child_metadata.update(src.metadata)
            # 자식 조각 단위로도 상호참조 표면 토큰을 찾아 저장(부모와 별개로 자식 자체에도 필요할 수 있음).
            child_metadata["_cross_ref_surface_tokens"] = [m.group(0) for m in REF_PAT.finditer(piece)]
            child_tn = TextNode(
                # CHILD_TEMPLATE으로 "법령제목 > 경로\n내용: ..." 형태의 최종 임베딩 대상 텍스트를 만든다.
                text=CHILD_TEMPLATE.format(law_title=law_title, breadcrumb=breadcrumb_str, content=piece),
                metadata=child_metadata,
            )
            # index_id=parent_tn.node_id 로 "이 자식의 부모가 누구인지"를 연결한다.
            children.append(IndexNode.from_text_node(child_tn, index_id=parent_tn.node_id))
    return parent_tn, children


# ============================================================
# 10) Cross-reference 2-pass 해결
#
# 법령 문장에는 "제3조에 따라", "별표2를 참고" 처럼 다른 조항/별표를 가리키는
# 표현이 자주 나온다. 이런 "언급(mention)"을 실제 검색 가능한 node_id로
# 연결해주는 기능이 cross-ref 해결이다.
#
# 왜 "2-pass"인가?
#   1-pass(먼저): materialize_parent_group()에서 각 노드를 만들 때, "이 텍스트 안에
#     제3조라는 언급이 있다"는 표면 텍스트만 저장해둔다(_cross_ref_surface_tokens).
#     이 시점에는 아직 문서 전체의 모든 조문/별표가 다 안 만들어졌을 수 있어서,
#     "제3조가 실제로 어떤 node_id인지"를 바로 알 수 없다.
#   2-pass(나중): 문서 전체(또는 여러 문서 전체)의 모든 조문/별표가 다 만들어지고
#     article_index / byeolpyo_index 같은 "번호 → node_id 매핑표"가 완성된 뒤에,
#     그제서야 저장해뒀던 표면 텍스트들을 실제 node_id로 변환(resolve)한다.
# ============================================================
def _resolve_token(tok: str, article_index: dict, article_paragraph_index: dict, byeolpyo_index: dict) -> list:
    """"제3조", "별표2", "제3조제2항" 같은 언급 문자열 하나(tok)를 실제 node_id
    리스트로 변환한다. 매칭되는 게 없으면 빈 리스트를 반환한다.
    """
    tok = tok.strip()
    m = _TOK_BYEOLPYO_RE.match(tok)
    if m:
        # "별표N" 형태 → byeolpyo_index에서 그 번호에 해당하는 노드들을 찾음
        return list(byeolpyo_index.get(f"별표{m.group(1)}", []))
    m = _TOK_ARTICLE_RE.match(tok)
    if m:
        article_no = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        clause = m.group(3)  # "①②③..." 항 기호가 있으면(더 구체적인 참조)
        if clause:
            # "제3조제2항"처럼 항까지 특정된 경우, 그 항 노드를 정확히 찾아본다.
            node_id = article_paragraph_index.get((article_no, clause))
            if node_id:
                return [node_id]
            # 항까지 특정된 node를 못 찾으면, 아래로 내려가서 조 단위로라도 매칭 시도(폴백)
        # 항 지정이 없거나, 항 단위로 못 찾았으면 조 단위(article_no) 전체를 반환.
        return list(article_index.get(article_no, []))
    return []


def resolve_cross_refs(parent_nodes: list, article_index: dict, article_paragraph_index: dict, byeolpyo_index: dict) -> None:
    """모든 부모 노드를 순회하면서, 1-pass에서 저장해둔 "언급 표면 텍스트"들을
    실제 node_id로 변환하여 각 노드의 metadata에 cross_refs / cross_refs_unresolved로 저장한다.

    이 함수는 반환값이 없다(None) — 각 노드의 metadata를 직접 수정(in-place)하는 방식으로 동작한다.
    """
    for node in parent_nodes:
        # pop으로 꺼내면서 동시에 제거: 최종 결과물에는 이 임시 필드가 남지 않도록 정리도 함께 한다.
        tokens = node.metadata.pop("_cross_ref_surface_tokens", [])
        resolved, unresolved = [], []
        for tok in tokens:
            ids = _resolve_token(tok, article_index, article_paragraph_index, byeolpyo_index)
            if ids:
                resolved.extend(ids)
            else:
                # 참조 대상을 문서 안에서 못 찾은 경우(예: 다른 법령을 언급하는 경우 등) → 미해결 목록에 기록
                unresolved.append(tok)
        # set으로 중복 제거 후 정렬, 그리고 "자기 자신을 참조하는 경우"는 의미가 없으므로 자기 자신은 제외.
        node.metadata["cross_refs"] = sorted(set(resolved) - {node.node_id})
        node.metadata["cross_refs_unresolved"] = sorted(set(unresolved))


def _extract_law_title(llama_docs):
    """단일 doc 기준 (파일명, 법령제목) 추출.

    llama_docs는 리스트로 받지만 실제로는 [doc] 처럼 원소 1개짜리로만 호출된다
    (여러 문서를 합쳐 처리하지 않기 위한 관례). 메타데이터에서 file_name을 꺼내
    확장자를 뗀 것을 법령 제목으로 사용하고, 못 찾으면 "답변자료"라는 기본값을 쓴다.

    self를 쓰지 않는 순수 함수라 그래프 노드에서 바로 재사용하도록 모듈 레벨로 옮김.
    """
    file_name, law_title = None, "답변자료"
    if llama_docs:
        first_meta = getattr(llama_docs[0], "metadata", {}) or {}
        if isinstance(first_meta, dict):
            file_name = first_meta.get("file_name")
            if file_name:
                law_title = os.path.splitext(file_name)[0]  # "규정.pdf" → "규정"
    return file_name, law_title


# ============================================================
# 11) 영역별 처리 함수 (조문 / 별표 / 부칙)
#
# 여기서부터는 지금까지 만든 부품 함수들(정규식, 트리 빌더, parent/child 변환기 등)을
# 실제로 조합해서 "조문 영역 텍스트 전체" 또는 "별표 영역 텍스트 전체"를 받아
# 최종 TextNode/IndexNode들을 만들어내는 상위 조립 함수 3개다.
#
# 원본 코드에서는 이 3가지 로직이 하나의 큰 if/elif 함수(_process_region_worker)
# 안에 다 들어있었는데, 여기서는 "조문 처리", "별표 처리", "부칙 처리"를 각각
# 독립된 함수로 분리했다. 이렇게 나누면:
#   - 함수 하나하나가 짧아져서 읽기 쉽고,
#   - REGION_HANDLERS 딕셔너리를 통해 "영역 이름 → 처리 함수" 로 자연스럽게 라우팅되며,
#   - process_document 노드에서 "텍스트가 있는 영역만 처리한다"는 조건부 로직이
#     간단한 dict 조회 + continue 문으로 표현된다.
#
# 각 함수는 반환값이 없다(None). 대신 함수 밖에서 미리 만들어둔 "누적용 그릇들"
# (doc_nodes, node_dict, article_index 등)을 인자로 넘겨받아 그 안에 결과를 직접
# 채워 넣는(append/업데이트) 방식으로 동작한다. 이런 패턴을 "부작용을 통한 누적"이라고
# 부르는데, 여러 함수가 "같은 문서 하나를 처리하는 동안"만 공유하는 임시 그릇이라서
# 위험하지 않다(문서가 바뀌면 매번 새 그릇을 만들기 때문).
# ============================================================
def process_jomun_region(*, region_text, law_title, cfg, doc_nodes, node_dict,
                          article_index, article_paragraph_index, byeolpyo_index,
                          parent_nodes, warnings):
    """조문 영역 텍스트 전체를 받아서, 조문 하나하나를 파싱하고 parent/child
    TextNode를 만들어 doc_nodes/node_dict 등에 채워 넣는다.
    """
    # 1) "제1조가 시작되기 전 머리말"과 "조문별 (매치, 원문) 리스트"로 분리
    intro, spans = split_jomun_articles(region_text)
    reconstructed_parts = [intro] if intro else []
    articles_groups = []  # 조문별로 만들어진 ParentGroup 리스트들을 모아두는 곳

    for m, span_text in spans:
        root, meta = _build_article_root(m, span_text)         # 조문 하나를 LawNode 트리로 변환
        reconstructed_parts.append(render_full_text(root))     # 나중에 "원문 유실 검사"를 위해 재구성 텍스트 누적
        groups = to_parent_child(root, [breadcrumb_label(root)], cfg)  # 적당한 크기의 부모 경계로 나눔
        for g in groups:
            # 조문 공통 메타데이터(article_no, article_title 등)를 각 ParentGroup의 부모 노드에 병합.
            # {**meta, **g.parent_node.metadata} 순서이므로, 만약 겹치는 키가 있으면
            # g.parent_node.metadata(더 구체적인 정보)가 우선한다.
            g.parent_node.metadata = {**meta, **g.parent_node.metadata}
        articles_groups.append(groups)

    # 2) "손실 검사(reconstruction check)": 우리가 쪼개고 다시 이어붙인 텍스트가
    #    원본과 얼마나 똑같은지를 difflib으로 비교해서 비율(ratio, 0~1)을 구한다.
    #    이 비율이 기준(reconstruction_min_ratio)보다 낮으면 "파싱 중 텍스트가
    #    유실됐을 수 있다"는 경고를 남긴다(정규식이 처리 못한 예외적인 문서 형태를
    #    조기에 발견하기 위한 안전장치).
    ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), region_text.strip()).ratio()
    ok = ratio >= cfg["reconstruction_min_ratio"]
    if not ok:
        warnings.append(f"[{law_title}] 조문 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")

    # 3) intro(제1조 이전의 머리말)도 하나의 "부모" 취급해서 노드로 만든다.
    intro_text = (intro or "").strip()
    if intro_text:
        pieces = [intro_text] if count_tokens(intro_text) <= cfg["max_parent_tokens"] \
            else tiktoken_window_split(intro_text, cfg["max_parent_tokens"], cfg["child_overlap_tokens"])
        for piece in pieces:
            p_node = TextNode(text=piece, metadata={
                "law_title": law_title, "doc_type": "intro", "type": "parent",
                "reconstruction_ratio": ratio, "reconstruction_ok": ok,
            })
            doc_nodes.append(p_node)
            node_dict[p_node.node_id] = p_node

    # 4) 조문별 ParentGroup들을 실제 TextNode/IndexNode로 실체화하고,
    #    상호참조 해결에 필요한 인덱스(article_index 등)도 함께 채운다.
    for groups in articles_groups:
        for g in groups:
            parent_tn, child_nodes = materialize_parent_group(g, law_title, "조문", cfg)
            # 이 영역 전체의 손실검사 결과(ratio/ok)를 부모/자식 노드 각각에 남겨서,
            # 나중에 "이 청크가 신뢰할 만한 파싱 결과인지"를 판단할 수 있게 한다.
            parent_tn.metadata["reconstruction_ratio"] = ratio
            parent_tn.metadata["reconstruction_ok"] = ok
            for c in child_nodes:
                c.metadata["reconstruction_ratio"] = ratio
                c.metadata["reconstruction_ok"] = ok

            node_dict[parent_tn.node_id] = parent_tn   # node_id → 노드 객체 조회용 사전에 등록
            doc_nodes.append(parent_tn)                # 이 문서의 전체 노드 리스트에 부모 추가
            doc_nodes.extend(child_nodes)               # 자식들도 전부 추가
            parent_nodes.append(parent_tn)              # "부모 노드만 모은 리스트"에도 별도로 추가 (나중에 cross-ref 해결 시 이 리스트만 순회하면 되도록)

            # cross-ref 조회용 인덱스 구축: "제5조" → [해당 조문의 부모 node_id들]
            article_no = parent_tn.metadata.get("article_no")
            if article_no:
                article_index[article_no].append(parent_tn.node_id)
                clause_marker = parent_tn.metadata.get("clause_marker")
                if clause_marker:
                    # 이 부모가 특정 "항" 단위라면, ("제5조", "①") → node_id 처럼
                    # 더 세밀한 조회를 위한 인덱스도 함께 만들어둔다.
                    article_paragraph_index[(article_no, clause_marker)] = parent_tn.node_id


def process_byeolpyo_region(*, region_text, law_title, cfg, doc_nodes, node_dict,
                             article_index, article_paragraph_index, byeolpyo_index,
                             parent_nodes, warnings):
    """별표 영역 텍스트 전체를 받아서, 별표 하나하나를 파싱하고 parent/child
    TextNode를 만들어 doc_nodes/node_dict 등에 채워 넣는다. process_jomun_region과
    구조가 거의 동일하되, "intro" 개념이 없고 article_no 대신 table_no를 쓴다.
    """
    spans = split_byeolpyo_tables(region_text)
    reconstructed_parts = []
    tables_groups = []

    for m, span_text in spans:
        root, meta = _build_byeolpyo_root(m, span_text)
        reconstructed_parts.append(render_full_text(root))
        groups = to_parent_child(root, [breadcrumb_label(root)], cfg)
        for g in groups:
            g.parent_node.metadata = {**meta, **g.parent_node.metadata}
        tables_groups.append(groups)

    # 조문 처리와 동일한 원리의 손실 검사.
    ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), region_text.strip()).ratio()
    ok = ratio >= cfg["reconstruction_min_ratio"]
    if not ok:
        warnings.append(f"[{law_title}] 별표 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")

    for groups in tables_groups:
        for g in groups:
            parent_tn, child_nodes = materialize_parent_group(g, law_title, "별표", cfg)
            parent_tn.metadata["reconstruction_ratio"] = ratio
            parent_tn.metadata["reconstruction_ok"] = ok
            for c in child_nodes:
                c.metadata["reconstruction_ratio"] = ratio
                c.metadata["reconstruction_ok"] = ok
            node_dict[parent_tn.node_id] = parent_tn
            doc_nodes.append(parent_tn)
            doc_nodes.extend(child_nodes)
            parent_nodes.append(parent_tn)

            # cross-ref 조회용 인덱스 구축: "별표3" → [해당 별표의 부모 node_id들]
            # (뒤에 있는 node_compute_peer_tables 노드가 이 byeolpyo_index를 이용해
            #  "이 별표와 같은 문서에 있는 다른 별표들" 목록(peer_tables)도 계산한다.)
            table_no = parent_tn.metadata.get("table_no")
            if table_no:
                byeolpyo_index[table_no].append(parent_tn.node_id)


def process_buchil_region(*, region_text, law_title, cfg, doc_nodes, node_dict,
                           article_index, article_paragraph_index, byeolpyo_index,
                           parent_nodes, warnings):
    """부칙 영역 텍스트 전체를 받아서, 부칙 항목 하나하나(개정 이력 각각)를
    파싱하고 TextNode를 만든다. 부칙은 항-호-목 같은 하위 트리 구조로 더 쪼개지
    않고(법 개정 이력이라 문장이 짧고 성격이 다름), 항목 하나 = 노드 하나로 취급한다.
    """
    spans = split_buchil_entries(region_text)
    reconstructed_parts = []

    for idx, (m, span_text) in enumerate(spans):
        law_no, published_date_raw = m.group(1), m.group(2)
        text = span_text.strip()
        reconstructed_parts.append(text)
        dates = parse_buchil_dates(span_text, published_date_raw)  # 공포일/시행일 등 계산
        node = TextNode(text=text, metadata={
            "law_title": law_title, "doc_type": "부칙", "type": "parent",
            "law_no": law_no,
            # spans는 문서에 등장하는 순서(보통 오래된 개정 → 최신 개정)이므로,
            # 리스트의 "마지막" 항목이 가장 최신 개정본이라고 판단한다.
            "is_latest": idx == len(spans) - 1,
            **dates,  # published_date, effective_date, effective_date_exceptions 를 그대로 펼쳐 넣음
        })
        doc_nodes.append(node)
        node_dict[node.node_id] = node

    # 조문/별표와 동일한 원리의 손실 검사.
    ratio = difflib.SequenceMatcher(None, "\n".join(reconstructed_parts).strip(), region_text.strip()).ratio()
    ok = ratio >= cfg["reconstruction_min_ratio"]
    if not ok:
        warnings.append(f"[{law_title}] 부칙 영역 텍스트 유실 의심 - 일치율 {ratio:.4f}")
    # 부칙 노드들은 위 루프에서 이미 doc_nodes에 다 추가된 뒤이므로, 여기서 한 번 더
    # 순회하면서 손실검사 결과(ratio/ok)를 사후에 채워 넣는다. 이미 다른 값이 채워진
    # 노드(다른 region 처리에서 온 노드)는 건드리지 않도록 "부칙 타입이면서 아직
    # reconstruction_ratio가 없는 노드"만 골라서 채운다.
    for n in doc_nodes:
        if n.metadata.get("doc_type") == "부칙" and "reconstruction_ratio" not in n.metadata:
            n.metadata["reconstruction_ratio"] = ratio
            n.metadata["reconstruction_ok"] = ok


# "영역 이름(jomun/byeolpyo/buchil)" → "그 영역을 처리할 함수"를 연결하는 라우팅 테이블.
# node_process_document()에서 이 딕셔너리를 이용해 영역별로 알맞은 처리 함수를 호출한다.
REGION_HANDLERS = {
    "jomun": process_jomun_region,
    "byeolpyo": process_byeolpyo_region,
    "buchil": process_buchil_region,
}


# ============================================================
# 12) LangGraph 상태(State) 정의
#
# LangGraph의 "상태(state)"는 그래프 전체가 공유하며 노드를 거칠 때마다 갱신되는
# 하나의 큰 딕셔너리라고 생각하면 된다. TypedDict로 "이 딕셔너리에는 어떤 키들이
# 들어있을 수 있는지"를 미리 선언해두면, 코드 자동완성/타입 체크에 도움이 된다.
#
# Annotated[list, operator.add] 가 붙은 필드(doc_results, warnings)는 특별하다:
# 여러 노드가 "동시에(병렬로)" 같은 state 키를 갱신하려고 할 때(Send로 팬아웃된 경우),
# LangGraph는 기본적으로 "마지막에 쓴 값으로 덮어쓰기"를 하는데, 이렇게 되면 병렬로
# 실행된 문서 결과들이 서로를 덮어써서 유실된다. operator.add를 reducer(병합 함수)로
# 지정해두면, 여러 곳에서 온 리스트들을 "덮어쓰기" 대신 "이어붙이기(list + list)"로
# 자동 병합해준다. 그래서 process_document가 문서마다 {"doc_results": [결과]}를
# 반환해도, 최종적으로는 모든 문서의 결과가 doc_results 리스트 하나에 다 모인다.
# ============================================================
class DocTask(TypedDict):
    """Send로 process_document 노드를 팬아웃할 때 각 분기에 전달되는 입력 하나."""
    doc: Any     # llama_index Document 객체 1개
    cfg: dict    # 파이프라인 설정값


class DocResult(TypedDict):
    """process_document 노드 하나가 문서 1건을 다 처리한 뒤 만들어내는 결과 묶음.
    아직 다른 문서들과 합쳐지기 전, "이 문서 하나만의" 부분 결과라는 뜻에서
    필드 이름에 "_partial"이 붙은 것들이 있다.
    """
    doc_nodes: list                        # 이 문서에서 만들어진 모든 TextNode/IndexNode (부모+자식)
    node_dict: dict                        # node_id → 노드 객체 조회용 사전 (이 문서분만)
    article_index_partial: dict            # "제5조" → [node_id, ...] (이 문서분만)
    article_paragraph_index_partial: dict  # ("제5조", "①") → node_id (이 문서분만)
    byeolpyo_index_partial: dict           # "별표3" → [node_id, ...] (이 문서분만)
    parent_nodes: list                     # 이 문서에서 만들어진 "부모" 노드만 모은 리스트
    warnings: list                         # 이 문서를 처리하며 발생한 경고 메시지들


class GraphState(TypedDict, total=False):
    """그래프 전체가 공유하는 상태. total=False라서 모든 키가 "있어도 되고 없어도
    되는" 선택적(optional) 필드로 취급된다(그래프 진행 단계마다 채워지는 키가 다르므로).
    """
    llama_docs: list                                  # 입력: 처리할 문서 리스트
    cfg: dict                                          # 입력: 설정값
    structure_stats: dict                              # detect_structure 노드의 결과 (조문/별표/부칙 존재 여부)
    doc_results: Annotated[list, operator.add]         # process_document들의 결과가 자동으로 모이는 리스트 (Send 팬인)
    warnings: Annotated[list, operator.add]            # 전체 파이프라인에서 쌓인 경고 메시지 (Send 팬인)
    all_nodes: list                                    # merge_documents 이후: 전체 문서를 합친 노드 리스트
    node_dict: dict                                    # merge_documents 이후: 전체 node_id → 노드 조회 사전
    article_index: dict                                # merge_documents 이후: 전체 문서 기준 조문 인덱스
    article_paragraph_index: dict                       # merge_documents 이후: 전체 문서 기준 조문+항 인덱스
    byeolpyo_index: dict                                # merge_documents 이후: 전체 문서 기준 별표 인덱스
    parent_nodes: list                                  # merge_documents 이후: 전체 부모 노드 리스트
    final_nodes: list                                   # 그래프의 최종 출력 (all_nodes 또는 폴백 결과)
    final_node_dict: dict                               # 그래프의 최종 출력 (node_dict 또는 폴백 결과)
    fallback: bool                                      # 폴백 경로로 처리됐는지 여부 표시


# ============================================================
# 13) 노드 1) 구조 탐지
#
# 이 노드는 그래프의 맨 처음에 실행되며, "이 문서 뭉치 안에 애초에 조문/별표/부칙
# 구조가 있기는 한가?"를 빠르게 훑어본다(전체 텍스트를 1번만 스캔). 이 결과에 따라
# 뒤에서 "계층 파싱 경로로 갈지, 그냥 폴백(fallback) 경로로 갈지"가 결정된다.
# ============================================================
def node_detect_structure(state: GraphState) -> dict:
    """LangGraph 노드 함수는 항상 "state를 입력받아, state에 병합할 딕셔너리를
    반환"하는 형태다. 여기서는 state["llama_docs"]만 읽고, {"structure_stats": {...}}
    를 반환해서 state에 structure_stats 키를 새로 추가(또는 갱신)한다.
    """
    llama_docs = state["llama_docs"]
    # 모든 문서의 텍스트를 하나로 합쳐서 정규화 후, 정규식으로 각 패턴이 몇 번 나오는지 센다.
    normalized = _normalize_text("\n".join(d.text for d in llama_docs))
    article_matches = len(list(PAT_JOMUN.finditer(normalized)))
    table_matches = len(list(PAT_BYEOLPYO.finditer(normalized)))
    addendum_matches = len(list(PAT_BUCHIL.finditer(normalized)))

    logger.info(
        f"[LawChunker] 구조 탐지 결과(전문 스캔) - 조문:{article_matches} "
        f"별표:{table_matches} 부칙:{addendum_matches}"
    )

    return {
        "structure_stats": {
            "has_article": article_matches > 0,
            "has_byeolpyo": table_matches > 0,
            "has_addendum": addendum_matches > 0,
            "article_matches": article_matches,
            "table_matches": table_matches,
            "addendum_matches": addendum_matches,
        }
    }


def route_after_detect(state: GraphState) -> str:
    """조건부 엣지(conditional edge) 함수. 일반 노드와 달리, state를 보고 "다음에
    어느 노드로 갈지"를 문자열로 반환한다. 이 문자열은 그래프 정의부(build_law_chunker_graph)
    에서 {"hierarchical": "prepare_documents", "fallback": "fallback_chunk"} 라는
    매핑을 통해 실제 노드 이름으로 변환된다.

    쉽게 말해 "if 조문있음 or 별표있음: 계층파싱으로 else: 폴백으로" 라는 if문을
    그래프의 화살표 분기로 표현한 것이다.
    """
    stats = state["structure_stats"]
    return "hierarchical" if (stats["has_article"] or stats["has_byeolpyo"]) else "fallback"


# ============================================================
# 14) 노드 2-a) 폴백 경로
#
# 조문도 별표도 하나도 발견되지 않은 문서(예: 일반 산문 형태의 답변자료)는
# 이 파일의 법령 전용 파서로 처리할 수 없으므로, 프로젝트 내 별도의
# "일반 텍스트용 폴백 청커(FallbackChunker)"에게 처리를 넘긴다.
# ============================================================
def node_fallback_chunk(state: GraphState) -> dict:
    logger.info("[LawChunker] fallback_chunk 시작")
    try:
        from utils.chunking_patterns.fallback_chunker import FallbackChunker  # 프로젝트 내 기존 폴백 구현체 (이 파일에는 포함되어 있지 않음)
    except ImportError:
        # 폴백 모듈 자체를 못 찾는 경우, 파이프라인이 죽지 않도록 빈 결과를 반환하고
        # 경고만 남긴다(운영 환경 구성 문제를 조기에 로그로 알 수 있게).
        logger.warning("[LawChunker] FallbackChunker를 찾을 수 없어 빈 결과를 반환합니다.")
        return {"final_nodes": [], "final_node_dict": {}, "fallback": True}

    fb = FallbackChunker()
    all_nodes, node_dict = fb.create_fallback_nodes(state["llama_docs"])
    return {"final_nodes": all_nodes, "final_node_dict": node_dict, "fallback": True}


# ============================================================
# 15) 노드 2-b) 문서 단위 팬아웃 준비
#
# LangGraph에서 Send로 팬아웃하려면, "조건부 엣지 함수"가 Send 객체 리스트를
# 반환해야 한다. 그런데 조건부 엣지는 항상 "어떤 노드 뒤에" 붙어야 하므로,
# route_after_detect가 이미 붙어있는 detect_structure 노드 뒤에 또 다른 조건부
# 엣지를 바로 붙일 수는 없다(엣지 하나에는 분기 함수 하나만). 그래서 사이에
# "prepare_documents"라는 아무 일도 안 하는(no-op) 빈 노드를 하나 끼워 넣어,
# 그 노드 뒤에 "fan_out_documents"라는 두 번째 조건부 엣지를 붙인다.
# ============================================================
def node_prepare_documents(state: GraphState) -> dict:
    logger.info("[LawChunker] prepare_documents (팬아웃 준비) 실행")
    return {}  # 상태를 바꾸지 않는 빈 노드. 오직 "다음에 Send 팬아웃을 붙이기 위한 자리"로만 존재.


def fan_out_documents(state: GraphState):
    """조건부 엣지: 문서 개수만큼 process_document를 병렬 분기(Send)한다.

    일반 조건부 엣지 함수는 "다음 노드 이름(문자열)"을 반환하지만, 이 함수는
    Send 객체들의 "리스트"를 반환한다. LangGraph는 이 리스트를 보고 "process_document
    노드를 리스트 길이만큼 각각 다른 입력으로 동시에 실행하라"고 해석한다.

    예) 문서가 3개면 → [Send("process_document", 문서1+cfg), Send(..., 문서2+cfg), Send(..., 문서3+cfg)]
        즉 process_document 노드가 3번, 서로 독립적인 입력을 가지고 실행된다.
    """
    cfg = state["cfg"]
    logger.info(f"[LawChunker] fan_out_documents: 문서 수={len(state.get('llama_docs', []))} 만큼 process_document로 분기")
    return [Send("process_document", {"doc": doc, "cfg": cfg}) for doc in state["llama_docs"]]


# ============================================================
# 16) 노드 3) 문서 단위 처리 (Send로 문서마다 1회씩 실행됨)
#
# 이 노드는 "문서 1개"만을 입력으로 받는다는 점이 특이하다(다른 노드들은 GraphState
# 전체를 받지만, 이 노드는 Send가 넘겨준 DocTask({"doc":..., "cfg":...})만 받는다).
# Send로 팬아웃된 노드는 항상 이렇게 "자신에게 전달된 조각 데이터"만 입력으로 받는다.
# ============================================================
def node_process_document(task: DocTask) -> dict:
    doc, cfg = task["doc"], task["cfg"]
    file_name, law_title = _extract_law_title([doc])
    logger.info(f"[LawChunker] process_document 시작: file={file_name or '<unknown>'}, title={law_title}")
    normalized = _normalize_text(doc.text)          # 마크다운 장식 제거
    regions = split_top_level(normalized)            # 부칙→별표→조문 순으로 boundary-first 3영역 분리

    # 이 문서 하나를 처리하는 동안만 쓰이는 "임시 그릇"들. 아래 REGION_HANDLERS
    # 안의 함수들이 이 그릇들에 직접 결과를 채워 넣는다(11번 섹션 설명 참고).
    doc_nodes, node_dict = [], {}
    article_index = defaultdict(list)
    article_paragraph_index = {}
    byeolpyo_index = defaultdict(list)
    parent_nodes = []
    warnings = []

    for region_name, region_text in regions.items():
        if not region_text.strip():
            # 조건부 분기: 예를 들어 이 문서에 별표가 하나도 없으면 byeolpyo_text가
            # 빈 문자열이 되므로, 이 영역은 아예 처리 함수를 호출하지 않고 건너뛴다.
            # (원본 코드의 "tasks = [(name, text) for name, text in regions.items() if text.strip()]"
            #  필터링과 동일한 효과를 if + continue로 표현한 것)
            continue
        REGION_HANDLERS[region_name](  # region_name("jomun"/"byeolpyo"/"buchil")에 맞는 처리 함수를 dict에서 찾아 호출
            region_text=region_text, law_title=law_title, cfg=cfg,
            doc_nodes=doc_nodes, node_dict=node_dict,
            article_index=article_index, article_paragraph_index=article_paragraph_index,
            byeolpyo_index=byeolpyo_index, parent_nodes=parent_nodes, warnings=warnings,
        )

    if file_name:
        # 나중에 "이 청크가 어느 원본 파일에서 왔는지"로 필터링 검색을 할 수 있도록,
        # 이 문서에서 만들어진 모든 노드에 파일명을 메타데이터로 달아준다.
        for n in doc_nodes:
            n.metadata["file_name"] = file_name

    # 이 문서의 처리 결과를 DocResult 형태로 정리한다. defaultdict는 그대로 두면
    # pickle/직렬화나 이후 처리에서 예상 못한 동작을 할 수 있으므로 dict(...)로
    # 평범한 딕셔너리로 변환해서 담는다.
    result: DocResult = {
        "doc_nodes": doc_nodes,
        "node_dict": node_dict,
        "article_index_partial": dict(article_index),
        "article_paragraph_index_partial": article_paragraph_index,
        "byeolpyo_index_partial": dict(byeolpyo_index),
        "parent_nodes": parent_nodes,
        "warnings": warnings,
    }
    # {"doc_results": [result]} 형태로 "리스트 안에 결과 1개"를 반환하는 이유:
    # GraphState의 doc_results 필드가 Annotated[list, operator.add]라서, 이렇게
    # 반환하면 LangGraph가 자동으로 다른 문서들의 [result]와 이어붙여(list + list)
    # 최종적으로 doc_results = [문서1결과, 문서2결과, ...] 형태로 모아준다.
    return {"doc_results": [result], "warnings": warnings}


# ============================================================
# 17) 노드 4) 문서별 결과 합류 (merge)
#
# Send로 병렬 실행된 process_document들의 결과가 doc_results 리스트에 다 모이고
# 나면, 이 노드가 실행되어 "여러 문서의 부분 결과들"을 "전체 문서를 아우르는
# 하나의 통합 인덱스"로 합친다. cross-ref 해결이나 peer_tables 계산은 "문서
# 하나만의 정보"로는 부족하고(다른 문서에 있는 별표/조문도 참조할 수 있으므로)
# 전체 문서를 합친 인덱스가 필요하기 때문에 이 병합 단계가 꼭 필요하다.
# ============================================================
def node_merge_documents(state: GraphState) -> dict:
    logger.info("[LawChunker] merge_documents 시작 - 문서별 결과 병합")
    all_nodes, node_dict = [], {}
    article_index = defaultdict(list)
    article_paragraph_index = {}
    byeolpyo_index = defaultdict(list)
    parent_nodes = []

    for r in state.get("doc_results", []):
        all_nodes.extend(r["doc_nodes"])
        node_dict.update(r["node_dict"])
        # 각 문서의 부분 인덱스(article_index_partial 등)를 전체 인덱스에 이어붙인다.
        # 서로 다른 문서라도 같은 "제5조"라는 키가 있을 수 있으므로(다른 법령의 제5조),
        # 딕셔너리를 그냥 update하지 않고 리스트를 extend해서 두 문서의 node_id가
        # 모두 같은 키 아래에 누적되도록 한다.
        for k, v in r["article_index_partial"].items():
            article_index[k].extend(v)
        article_paragraph_index.update(r["article_paragraph_index_partial"])
        for k, v in r["byeolpyo_index_partial"].items():
            byeolpyo_index[k].extend(v)
        parent_nodes.extend(r["parent_nodes"])

    return {
        "all_nodes": all_nodes,
        "node_dict": node_dict,
        "article_index": dict(article_index),           # defaultdict → 평범한 dict로 변환해서 state에 저장
        "article_paragraph_index": article_paragraph_index,
        "byeolpyo_index": dict(byeolpyo_index),
        "parent_nodes": parent_nodes,
    }


# ============================================================
# 18) 노드 5) 별표 간 peer_tables 계산
#
# "peer_tables"란 "같은 문서(또는 전체 배치) 안에 있는, 나 말고 다른 별표들의
# node_id 목록"이다. 예를 들어 별표1의 metadata에 peer_tables=[별표2의 id, 별표3의 id]
# 를 저장해두면, 검색 시스템이 "별표1을 찾았으면 다른 별표들도 참고용으로 함께
# 보여줄 수 있다"는 힌트로 활용할 수 있다.
# ============================================================
def node_compute_peer_tables(state: GraphState) -> dict:
    logger.info("[LawChunker] compute_peer_tables 시작 - 별표 피어 계산")
    byeolpyo_index = state["byeolpyo_index"]   # {"별표1": [node_id, ...], "별표2": [...], ...}
    node_dict = state["node_dict"]
    all_table_nos = list(byeolpyo_index.keys())   # ["별표1", "별표2", "별표3", ...]
    for table_no in all_table_nos:
        # "나(table_no)를 제외한 다른 모든 별표 번호"에 속한 node_id들을 다 모은다.
        peers = [nid for other in all_table_nos if other != table_no for nid in byeolpyo_index[other]]
        # table_no에 해당하는 모든 노드(예: 별표1이 여러 부모 청크로 쪼개졌을 수 있음)에
        # 똑같은 peers 목록을 메타데이터로 달아준다.
        for nid in byeolpyo_index[table_no]:
            node_dict[nid].metadata["peer_tables"] = peers
    return {"node_dict": node_dict}


# ============================================================
# 19) 노드 6) cross-ref 2-pass 해결
#
# 10번 섹션에서 설명한 대로, 전체 문서를 아우르는 인덱스(article_index 등)가
# 완성된 지금 시점에서야 비로소 "제3조" 같은 언급을 실제 node_id로 정확히
# 연결(resolve)할 수 있다. resolve_cross_refs()는 각 노드의 metadata를
# 직접 수정하므로(in-place), 이 노드는 별도로 state를 갱신해서 반환할 필요가 없다
# (그래도 LangGraph 노드 규약을 지키기 위해 빈 딕셔너리를 반환한다).
# ============================================================
def node_resolve_cross_refs(state: GraphState) -> dict:
    logger.info("[LawChunker] resolve_cross_refs 시작 - 교차참조 2-pass 해결")
    resolve_cross_refs(
        state["parent_nodes"],
        state["article_index"],
        state["article_paragraph_index"],
        state["byeolpyo_index"],
    )
    return {}


# ============================================================
# 20) 노드 7) 마무리
#
# 그래프의 마지막 노드. 지금까지 누적된 경고들을 로그로 남기고, "이 파이프라인의
# 최종 산출물이 무엇인지"를 final_nodes / final_node_dict 라는 이름으로 확정한다.
# (LawChunker.parse_to_hierarchical_nodes()가 바로 이 두 값을 꺼내서 반환한다.)
# ============================================================
def node_finalize(state: GraphState) -> dict:
    logger.info("[LawChunker] finalize 시작 - 최종화 및 경고 로깅")
    for w in state.get("warnings", []):
        logger.warning(f"[LawChunker] {w}")
    logger.info(f"[LawChunker] 계층 노드 생성 완료 - 총 {len(state['all_nodes'])}개 (parent+child)")
    return {
        "final_nodes": state["all_nodes"],
        "final_node_dict": state["node_dict"],
        "fallback": False,
    }


# ============================================================
# 21) 그래프 조립
#
# 지금까지 만든 노드 함수들과 조건부 분기 함수들을 실제로 "연결"해서 하나의
# 실행 가능한 그래프로 컴파일하는 곳이다. add_node로 노드를 등록하고,
# add_edge/add_conditional_edges로 노드 사이의 화살표(흐름)를 정의한다.
# ============================================================
def build_law_chunker_graph():
    graph = StateGraph(GraphState)  # GraphState를 상태 스키마로 사용하는 그래프 생성

    # --- 1단계: 그래프에서 쓸 노드들을 이름과 함께 등록 ---
    # 문서 구조를 감지: 전처리된 텍스트에서 조항/조목 등 계층 구조를 판별하고
    # 이후 처리 경로(계층적 처리 또는 폴백)를 결정하는 정보를 생성한다.
    graph.add_node("detect_structure", node_detect_structure)

    # 폴백 처리 노드: 구조 감지에 실패하거나 비표준 문서인 경우 단순 청크로
    # 분할하여 처리하는 절차를 수행한다.
    graph.add_node("fallback_chunk", node_fallback_chunk)

    # 문서 준비: 각 원문을 파싱하고 필요한 메타데이터/토큰화를 수행하여
    # 병렬 처리 단위(문서 리스트)로 변환한다.
    graph.add_node("prepare_documents", node_prepare_documents)

    # 개별 문서 처리: 하나의 문서(또는 문서 단위)를 받아 청크 생성, 계층 노드
    # 생성, 교차참조 수집 등 문서 단위의 주요 처리를 수행한다.
    graph.add_node("process_document", node_process_document)

    # 문서 병합: 병렬로 처리된 각 문서의 결과를 수집하여 단일 상태로 병합하고
    # 중복 제거 및 정렬 등 후처리를 수행한다.
    graph.add_node("merge_documents", node_merge_documents)

    # 동등 항목(피어) 테이블 계산: 동일 레벨의 노드들 간의 관계표를 생성하여
    # 후속 참조 해결에 사용할 정보를 구성한다.
    graph.add_node("compute_peer_tables", node_compute_peer_tables)

    # 교차참조 해결: 문서 전반에 걸친 조문/조항 간의 참조를 찾아 실제 연결을
    # 확립하고 참조 메타데이터를 업데이트한다.
    graph.add_node("resolve_cross_refs", node_resolve_cross_refs)

    # 최종화: 모든 노드 데이터를 결합해 최종 계층 노드 리스트 및 딕셔너리를
    # 생성하고 경고 로깅 등을 수행하여 결과를 반환한다.
    graph.add_node("finalize", node_finalize)

    # --- 2단계: 노드 사이의 흐름(엣지) 정의 ---
    # START는 LangGraph가 제공하는 특별한 "그래프 진입점" 표시. 여기서 시작해서
    # 항상 detect_structure 노드가 가장 먼저 실행된다.
    graph.add_edge(START, "detect_structure")

    # 조건부 엣지 1: detect_structure가 끝난 뒤, route_after_detect 함수의 반환값
    # ("hierarchical" 또는 "fallback")에 따라 다음 노드가 갈라진다.
    # 세 번째 인자는 "함수가 반환한 문자열" → "실제 노드 이름" 매핑표다.
    graph.add_conditional_edges(
        "detect_structure",
        route_after_detect,
        {"hierarchical": "prepare_documents", "fallback": "fallback_chunk"},
    )

    # 조건부 엣지 2: prepare_documents 이후, fan_out_documents가 반환하는 Send
    # 리스트에 따라 문서 개수만큼 process_document가 병렬로 분기 실행된다.
    # 마지막 인자 ["process_document"]는 "이 조건부 엣지가 도달할 수 있는 노드 후보
    # 목록"을 LangGraph에게 미리 알려주는 용도다(그래프 구조 검증/시각화에 사용됨).
    graph.add_conditional_edges("prepare_documents", fan_out_documents, ["process_document"])

    # --- 3단계: 나머지는 병렬 분기 없이 순서대로 한 줄로 이어지는 일반 엣지 ---
    # (process_document는 여러 번 실행되지만, 그 결과가 doc_results로 다 모인
    #  뒤에는 merge_documents가 "딱 한 번만" 실행된다 — 이것이 Send의 fan-in 동작이다.)
    graph.add_edge("process_document", "merge_documents")
    graph.add_edge("merge_documents", "compute_peer_tables")
    graph.add_edge("compute_peer_tables", "resolve_cross_refs")
    graph.add_edge("resolve_cross_refs", "finalize")
    graph.add_edge("finalize", END)          # END도 LangGraph가 제공하는 특별한 "그래프 종료" 표시
    graph.add_edge("fallback_chunk", END)     # 폴백 경로도 자신만의 END로 종료

    # compile()을 호출해야 실제로 실행 가능한 객체가 된다(지금까지는 "설계도"만 만든 상태).
    return graph.compile()


# ============================================================
# 22) 외부 공개 클래스 — 기존 LawChunker와 동일한 사용법 유지
#
# 이 클래스는 위에서 만든 LangGraph 파이프라인을 "예전 LawChunker와 똑같은
# 방식으로 호출할 수 있게" 감싸주는 얇은 어댑터(wrapper)다. 즉 이 클래스를
# 쓰는 외부 코드는 내부가 LangGraph로 바뀐 것을 전혀 몰라도 된다.
# ============================================================
class LawChunker:
    """
    한국 법령/훈령류 문서(조문-항-호-목-세목 / 별표 / 부칙 구조)를 Parent-Child
    계층으로 청킹한다. 내부 파이프라인은 LangGraph StateGraph로 구성되어 있으며,
    구조 탐지 결과에 따라 계층 파싱 경로 또는 폴백 경로로 조건부 분기한다(v3).

    사용 흐름 (기존과 동일):
        chunker = LawChunker()
        stats = chunker.detect_structure(llama_docs)
        all_nodes, node_dict = chunker.parse_to_hierarchical_nodes(llama_docs)
        # → 구조가 없으면 그래프 내부에서 자동으로 폴백 경로를 타므로
        #   호출부에서 별도 분기 처리가 필요 없다.
    """

    def __init__(self, cfg: dict = None):
        # 사용자가 넘긴 cfg로 DEFAULT_CONFIG의 일부 값만 덮어쓴다(나머지는 기본값 유지).
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        # 그래프는 설정이 바뀌지 않는 한 매번 다시 만들 필요가 없으므로, 인스턴스
        # 생성 시 한 번만 컴파일해서 재사용한다(성능상 이점).
        self._graph = build_law_chunker_graph()

    def detect_structure(self, llama_docs) -> dict:
        # 그래프 전체를 실행할 필요 없이, node_detect_structure 함수 하나만 직접
        # 호출해도 같은 결과를 얻을 수 있으므로 그렇게 처리한다(가볍고 빠름).
        return node_detect_structure({"llama_docs": llama_docs})["structure_stats"]

    def parse_to_hierarchical_nodes(self, llama_docs):
        # graph.invoke(초기state)는 START부터 END까지 그래프 전체를 실행하고,
        # 최종 state(모든 노드를 거치며 누적된 값들)를 딕셔너리로 반환한다.
        final_state = self._graph.invoke({"llama_docs": llama_docs, "cfg": self.cfg})
        return final_state["final_nodes"], final_state["final_node_dict"]