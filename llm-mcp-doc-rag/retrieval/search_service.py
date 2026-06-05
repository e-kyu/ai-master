import re
from typing import List
from duckduckgo_search import DDGS
from utils import config
from llama_index.core import Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from datetime import datetime
import time


def improve_search_query(
    topic: str
) -> List[str]:
    """사용자의 질문을 바탕으로 최적화된 검색어 3개를 생성합니다."""
    
    system_prompt = (
        "당신은 검색 엔진 최적화 및 정보 검색 전문가입니다. "
        "사용자의 질문을 분석하여 가장 정확하고 풍부한 정보를 얻을 수 있는 검색어 3개를 생성하세요."
    )
    
    today_str = datetime.now().strftime("%Y년")
    user_prompt = (
        f"사용자 질문: '{topic}'\n\n"
        "위 질문에 대해 다음 전략을 사용하여 검색어를 최소 1개에서 최대 3개까지 만드세요:\n"
        "1. 핵심 키워드 및 주어 포함: 질문의 핵심 주어(대상)를 반드시 포함하고, 불필요한 조사를 뺀 명확한 단어 조합으로 구성.\n"
        f"2. 맥락 보강: 최신 정보가 필요하다면 '{today_str}' 키워드를 포함하고, 전문적인 정보가 필요하다면 관련 전문 용어를 포함.\n"
        "3. 검색 연산자 활용: 반드시 포함해야 할 단어는 큰따옴표(\"\"), 유사어는 OR 연산자, 특정 사이트 제한은 site: 연산자를 활용하여 검색 의도에 가장 부합하는 고급 검색 쿼리를 생성.\n"
        "출력 형식: 다른 설명 없이 검색어만 한 줄에 하나씩 출력."
    )

    llm = config.get_llm()
    
    # LlamaIndex LLM interface 사용
    response = llm.complete(f"{system_prompt}\n\n{user_prompt}")
    content = str(response.text)

    # 1. 번호 매기기 제거 (예: 1. 검색어 -> 검색어)
    content = re.sub(r'^\d+[\.\s\-]+', '', content, flags=re.MULTILINE)
    
    # 2. 콤마 또는 줄바꿈으로 분리 후 정제
    delimiters = [",", "\n"]
    suggested_queries = []
    
    for delimiter in delimiters:
        queries = [q.strip().strip('"\'') for q in content.split(delimiter) if q.strip()]
        if len(queries) >= 2:
            suggested_queries = queries
            break
        
    return suggested_queries[:3] if suggested_queries else [topic]


def get_search_content(
    improved_queries: str,
    language: str = "ko",
    max_results: int = 5,
) -> List[Document]:
    """
    개선된 검색어들을 사용하여 DuckDuckGo에서 웹 페이지 내용을 검색하고 가져옵니다.
    
    Args:
        improved_queries: AI가 다듬은 검색어 리스트
        language: 검색 지역 설정 (기본값 'ko')
        max_results: 검색어당 가져올 최대 결과 수
    """
    
    try:
        documents = []      # 최종 문서들을 담을 바구니
        seen_urls = set()   # 중복된 사이트 방문을 방지하기 위한 기록장
        ddgs = DDGS()       # DuckDuckGo 검색 엔진 객체 생성
        
        for query in improved_queries:
            try:
                # Rate Limit 방지를 위한 짧은 대기
                time.sleep(1)
                
                # 1. 검색 엔진에 질문을 던져 결과 목록(제목, URL, 요약문)을 받아옵니다.
                results = ddgs.text(
                    query,
                    region=language,
                    safesearch="moderate", # 유해 콘텐츠 필터링
                    timelimit="y",         # 최근 1년 이내의 최신 정보 위주
                    max_results=max_results, # 가져올 개수 제한
                )

                if not results:
                    continue

                # 2. 검색 결과에서 실제 접속 가능한 웹 주소(URL)만 추출합니다.
                for doc in results:
                    url = doc.get("href")
                    if not url or url in seen_urls:
                        continue
                    
                    seen_urls.add(url)

                    if len(documents) < 3 :
                        # 3. 추출한 URL들에 실제로 접속하여 웹 페이지의 본문 텍스트를 긁어옵니다.
                        try:
                            time.sleep(0.5) # 개별 페이지 로드 전 대기
                            loader = SimpleWebPageReader(html_to_text=True) # HTML을 깨끗한 텍스트로 변환하는 도구
                            web_docs = loader.load_data(urls=[url]) # 실제 웹 페이지 로드
                            for web_doc in web_docs:
                                web_doc.metadata.update({"query": query, "title": doc.get("title"), "url": url})
                                documents.append(web_doc)
                        except Exception as e:
                            print(f"웹 페이지 로드 중 오류 발생: {str(e)}")
                    else:
                        # json 형태의 검색 결과를 Document 객체로 변환
                        documents.append(Document(
                            text=doc.get("body", ""),
                            metadata={"title": doc.get("title"), "url": url, "query": query}
                        ))
                        
            except Exception as e:
                print(f"쿼리 '{query}' 검색 중 오류 발생: {str(e)}")
        return documents
    except Exception as e:
        print(f"검색 서비스 오류 발생: {str(e)}")
        return []
