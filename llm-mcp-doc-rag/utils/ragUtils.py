import os
import hashlib
import zipfile
import tarfile
import shutil
import re

from utils import config, loggerUtil
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode, TextNode
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core import Settings as LlamaSettings

logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")

# 문서의 md5 해시값을 파라미터로 전달하면, 저장된 LlamaIndex 인덱스를 로드하여 검색기(Retriever)를 반환한다.
def getRagRetriever(md5_doc) :
    # 저장된 벡터 데이터베이스 경로 설정
    save_path = f"./vectorstore/{md5_doc}"
    if os.path.isdir(save_path):
        logger.info(f"The folder '{save_path}' exists.")
        
        # 저장된 폴더로부터 인덱스 데이터 로드
        vector_store = FaissVectorStore.from_persist_dir(save_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=save_path
        )
        index = load_index_from_storage(storage_context)
        
        logger.info("LlamaIndex Query Engine 반환")
        return index.as_query_engine(similarity_top_k=20)
    return None

def getRagPromft():
    # RAG 답변 생성을 위한 기본 프롬프트 템플릿
    return """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Answer in Korean.

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:
        """


def generate_smart_chunking_patterns(llama_docs, sample_size=10):
    """
    간단한 스마트 청킹 패턴 생성기(정규식 최적화용).
    일부 문서 페이지를 샘플로 받아서 문서에서 자주 등장하는 조문 및 항 표기 패턴을 탐지하여
    article_pattern 및 child_pattern 후보를 반환합니다.
    """
    if llama_docs:
        if len(llama_docs) <= sample_size:
            texts = [doc.text for doc in llama_docs]
        else:
            step = len(llama_docs) / sample_size
            indices = [min(int(i * step), len(llama_docs) - 1) for i in range(sample_size)]
            texts = [llama_docs[idx].text for idx in indices]
    else:
        texts = [""]

    sample = "\n".join(texts)

    # 후보 패턴들 (법률 문서에서 흔히 쓰이는 형태들을 우선 제시)
    # 국내(한국) 패턴 외에 국제적으로 자주 쓰이는 패턴(영문/라틴식, 섹션 기호 등)을 추가
    article_candidates = [
        r'(제\d+조(?:의\d+)?\(.*?\))',
        r'(제\d+조(?:의\d+)?)',
        r'(^제\s*\d+\s*조[\s\S]{0,60}?)(?=\n|$)',
        r'(제\d+조\([^\)]+\))',
        r'(\b제\d+조(?:의\d+)?\b)',
        # English / International patterns
        r'\bArticle\s+\d+[A-Za-z0-9\-]*\b',    # Article 1, Article 1-2
        r'\bArt\.\s*\d+\b',                    # Art. 1
        r'\bSection\s+\d+\b',                  # Section 1
        r'\bSec\.\s*\d+\b',                    # Sec. 1
        r'§\s*\d+[A-Za-z0-9\-]*',                # § 1, § 1-2
        r'\bChapter\s+\d+\b'                   # Chapter 1
    ]
    child_candidates = [
        r'(\([①-⑳]\)|[①-⑳])',           # circled numbers
        r'(\(\d+\)|\d+\.)',            # (1) or 1.
        r'(\([가-힣]\)|[가-힣]\.)',       # (가) or 가.
        r'(\([A-Za-z]\)|[A-Za-z]\.)',    # (a) or a.
        # International/sub-clause patterns
        r'\b\([ivxIVX]+\)\b|\b[ivxIVX]+\.',   # (iv) or iv.
        r'\b\([a-z]\)\b|\b[a-z]\.',           # (a) or a.
        r'\b\([A-Z]\)\b|\b[A-Z]\.',           # (A) or A.
        r'\b\d+\)\b',                           # 1)
        r'^[\-\*]\s+',                           # bullet lists starting with - or *
    ]

    def score_pattern(pat, text):
        try:
            return len(re.findall(pat, text))
        except re.error:
            return 0

    best_article = max(article_candidates, key=lambda p: score_pattern(p, sample))
    best_child = max(child_candidates, key=lambda p: score_pattern(p, sample))

    return {
        "article_pattern": best_article,
        "child_pattern": best_child,
        "sample_counts": {
            "article_matches": score_pattern(best_article, sample),
            "child_matches": score_pattern(best_child, sample)
        }
    }



def parse_law_to_hierarchical_nodes(llama_docs, article_pattern, child_pattern):
    """
    법률 문서를 조(Parent) 단위와 항/호(Child) 단위로 계층 분할하는 함수
    """
    parent_nodes = []
    all_nodes = []
    node_dict = {}

    # 전체 문서를 하나의 텍스트로 합치기 (페이지 분할로 조항이 끊기는 것을 방지)
    full_text = "\n".join([doc.text for doc in llama_docs])

    # 문서 메타데이터에서 법률명을 추출
    law_title = "답변자료"
    if llama_docs:
        first_meta = getattr(llama_docs[0], "metadata", {})
        if isinstance(first_meta, dict):
            fname = first_meta.get("file_name")
            if fname:
                # 파일명에서 확장자 제거
                law_title = os.path.splitext(fname)[0]

    # 컴파일하여 멀티라인/유니코드 처리를 명시적으로 수행
    try:
        article_re = re.compile(article_pattern, flags=re.MULTILINE)
    except re.error:
        article_re = re.compile(r'(제\d+조(?:의\d+)?\(.*?\))', flags=re.MULTILINE)

    try:
        child_re = re.compile(child_pattern, flags=re.MULTILINE)
    except re.error:
        child_re = re.compile(r'(\([①-⑳]\)|[①-⑳])', flags=re.MULTILINE)

    splits = re.split(article_re, full_text)
    
    # 첫 조항이 나오기 전 서론/목적 정보 처리
    # 첫 요소가 조문 제목이 아니라면 서론으로 간주
    if splits and not article_re.match(splits[0].strip()):
        intro_text = splits.pop(0).strip()
        if intro_text:
            p_node = TextNode(text=intro_text, metadata={"type": "intro"})
            parent_nodes.append(p_node)

    # 조항 제목과 본문 매칭하여 Parent Node 생성
    articles = []
    for i in range(0, len(splits), 2):
        if i + 1 < len(splits):
            title = splits[i].strip()
            content = splits[i+1].strip()
            articles.append((title, content))

    for title, content in articles:
        full_article_text = f"{title}\n{content}"
        
        # 부모 노드 생성
        parent_node = TextNode(
            text=full_article_text,
            metadata={
                "law_title": law_title,
                "article_title": title,
                "type": "parent"
            }
        )
        parent_nodes.append(parent_node)
        node_dict[parent_node.node_id] = parent_node
        all_nodes.append(parent_node)

        # 2. 부모 본문 안에서 항(①, ②, ③) 단위로 자식 노드 분할
        # 항(Child) 분할 시에도 컴파일된 정규식을 사용
        paragraphs = re.split(child_re, content)
        
        child_chunks = []
        # 첫 항 시작 전 문구(예: 조항 본문 바로 시작)가 있다면 추가
        if paragraphs and not child_re.match(paragraphs[0].strip()):
            first_text = paragraphs.pop(0).strip()
            if first_text:
                child_chunks.append(first_text)
                
        for j in range(0, len(paragraphs), 2):
            if j + 1 < len(paragraphs):
                p_num = paragraphs[j].strip()
                p_text = paragraphs[j+1].strip()
                child_chunks.append(f"{p_num} {p_text}")

        # 자식 노드들을 IndexNode로 변환하여 부모 ID와 연결
        for chunk in child_chunks:
            if not chunk.strip():
                continue
            
            # 자식 검색 시 상위 맥락 유실을 방지하기 위해 '법률명 + 조항제목'을 Prefix로 주입
            contextualized_text = f"법률명: {law_title}\n조항: {title}\n내용: {chunk}"
            
            child_node = TextNode(
                text=contextualized_text,
                metadata={
                    "law_title": law_title,
                    "article_title": title,
                    "type": "child"
                }
            )
            # IndexNode를 통해 부모의 node_id를 가리키도록 설정 (재귀 탐색의 핵심)
            i_node = IndexNode.from_text_node(child_node, index_id=parent_node.node_id)
            all_nodes.append(i_node)

    return all_nodes, node_dict


def create_hierarchical_nodes_with_splitters(
    documents,
    parent_chunk_size=1024,
    parent_chunk_overlap=100,
    child_chunk_size=256,
    child_chunk_overlap=50,
):
    """
    기본 Parent-Child 계층형 청킹 구현.
    부모는 긴 문맥 유지용, 자식은 벡터 검색용 작은 청크로 생성합니다.
    """
    parent_splitter = SentenceSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )
    child_splitter = SentenceSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_nodes = parent_splitter.get_nodes_from_documents(documents)
    all_nodes = []
    node_dict = {}

    for parent_node in parent_nodes:
        child_nodes = child_splitter.get_nodes_from_documents([parent_node])
        for child_node in child_nodes:
            if not child_node.text.strip():
                continue
            indexed_child = IndexNode.from_text_node(child_node, index_id=parent_node.node_id)
            all_nodes.append(indexed_child)

        node_dict[parent_node.node_id] = parent_node
        all_nodes.append(parent_node)

    return all_nodes, node_dict

# 작성해주신 기존 엔진 생성 함수에 연동
def make_rag_query_engine(input_path, is_save=False, is_base_resource=False):
    if not input_path:
        return None

    today_str = datetime.now().strftime("%Y%m%d")
    is_compressed = False
    
    md5_doc = hashlib.md5(input_path.encode('utf-8')).hexdigest()
    temp_extract_path = f"./temp_extracted/{md5_doc}"
    save_path = f"./vectorstore/{today_str}/{md5_doc}"
    if is_base_resource:
        save_path = f"./vectorstore/base_resource/{md5_doc}"

    try:
        index = None
        node_dict = {}

        if is_save and os.path.isdir(save_path):
            logger.info(f"기존 벡터스토어를 로드합니다: {save_path}")
            vector_store = FaissVectorStore.from_persist_dir(save_path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=save_path
            )
            index = load_index_from_storage(storage_context)
            node_dict = index.docstore.docs
        else:
            actual_data_path = input_path
            if input_path.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
                logger.info(f"단계 0: 압축 파일 감지 - 해제 중... ({input_path})")
                os.makedirs(temp_extract_path, exist_ok=True)
                if input_path.endswith('.zip'):
                    with zipfile.ZipFile(input_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_path)
                elif input_path.endswith(('.tar', '.tar.gz', '.tgz')):
                    with tarfile.open(input_path, 'r:*') as tar_ref:
                        tar_ref.extractall(temp_extract_path)
                actual_data_path = temp_extract_path
                is_compressed = True

            logger.info(f"단계 1: 문서 로드 중... ({actual_data_path})")
            reader = SimpleDirectoryReader(
                input_dir=actual_data_path if os.path.isdir(actual_data_path) else None,
                input_files=[actual_data_path] if os.path.isfile(actual_data_path) else None,
                recursive=True
            )
            llama_docs = reader.load_data()

            # 🚀 [변경 포인트] 기존 SentenceSplitter 대신 법률 맞춤형 정적 분할 함수 호출
            logger.info("단계 2: 법률 구조 기반 계층형 노드 생성 (Parent-Child)")

            # 1. 샘플 문서를 기반으로 유효한 조문/항 패턴을 동적으로 탐지
            patterns = generate_smart_chunking_patterns(llama_docs)

            if patterns.get("sample_counts", {}).get("article_matches", 0) < 1 or patterns.get("sample_counts", {}).get("child_matches", 0) < 1:
                logger.warning("샘플 문서에서 조문 패턴을 찾지 못했습니다. 기본 Parent-Child 계층형 청킹을 적용합니다.")
                all_nodes, node_dict = create_hierarchical_nodes_with_splitters(llama_docs)
            else:
                article_pattern = patterns.get("article_pattern", r'(제\d+조(?:의\d+)?\(.*?\))')
                child_pattern = patterns.get("child_pattern", r'(\([①-⑳]\)|[①-⑳])')
                all_nodes, node_dict = parse_law_to_hierarchical_nodes(llama_docs, article_pattern, child_pattern)
                if not all_nodes:
                    logger.info("법률 구조 기반 노드 생성에 실패하여 기본 계층형 청킹으로 재시도합니다.")
                    all_nodes, node_dict = create_hierarchical_nodes_with_splitters(llama_docs)

            # 단계 3: 검색을 위한 벡터 인덱스 생성
            logger.info("단계 3: 재귀 탐색용 인덱스 생성")
            sample_embedding = LlamaSettings.embed_model.get_text_embedding("sample")
            d = len(sample_embedding)
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex(all_nodes, storage_context=storage_context)

            if is_save:
                logger.info(f"단계 4: 벡터스토어 저장 ({save_path})")
                index.storage_context.persist(persist_dir=save_path)

        # 5. 검색기(Retriever) 및 엔진 구성
        vector_retriever = index.as_retriever(similarity_top_k=10)

        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_dict, 
            verbose=True
        )

        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5),
            LLMRerank(top_n=2)  # 법률 선후관계 파악을 위해 top_n을 2 정도로 소폭 상향 조정 권장
        ]

        logger.info("단계 5: Recursive Query Engine 반환")
        return RetrieverQueryEngine.from_args(
            recursive_retriever,
            node_postprocessors=node_postprocessors,
            streaming=False,
            timeout=1800
        )

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    finally:
        if is_compressed and os.path.exists(temp_extract_path):
            logger.info(f"단계 6: 임시 파일 정리 중... ({temp_extract_path})")
            shutil.rmtree(temp_extract_path)



def makeRagRetrieverFromDocs(docs, bSave=False):
    """
    웹 검색 결과나 리스트 형태의 문서들로부터 질문에 답변할 수 있는 검색 엔진을 만듭니다.
    """
    if not docs:
        return None

    today_str = datetime.now().strftime("%Y%m%d")

    # 1. 문서 내용의 앞부분을 이용해 고유한 ID(해시)를 생성하여 저장 폴더명을 결정합니다.
    content_to_hash = "".join([getattr(doc, 'page_content', getattr(doc, 'text', ''))[:100] for doc in docs[:5]])
    md5_doc = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
    save_path = f"./vectorstore/{today_str}/{md5_doc}"
    
    try:
        index = None
        node_dict = {}

        # 이미 분석해서 저장해둔 데이터가 있다면 다시 계산하지 않고 불러옵니다.
        if bSave and os.path.isdir(save_path):
            logger.info(f"기존 벡터스토어를 로드합니다: {save_path}")
            vector_store = FaissVectorStore.from_persist_dir(save_path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=save_path
            )
            index = load_index_from_storage(storage_context)
            node_dict = index.docstore.docs
        else:
            # 2. 전달받은 문서들을 시스템이 이해할 수 있는 표준 형식으로 변환합니다.
            llama_docs = []
            for doc in docs:
                content = getattr(doc, 'page_content', getattr(doc, 'text', ''))
                metadata = getattr(doc, 'metadata', {})
                llama_docs.append(Document(text=content, metadata=metadata))

            # 3. 문서를 효율적으로 찾기 위해 큰 덩어리(부모)와 작은 덩어리(자식)로 쪼갭니다.
            logger.info("단계 2: 문서를 작은 조각으로 나누는 중 (계층 구조 생성)")
            parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            child_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

            parent_nodes = parent_splitter.get_nodes_from_documents(llama_docs)
            all_nodes = []

            # 작은 조각(자식)을 검색하면 관련된 큰 문맥(부모)을 함께 찾을 수 있도록 연결합니다.
            for parent in parent_nodes:
                child_nodes = child_splitter.get_nodes_from_documents([parent])
                for child in child_nodes:
                    i_node = IndexNode.from_text_node(child, index_id=parent.node_id)
                    all_nodes.append(i_node)
                node_dict[parent.node_id] = parent
                all_nodes.append(parent)

            # 4. 컴퓨터가 빠르게 검색할 수 있도록 수학적 수치(벡터)로 변환하여 저장소를 만듭니다.
            logger.info("단계 3: 빠른 검색을 위한 인덱스 생성 중")
            
            sample_embedding = LlamaSettings.embed_model.get_text_embedding("sample")
            d = len(sample_embedding)
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex(all_nodes, storage_context=storage_context)

            if bSave:
                logger.info(f"단계 4: 생성된 데이터를 파일로 저장 중 ({save_path})")
                index.storage_context.persist(persist_dir=save_path)

        # 5. 질문에 가장 적합한 문서 조각을 찾아내는 검색기를 설정합니다.
        vector_retriever = index.as_retriever(similarity_top_k=10)

        # 작은 조각을 찾았을 때 자동으로 주변 문맥(부모)까지 가져오는 똑똑한 검색 방식을 사용합니다.
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_dict,
            verbose=True
        )

        # 검색된 결과 중 연관성이 낮은 것은 버리고, 가장 정답에 가까운 순서로 다시 정렬합니다.
        # TODO: FlagEmbeddingReranker를 사용하여 검색속도를 상승시키는 것이 유의미 한지 검증 필요. LLMRerank는 품질은 좋으나 속도가 매우 느림. 꼭 필요한 경우에만 top_n을 최소화하여 사용
        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5),
            LLMRerank(top_n=2) 
        ]
        

        logger.info("단계 5: 최종 검색 엔진 준비 완료")
        return RetrieverQueryEngine.from_args(
            recursive_retriever,
            node_postprocessors=node_postprocessors,
            streaming=False,
            timeout=600
        )

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    
    finally:
        pass