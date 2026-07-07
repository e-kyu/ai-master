import os
import hashlib
import zipfile
import tarfile
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pydantic import PrivateAttr

from utils import loggerUtil
from utils.chunkingPatterns.lawChunker import LawChunker
from utils.chunkingPatterns.fallbackChunker import FallbackChunker
from tools import convertToMarkdown

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever, BaseRetriever
from llama_index.core.schema import IndexNode, BaseNode
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core import Settings as LlamaSettings

# ==============================================================================
# [설정 상수 정의]
# ==============================================================================
# 파일(문서) 한 건당 독립 인덱스에서 검색해 올 최대 청크 수
FILE_SIMILARITY_TOP_K: int = 10

# 리랭킹 최종 반환 개수의 하한값 정의 (필요 시 확장 가능)
RERANK_TOP_N_MIN: int = 2

# 코사인 유사도 기준 필터링 임계값 (정규화된 벡터의 내적 결과 기준)
SIMILARITY_CUTOFF: float = 0.8

# 확장자 및 무시 대상 정의
_IGNORED_DIR_NAMES: set = {"__MACOSX"}

logger = loggerUtil.get_logger("./log", "llm-mcp-doc-rag")


# ==============================================================================
# [코어 컴포넌트: 임베딩 래퍼 & 멀티 문서 검색기]
# ==============================================================================
class MultiDocRetriever(BaseRetriever):
    """
    [문서 단위 독립 검색기 병합 레이어]
    각 파일별로 격리된 FAISS 인덱스로부터 상위 K개의 문맥을 독립적으로 추출한 후 병합합니다.
    특정 대용량 문서가 전체 상위 결과(Top-K)를 독점하는 현상을 방지합니다.
    """
    def __init__(self, retrievers: List[BaseRetriever], callback_manager: Optional[Any] = None) -> None:
        self._retrievers = retrievers
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: Any) -> List[Any]:
        results = []
        for retriever in self._retrievers:
            results.extend(retriever.retrieve(query_bundle))
        return results


class NormalizedEmbedding(BaseEmbedding):
    """
    [L2 정규화 임베딩 래퍼]
    기존 임베딩 모델의 모든 연산 결과(동기/비동기 단일 및 배치)를 L2 정규화(Unit Vector)합니다.
    FAISS의 IndexFlatIP(내적)와 결합하여 추가 연산 없이 완전한 코사인 유사도를 구현합니다.
    """
    _inner_embed_model: BaseEmbedding = PrivateAttr()

    def __init__(self, inner_embed_model: BaseEmbedding, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._inner_embed_model = inner_embed_model

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        arr = np.array(vec, dtype="float32")
        norm = np.linalg.norm(arr)
        return (arr / norm).tolist() if norm > 0 else arr.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._normalize(self._inner_embed_model._get_query_embedding(query))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._normalize(await self._inner_embed_model._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._normalize(self._inner_embed_model._get_text_embedding(text))

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._normalize(await self._inner_embed_model._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._normalize(v) for v in self._inner_embed_model._get_text_embeddings(texts)]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        raw = await self._inner_embed_model._aget_text_embeddings(texts)
        return [self._normalize(v) for v in raw]


# ==============================================================================
# [내부 헬퍼 함수]
# ==============================================================================
def _ensure_normalized_embedding() -> None:
    """글로벌 LlamaIndex 환경의 임베딩 모델이 정규화 래퍼로 래핑되어 있는지 보장합니다."""
    if not isinstance(LlamaSettings.embed_model, NormalizedEmbedding):
        logger.info("전역 임베딩 모델을 NormalizedEmbedding 래퍼로 교체합니다.")
        LlamaSettings.embed_model = NormalizedEmbedding(inner_embed_model=LlamaSettings.embed_model)


def _build_faiss_index(nodes: List[BaseNode], persist_dir: Optional[str] = None) -> VectorStoreIndex:
    """정규화 임베딩 모델과 IndexFlatIP(내적)기반의 FAISS VectorStore 인덱스를 빌드 및 유지합니다."""
    _ensure_normalized_embedding()

    sample_embedding = LlamaSettings.embed_model.get_text_embedding("sample")
    dimension = len(sample_embedding)

    # 벡터 정규화 상태이므로 내적(IP) 연산이 코사인 유사도와 동치입니다.
    faiss_index = faiss.IndexFlatIP(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=LlamaSettings.embed_model,
    )

    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)

    return index


def _load_faiss_index(persist_dir: str) -> VectorStoreIndex:
    """로컬 저장소 경로로부터 FAISS VectorStore 인덱스를 안전하게 복원합니다."""
    _ensure_normalized_embedding()
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    return load_index_from_storage(storage_context, embed_model=LlamaSettings.embed_model)


def _vectorstore_root(md5_doc: str, is_base_resource: bool) -> str:
    """입력 소스의 MD5 고유 해시값을 기반으로 표준화된 영구 벡터 데이터베이스 루트 경로를 반환합니다."""
    if is_base_resource:
        return f"./vectorstore/base_resource/{md5_doc}"
    return f"./vectorstore/{md5_doc}"


def _collect_file_paths(data_path: str) -> List[str]:
    """단일 파일 혹은 디렉토리 트리 내에서 정형/숨김 폴더를 필터링하여 유효한 파일 경로셋을 재귀 수집합니다."""
    if os.path.isfile(data_path):
        return [data_path]
        
    file_paths = []
    for root, dirs, names in os.walk(data_path):
        # 디렉토리 인플레이스(in-place) 필터링으로 불필요 트래버설 차단
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in _IGNORED_DIR_NAMES]
        for name in names:
            if name.startswith('.'):
                continue
            file_paths.append(os.path.join(root, name))
    return file_paths


def _load_documents(data_path: str) -> List[Document]:
    """
    특정 원본 파일들을 Document 객체 레이어로 마이그레이션합니다.
    PDF/HTML의 경우 마크다운 파서를 실행한 뒤 파편화된 모든 텍스트 청크를 누수 없이 단일 컨텍스트로 병합합니다.
    """
    llama_docs: List[Document] = []
    remaining_files: List[str] = []

    for file_path in _collect_file_paths(data_path):
        remaining_files.append(file_path)

    # 마크다운 비대상 타 포맷 문서 일괄 로드 백업 프로세스
    if remaining_files:
        reader = SimpleDirectoryReader(input_files=remaining_files, recursive=True)
        llama_docs.extend(reader.load_data())

    return llama_docs


def _build_file_nodes(file_docs: List[Document], law_chunker: LawChunker, fallback_chunker: FallbackChunker) -> Tuple[List[BaseNode], Dict[str, BaseNode]]:
    """문서 소스의 형태를 진단하여 법률 조문/별표 파서 기반 계층 노드 혹은 일반 폴백 계층 노드를 분기 빌드합니다."""
    stats = law_chunker.detect_structure(file_docs)

    if not stats["has_article"] and not stats["has_byeolpyo"]:
        logger.warning("특이 구조 미탐지(일반 텍스트). 기본 Parent-Child 계층 청킹 파이프라인으로 전환합니다.")
        return fallback_chunker.create_fallback_nodes(file_docs)

    nodes, node_dict = law_chunker.parse_to_hierarchical_nodes(file_docs)
    if not nodes:
        logger.info("구조화 파싱 실패, 세이프티 가드로 기본 계층 청킹을 재적용합니다.")
        return fallback_chunker.create_fallback_nodes(file_docs)
        
    return nodes, node_dict


def _extract_archive(input_path: str, temp_extract_path: str) -> str:
    """압축 아카이브 포맷(.zip, .tar, .tgz 등)을 지정 임시 경로에 압축 해제합니다."""
    os.makedirs(temp_extract_path, exist_ok=True)
    if input_path.endswith('.zip'):
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
    else:
        with tarfile.open(input_path, 'r:*') as tar_ref:
            tar_ref.extractall(temp_extract_path)
    return temp_extract_path


def _load_file_retrievers(files_save_root: str) -> Tuple[List[BaseRetriever], Dict[str, BaseNode]]:
    """디렉토리 내 저장되어 있는 파일별 로컬 FAISS DB 세트들을 일괄 수집하여 병합 검색 인터페이스 데이터셋을 재구성합니다."""
    file_retrievers: List[BaseRetriever] = []
    combined_node_dict: Dict[str, BaseNode] = {}

    for entry in sorted(os.listdir(files_save_root)):
        file_persist_dir = os.path.join(files_save_root, entry)
        if not os.path.isdir(file_persist_dir):
            continue
        try:
            index = _load_faiss_index(file_persist_dir)
        except Exception as e:
            logger.warning(f"로컬 파일 벡터 인덱스 오염 또는 로드 실패, 격리 스킵: ({file_persist_dir}) - {e}")
            continue
            
        combined_node_dict.update(index.docstore.docs)
        file_retrievers.append(index.as_retriever(similarity_top_k=FILE_SIMILARITY_TOP_K))

    return file_retrievers, combined_node_dict


def _build_query_engine(file_retrievers: List[BaseRetriever], combined_node_dict: Dict[str, BaseNode], timeout: int = 1800) -> Optional[RetrieverQueryEngine]:
    """멀티 도큐먼트 리트리버 세트와 계층형 재귀(Recursive) 매핑 딕셔너리를 활용해 지포스 포스트프로세서 믹스인 질의 엔진을 조립합니다."""
    if not file_retrievers:
        return None

    vector_retriever = MultiDocRetriever(file_retrievers)

    # 자식 노드 검색 시 부모 노드 컨텍스트를 스와핑하여 참조 범위를 확장하는 핵심 리트리버
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        node_dict=combined_node_dict,
        verbose=True
    )

    # 코사인 임계값 필터링 수행 및 후속 법률 선후 구조 인지를 위한 미니멀 Rerank 설계 적용
    node_postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF),
        LLMRerank(top_n=RERANK_TOP_N_MIN)
    ]

    return RetrieverQueryEngine.from_args(
        recursive_retriever,
        node_postprocessors=node_postprocessors,
        streaming=False,
        timeout=timeout
    )


# ==============================================================================
# [공개 비즈니스 인터페이스 API]
# ==============================================================================
def getRagRetriever(md5_doc: str, is_base_resource: bool = False) -> Optional[RetrieverQueryEngine]:
    """
    [저장 파일 기반 RAG 엔진 복원 API]
    문서의 MD5 해시 식별자 경로에 매핑 보관 중인 다중 파일 독립 FAISS 인덱스들을 로드하여 대등한 재귀식 쿼리 엔진 인터페이스를 반환합니다.
    """
    save_path = _vectorstore_root(md5_doc, is_base_resource)
    files_save_root = os.path.join(save_path, "files")

    if not os.path.isdir(files_save_root):
        logger.info(f"캐싱된 활성 벡터스토어가 탐지되지 않음: {files_save_root}")
        return None

    logger.info(f"유효 영구 벡터스토어를 검출했습니다, 세그먼트 로드 개시: {files_save_root}")
    file_retrievers, combined_node_dict = _load_file_retrievers(files_save_root)
    
    if not file_retrievers:
        logger.warning("로컬 파일 디렉토리 내에 파싱 가능한 유효 유닛 인덱스가 부재합니다.")
        return None

    logger.info("성공적으로 복원된 인덱스 셋 기반 LlamaIndex Recursive Query Engine 컴파일 완료")
    return _build_query_engine(file_retrievers, combined_node_dict)


def make_rag_query_engine(input_path: str, is_save: bool = False, is_base_resource: bool = False) -> Optional[RetrieverQueryEngine]:
    """
    [로컬 자원 기반 RAG 엔진 신규 빌드/로드 인터페이스]
    단일/다중 파일 또는 압축 아카이브 자원을 순회 분석하여 전용 파일별 인덱스 클러스터를 형성하고 상호 유기적인 재귀 RAG 파이프라인을 최종 빌드합니다.
    """
    if not input_path:
        return None

    is_compressed = input_path.endswith(('.zip', '.tar', '.tar.gz', '.tgz'))
    md5_doc = hashlib.md5(input_path.encode('utf-8')).hexdigest()
    
    temp_extract_path = f"./temp_extracted/{md5_doc}"
    save_path = _vectorstore_root(md5_doc, is_base_resource)
    files_save_root = os.path.join(save_path, "files")

    try:
        # 캐싱 레이어 검사 및 패스트 로드 분기
        if is_save and os.path.isdir(files_save_root):
            logger.info(f"저장된 히스토리 벡터 스토리지가 감지되어 콜백 복원을 수행합니다: {files_save_root}")
            file_retrievers, combined_node_dict = _load_file_retrievers(files_save_root)
        else:
            actual_data_path = input_path
            if is_compressed:
                logger.info(f"압축 컨테이너 감지 - 데이터 스테이징 추출을 실행합니다: ({input_path})")
                actual_data_path = _extract_archive(input_path, temp_extract_path)

            file_paths = _collect_file_paths(actual_data_path)
            logger.info(f"수집 타겟 도큐먼트 [{len(file_paths)}개] 개별 분석 청킹 시퀀스 가동")

            law_chunker = LawChunker()
            fallback_chunker = FallbackChunker()

            file_retrievers = []
            combined_node_dict = {}

            for file_path in file_paths:
                logger.info(f"파일 섭취 파이프라인 트리거: ({file_path})")
                try:
                    file_docs = _load_documents(file_path)
                except Exception as e:
                    logger.warning(f"문서 인메모리 로드 전처리 실패, 격리 후 진행: ({file_path}) - {e}")
                    continue

                if not file_docs:
                    continue

                # 개별 파일 수준 고장 격리를 위한 마이크로 트라이-캐치 아키텍처
                try:
                    nodes, node_dict = _build_file_nodes(file_docs, law_chunker, fallback_chunker)
                except Exception as e:
                    logger.warning(f"구조 분석 파서 치명적 청킹 오류 발생, 해당 개체 스킵: ({file_path}) - {e}")
                    continue

                if not nodes:
                    continue

                file_name = os.path.basename(file_path)
                for node in nodes:
                    node.metadata.setdefault("file_name", file_name)

                # 개별 서브 파일용 저장소 세그먼트 생성 설정 
                file_persist_dir = None
                if is_save:
                    file_md5 = hashlib.md5(file_path.encode('utf-8')).hexdigest()
                    file_persist_dir = os.path.join(files_save_root, file_md5)

                logger.info(f"파일 개별 임베딩 벡터 FlatIP FAISS 인덱스 컴파일 중 ({file_name}, 노드 수: {len(nodes)}개)")
                try:
                    index = _build_faiss_index(nodes, persist_dir=file_persist_dir)
                except Exception as e:
                    logger.warning(f"벡터 공간 매핑 빌드 실패, 해당 유닛 폐기: ({file_path}) - {e}")
                    continue

                combined_node_dict.update(node_dict)
                file_retrievers.append(index.as_retriever(similarity_top_k=FILE_SIMILARITY_TOP_K))

        return _build_query_engine(file_retrievers, combined_node_dict, timeout=1800)

    except Exception as e:
        logger.error(f"컨텍스트 빌더 엔진 팩토리 내에서 런타임 오류 차단 실패: {e}")
        raise
    finally:
        # 안전한 가비지 컬렉션 디렉토리 핸들링 보장
        if is_compressed and os.path.exists(temp_extract_path):
            logger.info(f"가상 해제 샌드박스 내부 스테이지 임시 가비지 클리닝 처리: ({temp_extract_path})")
            shutil.rmtree(temp_extract_path, ignore_errors=True)


def make_rag_query_engine_from_docs(docs: List[Any], is_save: bool = False) -> Optional[RetrieverQueryEngine]:
    """
    [외부 데이터 메모리 주입형 RAG 엔진 생성 API]
    웹 크롤링 데이터 피드나 임의 객체 리스트 소스를 구조화된 Parent-Child 임베딩 인덱스 레이어로 수용하는 정형 질의 엔진입니다.
    """
    if not docs:
        return None

    content_to_hash = "".join([getattr(doc, 'page_content', getattr(doc, 'text', ''))[:100] for doc in docs[:5]])
    md5_doc = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
    save_path = f"./vectorstore/web_docs/{md5_doc}"

    try:
        # 캐싱 레이어 검사 및 패스트 로드 분기
        if is_save and os.path.isdir(save_path):
            logger.info(f"기존 외부 피드 정형 벡터스토어를 로드합니다: {save_path}")
            index = _load_faiss_index(save_path)
            node_dict = index.docstore.docs
        else:
            llama_docs = []
            for doc in docs:
                content = getattr(doc, 'page_content', getattr(doc, 'text', ''))
                metadata = getattr(doc, 'metadata', {})
                llama_docs.append(Document(text=content, metadata=metadata))

            logger.info("단계 1: 입력 메모리 피드 기본 토큰 단위 청크 슬라이싱 계층화 개시")
            parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            child_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

            parent_nodes = parent_splitter.get_nodes_from_documents(llama_docs)
            all_nodes: List[BaseNode] = []
            node_dict: Dict[str, BaseNode] = {}

            for parent in parent_nodes:
                child_nodes = child_splitter.get_nodes_from_documents([parent])
                for child in child_nodes:
                    index_node = IndexNode.from_text_node(child, index_id=parent.node_id)
                    all_nodes.append(index_node)
                node_dict[parent.node_id] = parent
                all_nodes.append(parent)

            logger.info("단계 2: 임시 인메모리 피드 전용 코사인 매핑 벡터 스토리지 생성 실행")
            persist_dir = save_path if is_save else None
            index = _build_faiss_index(all_nodes, persist_dir=persist_dir)

        # 쿼리 엔진 구성 (MultiDoc과 유사한 위계 확보를 위한 검색 탑K 할당)
        vector_retriever = index.as_retriever(similarity_top_k=20)

        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_dict,
            verbose=True
        )

        node_postprocessors = [
            LLMRerank(top_n=2)
        ]

        logger.info("단계 3: 메모리 기반 외부 지식 소스 결합형 쿼리 엔진 구축 완료")
        return RetrieverQueryEngine.from_args(
            recursive_retriever,
            node_postprocessors=node_postprocessors,
            streaming=False,
            timeout=600
        )

    except Exception as e:
        logger.error(f"메모리 RAG 엔진 팩토리 라인 장애 발생: {e}")
        raise


def getRagPromft() -> str:
    """RAG 프롬프트 컨텍스트 템플릿 스트링을 반환합니다."""
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