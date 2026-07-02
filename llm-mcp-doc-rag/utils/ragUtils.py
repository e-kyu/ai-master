import os
import hashlib
import zipfile
import tarfile
import shutil

from utils import config, loggerUtil
from utils.chunkingPatterns.lawChunker import LawChunker
from utils.chunkingPatterns.fallbackChunker import  FallbackChunker
from tools import convertToMarkdown
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Document, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
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


# 마크다운으로 우선 변환할 확장자. pymupdf4llm/markitdown이 표·헤더 구조를 인식해
# 변환해주므로, 원문 PDF/HTML 텍스트를 그대로 읽을 때보다 LawChunker의 조문/별표
# 탐지 정확도가 올라간다.
_MARKDOWN_CONVERTIBLE_EXTS = (".pdf", ".html", ".htm")


def _collect_file_paths(data_path):
    """단일 파일이면 그대로, 디렉토리면 하위 파일 전체를 재귀적으로 모아 반환한다."""
    if os.path.isfile(data_path):
        return [data_path]
    file_paths = []
    for root, _, names in os.walk(data_path):
        for name in names:
            file_paths.append(os.path.join(root, name))
    return file_paths


def _load_documents(data_path):
    """
    PDF/HTML(.htm) 문서는 tools/convertToMarkdown.py의 ConvertToMarkdownAgent(execute)로
    마크다운 변환한 뒤 Document로 감싸고, 그 외 형식은 기존 SimpleDirectoryReader로 로드한다.
    변환에 실패하면 해당 파일만 SimpleDirectoryReader 처리 목록으로 넘겨 원문 텍스트로라도
    로드되게 한다.
    """
    llama_docs = []
    remaining_files = []

    for file_path in _collect_file_paths(data_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in _MARKDOWN_CONVERTIBLE_EXTS:
            logger.info(f"마크다운 변환 중... ({file_path})")
            contents = convertToMarkdown.execute(file_path)
            if not contents:
                logger.warning(f"마크다운 변환 실패, 기본 리더로 대체합니다: ({file_path})")
                remaining_files.append(file_path)
                continue
            llama_docs.append(
                Document(
                    text=contents[0].text,
                    metadata={"file_name": os.path.basename(file_path), "file_path": file_path},
                )
            )
        else:
            remaining_files.append(file_path)

    if remaining_files:
        reader = SimpleDirectoryReader(input_files=remaining_files, recursive=True)
        llama_docs.extend(reader.load_data())

    return llama_docs


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
            # TODO: markdown 변환시 법률 구조 기반 청킹 정확도가 오히려 떨어짐...
            # llama_docs = _load_documents(actual_data_path)

            # 🚀 법률 맞춤형 계층 청킹을 LawChunker에 위임 (조문/항/별표/부칙 구조 기반 파싱)
            logger.info("단계 2: 법률 구조 기반 계층형 노드 생성 (Parent-Child)")

            law_chunker = LawChunker()
            fallback_chunker = FallbackChunker()

            stats = law_chunker.detect_structure(llama_docs)

            # 조문도 없고 별표도 없으면 이 문서 구조에 맞는 파싱이 불가능 -> fallback
            if not stats["has_article"] and not stats["has_byeolpyo"]:
                logger.warning("샘플 문서에서 조문/별표 구조를 찾지 못했습니다. 기본 Parent-Child 계층형 청킹을 적용합니다.")
                all_nodes, node_dict = fallback_chunker.create_fallback_nodes(llama_docs)
            else:
                all_nodes, node_dict = law_chunker.parse_to_hierarchical_nodes(llama_docs)
                if not all_nodes:
                    logger.info("법률 구조 기반 노드 생성에 실패하여 기본 계층형 청킹으로 재시도합니다.")
                    all_nodes, node_dict = fallback_chunker.create_fallback_nodes(llama_docs)

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
        vector_retriever = index.as_retriever(similarity_top_k=20)

        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_dict, 
            verbose=True
        )

        node_postprocessors = [
            #SimilarityPostprocessor(similarity_cutoff=0.2),
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