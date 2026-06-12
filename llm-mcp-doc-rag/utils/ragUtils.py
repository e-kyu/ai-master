import os
import hashlib
import zipfile
import tarfile
import shutil
from utils import config, loggerUtil
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



# 파일 경로를 전달하면 LlamaIndex를 사용하여 문서를 로드하고, 벡터 인덱스를 생성하여 검색기(Retriever)를 반환한다.
def make_rag_query_engine(input_path, is_save=False, is_base_resource=False):
    if not input_path:
        return None

    today_str = datetime.now().strftime("%Y%m%d")
    is_compressed = False
    
    # 1. 파일 경로를 기반으로 고유한 ID(해시) 생성 및 저장 경로 설정
    md5_doc = hashlib.md5(input_path.encode('utf-8')).hexdigest()
    temp_extract_path = f"./temp_extracted/{md5_doc}" # 압축 해제용 임시 폴더
    save_path = f"./vectorstore/{today_str}/{md5_doc}"
    if is_base_resource:
        save_path = f"./vectorstore/base_resource/{md5_doc}"

    try:
        index = None
        node_dict = {}

        # 이미 저장된 index가 존재한다면 기존 것을 사용
        if is_save and os.path.isdir(save_path):
            logger.info(f"기존 벡터스토어를 로드합니다: {save_path}")
            vector_store = FaissVectorStore.from_persist_dir(save_path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=save_path
            )
            index = load_index_from_storage(storage_context)
            # 저장된 인덱스의 docstore에서 노드 정보를 가져와 매핑 데이터 복원
            node_dict = index.docstore.docs
        else:
            # 2. 압축 파일 여부 확인 및 해제
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
            # 폴더 또는 파일을 읽어오는 Reader 설정
            reader = SimpleDirectoryReader(
                input_dir=actual_data_path if os.path.isdir(actual_data_path) else None,
                input_files=[actual_data_path] if os.path.isfile(actual_data_path) else None,
                recursive=True
            )
            llama_docs = reader.load_data()

            # 단계 2: 문서를 큰 덩어리(부모)와 작은 덩어리(자식)로 나누어 계층 구조 생성
            logger.info("단계 2: 계층형 노드 생성 (Parent-Child)")
            parent_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            child_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

            parent_nodes = parent_splitter.get_nodes_from_documents(llama_docs)
            all_nodes = []

            for parent in parent_nodes:
                child_nodes = child_splitter.get_nodes_from_documents([parent])
                for child in child_nodes:
                    # 자식 노드가 부모 노드의 ID를 참조하도록 연결
                    i_node = IndexNode.from_text_node(child, index_id=parent.node_id)
                    all_nodes.append(i_node)
                node_dict[parent.node_id] = parent
                all_nodes.append(parent)

            # 단계 3: 검색을 위한 벡터 인덱스 생성
            logger.info("단계 3: 재귀 탐색용 인덱스 생성")
            
            # 임베딩 차원을 동적으로 파악하여 FAISS 인덱스 생성 (속도 향상을 위해 IndexFlatIP 또는 HNSW 고려 가능)
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
        # 검색 범위를 좁혀 속도 향상 (similarity_top_k 조정)
        vector_retriever = index.as_retriever(similarity_top_k=10)

        # 자식 노드를 찾으면 자동으로 부모 노드까지 찾아주는 재귀적 검색기 사용
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_dict, # 부모 노드 참조를 위해 필수
            verbose=True
        )

        # LLMRerank는 품질은 좋으나 속도가 매우 느림. 꼭 필요한 경우에만 top_n을 최소화하여 사용
        node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=0.5),
            LLMRerank(top_n=1) 
        ]

        logger.info("단계 5: Recursive Query Engine 반환")
        return RetrieverQueryEngine.from_args(
            recursive_retriever,
            node_postprocessors=node_postprocessors,
            streaming=False, # 중간 컨텍스트 생성을 위해 기본 스트리밍 비활성화
            timeout=1800
        )

    except Exception as e:
        logger.error(f"RAG Query Engine creation failed: {e}")
        raise
    
    finally:
        # 6. 임시 폴더 정리 (압축 파일이었던 경우에만)
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
            LLMRerank(top_n=1) 
        ]
        

        logger.info("단계 5: 최종 검색 엔진 준비 완료")
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
        pass