

import re
import os

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode, TextNode
from utils import logger_util



class FallbackChunker:
    
    def __init__(self):
        self.logger = logger_util.get_logger("./log", "llm-mcp-doc-rag")

    # ============================================================
    # 7. 기본 Fallback (조문/별표를 전혀 찾지 못한 문서용)
    #    법령 구조가 아예 없는 일반 문서가 섞여 들어온 경우를 위한 안전망으로,
    #    부모는 긴 문맥 유지용, 자식은 벡터 검색용 작은 청크로 생성한다.
    # ============================================================
    def create_fallback_nodes(
        self,
        documents,
        parent_chunk_size=1024,
        parent_chunk_overlap=100,
        child_chunk_size=256,
        child_chunk_overlap=50,
    ):
        self.logger.warning("[LawChunker] 법률 구조를 찾지 못해 기본 SentenceSplitter로 폴백합니다.")

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

        self.logger.info(
            f"[LawChunker] 폴백 청킹 완료 - parent {len(parent_nodes)}개, 전체 노드 {len(all_nodes)}개"
        )
        return all_nodes, node_dict