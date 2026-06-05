import streamlit as st

from typing import Dict, Any

from components.convrstnHistory import fetch_convrstn_history, fetch_convrstn_by_convrstn_id, delete_convrstn_by_convrstn_id

# myUtils
from utils import fileUtils

def render_convrstn_form():
    """
    Conversation (Question and Answer) 시스템과 관련된 설정을 제공하는 사용자 인터페이스를 렌더링합니다.
    이 함수는 스트림리트 컴포넌트의 일부로, Conversation 시스템의 동작 방식을 사용자에게 설정할 수 있는 필드를 제공합니다.
    예를 들어, 모델, 프롬프트, 온도 등의 설정을 변경할 수 있습니다.
    """

    if "ui_chat_mode" not in st.session_state:
        st.session_state.update({"ui_chat_mode": "AgentView"})

    # 새로운 대화 시작 버튼
    if st.button("➕ 새로운 대화 시작", use_container_width=True, type="secondary"):
        # 세션 상태 초기화
        st.session_state.update({"app_mode": "AgentView"})
        
        st.rerun()  # 페이지를 다시 실행하여 Conversation 뷰를 보여줌




    # RAG를 선택했을 경우에만 활성화 되는 컴포넌트
    
    if st.session_state.ui_chat_mode == "AgentView":
        # 앱 소개
        ""
    elif st.session_state.ui_chat_mode == "DeepResearch":
        # 앱 소개
        st.markdown(
            """
            - 이 애플리케이션은 *사용자*의 질의를 심층 리서치 합니다.
            """
        )
    else:
        # 앱 소개
        st.markdown(
            """
            - 이 Agent는 *사용자가* 업로드한 첨부파일을 분석하여 답변을 제공합니다.
            """
        )

        # RAG 외부 문서 검색 기능 활성화 옵션
        st.checkbox(
            "외부 문서 검색 허용",
            value=False,  # 기본값은 False (외부 문서 검색 비활성화)
            help="체크한경우 첨부문서가 아닌 외부 문서을 검색하여 답변을 처리합니다. 첨부문서 내에서만 답변을 원하는 경우 체크를 해제하세요.",
            key="ui_enable_ext_docse",
        )



    # Conversation 이력을 표시하는 섹션
    convrstn_history = fetch_convrstn_history()    # 대화 이력 조회
    st.markdown("### 채팅")
    for convrstn_id, topic, date, agent_id, mode, name, description in convrstn_history:
        # 날짜 포맷팅 (YYYY-MM-DD HH:mm)
        display_date = date[:16] if date else ""

        display_topic = f"{topic[:20]} ···" if len(topic) > 20 else topic
        
        with st.expander(f"{display_topic}", expanded=False):
            
            # 정보 표시
            st.markdown(f"**{topic}**")
            st.markdown(f"<div style='text-align: right; color: #888888; font-size: 0.9rem;'>{name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: right; color: #bbbbbb; font-size: 0.8rem;'>{display_date}</div>", unsafe_allow_html=True)

            # 버튼 레이아웃
            col1, col2 = st.columns(2)

            # 보기 버튼
            with col1:
                # 보기 버튼 클릭 시 실행될 함수 정의
                if st.button("💬 이어하기", key=f"view_{convrstn_id}", use_container_width=True, type="secondary"):
                    # 세션 상태 초기화
                    st.session_state.update({"question": ""})

                    # 첨부파일을 초기화 한다. 
                    st.session_state.uploaded_files = []

                    # Conversation ID에 해당하는 Conversation 목록 조회
                    convrstnDetails = fetch_convrstn_by_convrstn_id(convrstn_id)
                    
                    # 세션 상태 메시지 목록 초기화 
                    st.session_state.update({"app_mode": "CONVERSATION"})
                    st.session_state.update({"convrstn_id": convrstn_id})
                    st.session_state.update({"messages": convrstnDetails})

                    st.session_state.update({"ui_chat_agent_id": agent_id})
                    st.session_state.update({"ui_chat_agent_mode": mode})
                    st.session_state.update({"ui_chat_agent_name": name})
                    st.session_state.update({"ui_chat_agent_desc": description})
                    
                    st.rerun()  # 페이지를 다시 실행하여 Conversation 뷰를 보여줌
                    
            # 삭제 버튼
            with col2:
                if st.button("🗑️ 삭제", key=f"del_{convrstn_id}", use_container_width=True, type="primary"):
                    if delete_convrstn_by_convrstn_id(convrstn_id):
                        # Conversation 삭제 성공 시 세션 상태를 초기화하고 페이지를 다시 실행
                        st.session_state.messages = []
                        st.session_state.update({"app_mode": "AgentView"})
                        
                        st.rerun()  # 페이지를 다시 실행하여 Conversation 뷰를 보여줌
                    

def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        render_convrstn_form()
