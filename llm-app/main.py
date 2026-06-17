import json
import os
import time
import uuid
from dotenv import load_dotenv
import requests
import streamlit as st
from datetime import datetime

from components.agent import fetch_agent_list
from components.sidebar import render_sidebar
from utils.state_manager import init_session_state, reset_session_state
from ui import noticeScanAgent, ppsAssistAgent

# myUtils
from utils import fileUtils




def render_ui():
        
    if st.session_state.ui_chat_agent_mode == "PpsAssistAgent":
        # 페이지 설정
        st.set_page_config(page_title="공공조달 어시스턴트"
                        , page_icon="🤖"
                        , initial_sidebar_state="collapsed"
                        , layout="centered"
                        )

        render_sidebar()

        ppsAssistAgent.render()
            
    elif st.session_state.ui_chat_agent_mode == "NoticeScanAgent":
        # 페이지 설정
        st.set_page_config(page_title="공고서 파싱 에이전트"
                        , page_icon="🤖"
                        , initial_sidebar_state="collapsed"
                        , layout="wide"
                        )

        render_sidebar()

        noticeScanAgent.render()
    else:
        # 페이지 설정
        st.set_page_config(page_title="새로운 에이전트 선택"
                        , page_icon="🤖"
                        , initial_sidebar_state="collapsed"
                        , layout="centered"
                        )

        # 제목 및 소개
        st.markdown("""
            <div style="text-align: left; padding: 2rem 0rem;">
                <h1 style="font-size: 1.4rem; color: #333; font-weight: 400;">안녕하세요!</h1>
                <h2 style="font-size: 2.0rem; color: #111; font-weight: 600;">어떤 도움이 필요하신가요?</h2>
            </div>
        """, unsafe_allow_html=True)

        ## TODO: DB

        # Agent 그리드 구성을 위한 데이터 조회        
        agents = fetch_agent_list();

        # 2개씩 렌더링
        for i in range(0, len(agents), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j >= len(agents):
                    break
                agent = agents[i + j]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 15px;
                            border-radius: 15px;
                            border: none;
                            margin-bottom: 10px;
                            height: 180px;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                            color: white;
                        ">
                            <h3 style="margin-top: 0; color: #ffffff; font-size: 1.2rem; font-weight: 700;">{agent['name']}</h3>
                            <p style="font-size: 0.9rem; color: #e0e0e0; line-height: 1.4;">{agent['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if st.button("시작하기", key=f"btn_{agent['agent_id']}", use_container_width=True, type="secondary"):
                        # 세션 상태 업데이트 및 모드 변경
                        st.session_state.ui_chat_agent_id = agent["agent_id"]
                        st.session_state.ui_chat_mode = agent["description"]
                        st.session_state.ui_chat_agent_mode = agent["mode"]
                        st.session_state.ui_chat_agent_name = agent["name"]
                        st.session_state.ui_chat_agent_desc = agent["description"]
                        
                        st.session_state.messages = []
                        st.session_state.convrstn_id = str(uuid.uuid4())
                        st.rerun()

        st.info("원하는 에이전트의 '시작하기' 버튼을 클릭하면 대화 화면으로 이동합니다.")



if __name__ == "__main__":
    # 전체 파라미터 딕셔너리 가져오기
    params = st.query_params

    if "ui_chat_agent_mode" not in st.session_state:
        st.session_state.ui_chat_agent_mode = ""

    # 개별 파라미터 읽기
    agentMode = st.query_params.get("agentMode", st.session_state.ui_chat_agent_mode)

    if agentMode != "":

        # Agent 그리드 구성을 위한 데이터 조회        
        agents = fetch_agent_list();
        
        if agents is not None and len(agents) > 0:
            selected_agent = next((agent for agent in agents if agent["mode"] == agentMode), None)
            if selected_agent is not None:
                st.session_state.ui_chat_agent_id = selected_agent["agent_id"]
                st.session_state.ui_chat_mode = selected_agent["description"]
                st.session_state.ui_chat_agent_name = selected_agent["name"]
                st.session_state.ui_chat_agent_desc = selected_agent["description"]
            else:
                st.warning(f"지정된 agentMode '{agentMode}'에 해당하는 에이전트를 찾을 수 없습니다.")



    load_dotenv()

    # 세션 상태 초기화
    init_session_state()

    render_ui()
