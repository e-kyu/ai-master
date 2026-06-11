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

# myUtils
from utils import fileUtils

####################################################################################
### MCP를 호출하는 CONVERSATION 서버의 API를 호출하고 응답을 화면에 출력한다.                    ###
####################################################################################
def startConvrstn():

    # --- 세션 상태 초기화 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.convrstn_id = str(uuid.uuid4())
        
    if "convrstn_id" not in st.session_state:
        st.session_state.convrstn_id = str(uuid.uuid4())

    # --- 세션 상태 초기화 ---
    if len(st.session_state.messages) > 0 and "question" not in st.session_state.messages[0]:
        st.session_state.messages = []
        
    for conversation in st.session_state.messages:
        st.chat_message('user').container(border=True).write(conversation['question'])
        st.markdown(conversation['answer'])

    if "question" in st.session_state and st.session_state.question != "":
        conversation = {"question": st.session_state.question, "answer": ""}
        st.chat_message('user').container(border=True).write(conversation['question'])
        st.session_state.question = None;
    else:
        return


    # 포트 충돌 방지를 위해 환경변수 사용
    API_BASE_URL = os.getenv("API_BASE_URL")

    if conversation['question'] is not None:

        # API 요청 데이터
        data = {
            "agent_id": st.session_state.get("ui_chat_agent_id", ""),
            "agent_mode": st.session_state.get("ui_chat_agent_mode", ""),
            "convrstnId": st.session_state.get("convrstn_id", ""),
            "fileFullPath": st.session_state.get("fileFullPath", ""),
            "question": conversation['question'],
            "enableExtDocse": st.session_state.get("ui_enable_ext_docse", False),
        }
        
        try:

            # STREAM(타이핑) 효과를 위한 빈 공간 생성
            response_placeholder = st.empty()
            full_response = ""
            
            with st.status("🔍 대화 맥락을 분석하고 도구를 준비 중입니다...", state="running") as status:
                # FastAPI 백엔드로 스트리밍 요청
                with requests.post(f"{API_BASE_URL}/convrstn/question", json=data, headers={"Content-Type": "application/json"}, stream=True) as r:
                    status.update(label="⚙️ 최적의 답변을 생성하는 중입니다...", state="running")
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            if not full_response:
                                # 첫 번째 청크가 들어오면 상태 업데이트
                                status.update(label="✍️ 답변을 작성합니다.", state="running")
                            
                            full_response += chunk
                            # 커서 효과와 함께 실시간 업데이트
                            
                            response_placeholder.markdown(full_response + "▌")
                
                status.update(label="✅ 답변 생성이 완료되었습니다.", state="complete")


            conversation["answer"] = full_response.replace("**' ", "**").replace("'**", "**") # messages

            ## 서버에서 응답받은 내용 Main Page에 셋팅
            st.session_state.app_mode = "CONVERSATION"
            st.session_state.viewing_history = False
            st.session_state.messages.append(conversation)

            response_placeholder.markdown(conversation['answer'])
            
            # 파일 UI
            for i, f in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([4, 1])
                col1.write(f"📎 {f.name}")

                if col2.button("❌", key=f"del_{i}", type="tertiary"):
                    st.session_state.fileFullPath = None
                    st.session_state.uploaded_files.pop(i)

        except requests.RequestException as e:
            st.error(f"API 요청 오류: {str(e)}")



        



def render_ui():
    # 페이지 설정
    st.set_page_config(page_title="AI Assistant"
                       , page_icon="🤖"
                       , initial_sidebar_state="collapsed"  # 처음에는 접힌 상태
                      )

    render_sidebar()
        
    if st.session_state.app_mode == "CONVERSATION":
        # 제목 및 소개
        st.title(st.session_state.ui_chat_agent_name)
        st.markdown(st.session_state.ui_chat_agent_desc )
    
        # RAG모드일 때만 accept_file을 허용한다.
        bAcceptFile = False
        if st.session_state.ui_chat_agent_mode == "NoticeScanAgent":
            bAcceptFile = True
        elif st.session_state.ui_chat_agent_mode == "PpsAssistAgent":
            bAcceptFile = True

        # uploaded_files값이 없는 경우 기본값 셋팅
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []

        # 사용자 chat 입력창을 생성한다.
        if prompt := st.chat_input(placeholder="메세지를 입력하세요.", accept_file=bAcceptFile):
            st.session_state.update({"app_mode": "CONVERSATION"})

            if "files" in prompt and prompt.files:
                # 업로드된 파일을 저장하고 파일 경로를 세션 상태에 저장합니다.
                st.session_state.uploaded_files.extend(prompt.files)

                # 업로드된 파일을 저장하고 파일 경로를 세션 상태에 저장합니다.
                today_str = datetime.now().strftime("%Y%m%d")
                st.session_state.fileFullPath = fileUtils.save_uploaded_file(f"./uploadFile/{today_str}/", prompt.files[0])
                
                st.session_state.update({"question": prompt.text})
            else:
                if "text" in prompt:
                    st.session_state.update({"question": prompt.text})
                else:
                    st.session_state.update({"question": prompt})

        current_mode = st.session_state.app_mode

        if current_mode == "CONVERSATION":
            startConvrstn()
    else:
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
                        

                        st.session_state.app_mode = "CONVERSATION"
                        st.session_state.messages = []
                        st.session_state.convrstn_id = str(uuid.uuid4())
                        st.rerun()

        st.info("원하는 에이전트의 '시작하기' 버튼을 클릭하면 대화 화면으로 이동합니다.")



if __name__ == "__main__":
    # 전체 파라미터 딕셔너리 가져오기
    params = st.query_params

    # 개별 파라미터 읽기
    app_mode = st.query_params.get("appMode", "")

    # 개별 파라미터 읽기
    agent_mode = st.query_params.get("agentMode", "")

    if app_mode == "CONVERSATION":
        if agent_mode != "":
            
            st.session_state.app_mode = app_mode

            # Agent 그리드 구성을 위한 데이터 조회        
            agents = fetch_agent_list();
            
            if agents is not None and len(agents) > 0:
                selected_agent = next((agent for agent in agents if agent["mode"] == agent_mode), None)
                if selected_agent is not None:
                    st.session_state.ui_chat_agent_id = selected_agent["agent_id"]
                    st.session_state.ui_chat_mode = selected_agent["description"]
                    st.session_state.ui_chat_agent_mode = selected_agent["mode"]
                    st.session_state.ui_chat_agent_name = selected_agent["name"]
                    st.session_state.ui_chat_agent_desc = selected_agent["description"]
                else:
                    st.warning(f"지정된 agentMode '{agent_mode[0]}'에 해당하는 에이전트를 찾을 수 없습니다.")

            st.session_state.ui_chat_agent_mode = agent_mode[0]


    load_dotenv()

    # 세션 상태 초기화
    init_session_state()

    render_ui()
