import os
import uuid
import requests
import streamlit as st
from datetime import datetime


# myUtils
from utils import fileUtils

####################################################################################
### MCP를 호출하는 CONVERSATION 서버의 API를 호출하고 응답을 화면에 출력한다.                    ###
####################################################################################
def render():
    
    # 제목 및 소개
    st.title(st.session_state.ui_chat_agent_name)
    st.markdown(st.session_state.ui_chat_agent_desc )

    # uploaded_files값이 없는 경우 기본값 셋팅
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # 사용자 chat 입력창을 생성한다.
    if prompt := st.chat_input(placeholder="메세지를 입력하세요.", accept_file=True):
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