import os
import json
import uuid
import requests
import streamlit as st
from datetime import datetime


# myUtils
from utils import file_utils

####################################################################################
### MCP를 호출하는 CONVERSATION 서버의 API를 호출하고 응답을 화면에 출력한다.                    ###
####################################################################################
def render():
    # 제목 및 소개
    st.title(st.session_state.ui_chat_agent_name)
    st.markdown(st.session_state.ui_chat_agent_desc)

    # 상태 초기화
    if "selected_file_name" not in st.session_state:
        st.session_state.selected_file_name = None
    if "fileFullPath" not in st.session_state:
        st.session_state.file_full_path = None
    if "response_text" not in st.session_state:
        st.session_state.response_text = None
    if "status_message" not in st.session_state:
        st.session_state.status_message = "파일을 선택하면 자동으로 분석을 시작합니다."



    # 초기화 버튼
    if st.button("초기화", type="secondary"):
        st.session_state.selected_file_name = None
        st.session_state.file_full_path = None
        st.session_state.response_text = None
        st.session_state.status_message = "파일을 선택하면 자동으로 분석을 시작합니다."
        st.rerun()

    # 파일 업로드 UI
    with st.container(border=True):
        st.markdown("#### 📄 분석 대상 문서")
        uploaded_file = st.file_uploader(
            "공고문 파일(PDF, HTML)을 업로드하세요.", 
            type=["pdf", "html", "htm"], 
            key="notice_scan_file",
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        if st.session_state.selected_file_name != uploaded_file.name:
            # 새 파일을 선택하면 이전 결과 초기화
            st.session_state.response_text = None
            
            st.session_state.selected_file_name = uploaded_file.name
            st.session_state.response_text = None
            st.session_state.status_message = "파일을 저장하고 분석을 준비 중입니다..."

            today_str = datetime.now().strftime("%Y%m%d")
            st.session_state.file_full_path = file_utils.save_uploaded_file(f"./uploadFile/{today_str}/", uploaded_file)

            # API 호출
            API_BASE_URL = os.getenv("API_BASE_URL")
            request_data = {
                "agent_id": st.session_state.get("ui_chat_agent_id", ""),
                "agent_mode": st.session_state.get("ui_chat_agent_mode", ""),
                "conversation_id": st.session_state.get("conversation_id", str(uuid.uuid4())),
                "fileFullPath": st.session_state.file_full_path,
                "question": "",
                "enableExtDocse": st.session_state.get("ui_enable_ext_docse", False),
            }

            try:
                response_placeholder = st.empty()
                full_response = ""

                with st.status("🔍 분석을 시작합니다...", state="running") as status:
                    status.update(label="⚙️ 파일을 분석하는 중입니다...", state="running")
                    r = requests.post(f"{API_BASE_URL}/convrstn/question", json=request_data, headers={"Content-Type": "application/json"}, stream=False)
                    r.raise_for_status()
                    full_response = r.text
                    status.update(label="✅ 분석이 완료되었습니다.", state="complete")

                st.session_state.response_text = full_response.strip()
                st.session_state.status_message = "분석 결과가 아래에 표시됩니다."

            except requests.RequestException as e:
                st.session_state.status_message = f"API 요청 오류: {str(e)}"
                st.error(st.session_state.status_message)

    # 상태 알림 (모바일 가독성을 위해 간결하게)
    if st.session_state.selected_file_name:
        st.info(f"📁 **파일명:** {st.session_state.selected_file_name}\n\n💡 {st.session_state.status_message}")
    else:
        st.write(f"ℹ️ {st.session_state.status_message}")

    if st.session_state.response_text:
        # 규정 위반 경고 표시 (최상단 배치)
        res_json = json.loads(st.session_state.response_text)
        
        st.markdown("### 분석 결과")
        
        try:
            # JSON 데이터 파싱
            res_json = json.loads(st.session_state.response_text)
            data = res_json.get("extracted_data", {})
            
            if not data:
                st.warning("추출된 구조화 데이터가 없습니다.")
                st.code(st.session_state.response_text, language="json")
            else:
                
                gen = data.get("general", {})
                exe = data.get("execution", {})
                items = data.get("items", [])
                
                if res_json.get("is_violating"):
                    st.warning("⚠️ **규정 위반 의심:** 공고문 내 독소조항이나 규정 위반 가능성이 감지되었습니다.")

                # 1. 요약 메트릭 (모바일에서는 2열씩 배치되도록 조정)
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                with col1:
                    st.metric("공고 종류", gen.get("noticeType", "-"))
                with col2:
                    st.metric("계약 방법", gen.get("contractMethod", "-"))
                with col3:
                    bid_end = exe.get("bidEndDate", "-").split('T')[0]
                    st.metric("입찰 마감", bid_end)
                with col4:
                    st.metric("품목 수", len(items))

                # 2. 탭 구성
                tab1, tab2, tab3, tab4 = st.tabs(["📋 상세정보", "📦 품목", "💻 공고입력 데이터(JSON)", "🔍 원문"])

                with tab1:
                    st.subheader(gen.get('noticeName', '공고명 없음'))
                    
                    with st.expander("🎯 낙찰 방법 및 상세", expanded=True):
                        st.write(f"**방법:** {gen.get('awardMethod', '-')}")
                        st.write(f"**상세:** {gen.get('awardDetail', '-')}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**공고번호**\n{gen.get('noticeNo', '-')}")
                        st.markdown(f"**게시일시**\n{gen.get('postDate', '-')}")
                        st.markdown(f"**공고기관**\n{gen.get('agency', '-')}")
                    with c2:
                        st.markdown(f"**담당자**\n{exe.get('manager', '-')}")
                        st.markdown(f"**개찰일시**\n{exe.get('openDate', '-')}")
                        st.markdown(f"**보증금면제**\n{exe.get('depositExemptYn', '-')}")
                    
                    st.markdown(f"**개찰장소**\n{exe.get('openPlace', '-')}")

                with tab2:
                    if items:
                        # 데이터프레임으로 깔끔하게 표시
                        st.dataframe(items, use_container_width=True, hide_index=True)
                    else:
                        st.write("등록된 품목 정보가 없습니다.")
                
                with tab3:
                    st.markdown("#### 추출 데이터 구조")
                    json_string = json.dumps(data, ensure_ascii=False, indent=2)
                    st.code(json_string, language="json")


                with tab4:
                    st.json(res_json)

        except Exception as e:
            # JSON 파싱 실패 시 일반 텍스트 출력 (폴백)
            st.markdown(st.session_state.response_text)