import streamlit as st


def init_session_state():
    # 세션 스테이트 초기화
    if "app_mode" not in st.session_state:
        reset_session_state()


def reset_session_state():
    st.session_state.round = 0
    st.session_state.viewing_history = False
    st.session_state.loaded_debate_id = None


def set_debate_to_state(topic, messages, debate_id):
    st.session_state.messages = messages
    st.session_state.viewing_history = True
    st.session_state.debate_topic = topic
    st.session_state.loaded_debate_id = debate_id
