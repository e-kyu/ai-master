import os
import streamlit as st
import requests
import json
from utils import config

# 포트 충돌 방지를 위해 환경변수 사용
API_BASE_URL = config.settings.API_BASE_URL

# API로 Q&A 이력 조회
def fetch_conversation_history():
    """API를 통해 Q&A 이력 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/conversation_history/")
        if response.status_code == 200:
            conversations = response.json()
            # API 응답 형식에 맞게 데이터 변환 (conversation_id, question, date)
            return [
                (conversation["conversation_id"], conversation["topic"], conversation["created_at"], conversation.get("agent_id", None), conversation.get("mode", None), conversation.get("name", None), conversation.get("description", None))
                for conversation in conversations
            ]
        else:
            st.error(f"Q&A 이력 조회 실패: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return []


# API로 특정 Q&A 데이터 조회
def fetch_conversation_by_conversation_id(conversation_id):
    """API를 통해 특정 Q&A 데이터 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/conversation_history/{conversation_id}")
        if response.status_code == 200:
            conversation_details = response.json()

            return conversation_details
        else:
            st.error(f"Q&A 데이터 조회 실패: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return None, None


# API로 Q&A 삭제
def delete_conversation_by_conversation_id(conversation_id):
    """API를 통해 특정 Q&A 삭제"""
    try:
        response = requests.delete(f"{API_BASE_URL}/conversation_history/{conversation_id}")
        if response.status_code == 200:
            st.success("Q&A이 삭제되었습니다.")
            return True
        else:
            st.error(f"Q&A 삭제 실패: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return False


# API로 모든 Q&A 삭제
def delete_all_conversations():
    """API를 통해 모든 Q&A 삭제"""
    try:
        # 모든 Q&A 목록 조회
        conversations = fetch_conversation_history()
        if not conversations:
            return True

        # 각 Q&A 항목 삭제
        success = True
        for conversation_id, _, _, _ in conversations:
            response = requests.delete(f"{API_BASE_URL}/conversation_history/{conversation_id}")
            if response.status_code != 200:
                success = False

        if success:
            st.success("모든 Q&A이 삭제되었습니다.")
        return success
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return False