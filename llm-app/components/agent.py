import os
import streamlit as st
import requests
import json
from utils import config

# 포트 충돌 방지를 위해 환경변수 사용
API_BASE_URL = config.settings.API_BASE_URL

# API로 Q&A 이력 조회
def fetch_agent_list():
    """API를 통해 Agent 목록 가져오기"""
    try:
        response = requests.get(f"{API_BASE_URL}/agents/")
        if response.status_code == 200:
            agents = response.json()
            # API 응답 형식에 맞게 데이터 변환 (agent_id, name, description, created_at 등  필드 포함)
            return agents
        else:
            st.error(f"Agent 목록 조회 실패: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return []