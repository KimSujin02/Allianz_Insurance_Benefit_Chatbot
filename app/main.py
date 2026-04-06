from __future__ import annotations

from dotenv import load_dotenv
from pathlib import Path
import uuid

import streamlit as st
from rag_utils import run_chat_turn

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(page_title="Allianz Insurance RAG Chatbot", layout="wide")

st.title("Allianz 보험 혜택 상담 챗봇")
st.caption("LangGraph 기반 멀티턴 기억 + 추가 질문 + 문서 근거 답변")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "slots" not in st.session_state:
    st.session_state.slots = {}


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.slots = {}


with st.sidebar:
    st.header("예시 질문")
    st.markdown(
        """
- 싱가포르에서 입원 전에 사전승인이 필요한가요?
- 사전승인 폼에 어떤 정보를 입력해야 하나요?
- 영국에서 청구하려면 어떤 서류가 필요한가요?
- 홍콩에서 출산 관련 보장은 어떻게 되나요?
- 두바이에서 외래 진료는 보장되나요?
"""
    )
    st.markdown("---")
    st.write(f"Thread ID: {st.session_state.thread_id}")
    if st.button("대화 초기화"):
        reset_conversation()
        st.rerun()

# 기존 대화 출력
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문을 입력하세요")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("문서 검색 및 답변 생성 중..."):
            result = run_chat_turn(
                question=user_input,
                thread_id=st.session_state.thread_id,
                prior_slots=st.session_state.slots,
            )

        answer = result.get("answer", "답변을 생성하지 못했습니다.")
        slots = result.get("slots", {})
        docs = result.get("retrieved_docs", [])
        suggested_next_questions = result.get("suggested_next_questions", [])
        search_queries = result.get("search_queries", [])
        needs_followup = result.get("needs_followup", False)

        st.markdown(answer)

        if slots:
            with st.expander("현재 기억 중인 정보"):
                st.json(slots)

        if search_queries:
            with st.expander("사용된 검색 질의"):
                st.json(search_queries)

        if docs:
            with st.expander("검색된 근거 문서 보기"):
                for i, d in enumerate(docs, start=1):
                    st.markdown(
                        f"### {i}. {d.metadata.get('source')} "
                        f"(page {d.metadata.get('page')}, region={d.metadata.get('region')}, type={d.metadata.get('doc_type')})"
                    )
                    st.write(d.page_content)

        if suggested_next_questions:
            st.markdown("### 이어서 물어볼 만한 질문")
            for q in suggested_next_questions:
                st.markdown(f"- {q}")
        elif needs_followup:
            st.caption("추가 정보를 알려주시면 더 정확한 문서 기반 답변을 드릴 수 있습니다.")

    st.session_state.slots = slots
    st.session_state.chat_history.append({"role": "assistant", "content": answer})