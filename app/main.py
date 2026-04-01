from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path

import streamlit as st
from rag_utils import generate_answer, detect_region, classify_intent

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(page_title="Allianz Insurance RAG", layout="wide")

st.title("Allianz 보험 혜택 RAG 챗봇")
st.caption("지역 자동 감지 + 공통 문서/지역 문서 동시 검색")

with st.sidebar:
    st.header("예시 질문")
    st.markdown("""
- 싱가포르에서 입원 치료 전에 사전승인이 필요한가요?
- 홍콩에서 출산 관련 보장은 어떻게 되나요?
- 중국에서 direct billing 가능한가요?
- 영국에서 청구하려면 어떤 서류가 필요한가요?
- 두바이에서 외래 진료 co-payment가 있나요?
- 스위스에서 치료 가능한 범위가 어떻게 되나요?
""")

question = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 싱가포르에서 입원 전에 사전승인이 필요한가요?"
)

if st.button("질문하기") and question:
    with st.spinner("문서 검색 및 답변 생성 중..."):
        detected_region = detect_region(question)
        detected_intent = classify_intent(question)
        answer, docs = generate_answer(question)

    st.subheader("질문 분석")
    st.write(f"- 감지된 지역: `{detected_region or '없음 (global 기준)'}`")
    st.write(f"- 감지된 의도: `{detected_intent}`")

    st.subheader("답변")
    st.write(answer)

    with st.expander("검색된 근거 문서 보기"):
        for i, d in enumerate(docs, start=1):
            st.markdown(
                f"### {i}. {d.metadata.get('source')} "
                f"(page {d.metadata.get('page')}, region={d.metadata.get('region')}, type={d.metadata.get('doc_type')})"
            )
            st.write(d.page_content)