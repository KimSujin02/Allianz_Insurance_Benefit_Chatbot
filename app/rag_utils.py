from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent

PERSIST_DIR = str(BASE_DIR / "vectordb")
COLLECTION_NAME = "allianz_care"


# ---------------------------------------------------------
# 1. 벡터스토어 연결
# ---------------------------------------------------------
def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ---------------------------------------------------------
# 2. 질문 의도 분류
# ---------------------------------------------------------
def classify_intent(question: str) -> str:
    q = question.lower()

    preauth_keywords = [
        "사전승인", "pre-author", "preauthor", "pre-auth", "preauth",
        "입원 전에", "입원 전", "승인받", "승인 필요", "direct billing",
        "planned hospitalisation", "planned hospitalization"
    ]
    claim_keywords = [
        "청구", "claim", "환급", "보험금", "영수증", "서류", "reimbursement",
        "invoice", "receipt", "청구서", "돌려받", "submit a claim"
    ]

    if any(k in q for k in preauth_keywords):
        return "preauth"
    if any(k in q for k in claim_keywords):
        return "claim"
    return "coverage"


# ---------------------------------------------------------
# 3. 질문에서 지역 자동 감지
# ---------------------------------------------------------
def detect_region(question: str) -> Optional[str]:
    q = question.lower()

    region_patterns = {
        "singapore": [r"싱가포르", r"\bsingapore\b"],
        "dubai_northern_emirates": [r"두바이", r"북부에미리트", r"\bdubai\b", r"northern emirates", r"\buae\b"],
        "lebanon": [r"레바논", r"\blebanon\b"],
        "indonesia": [r"인도네시아", r"\bindonesia\b"],
        "vietnam": [r"베트남", r"\bvietnam\b"],
        "hong_kong": [r"홍콩", r"hong kong", r"\bhk\b"],
        "china": [r"중국", r"\bchina\b", r"중화권"],
        "switzerland": [r"스위스", r"\bswitzerland\b", r"\bsuisse\b"],
        "uk": [r"영국", r"\buk\b", r"united kingdom", r"\bengland\b", r"britain"],
        "france_benelux_monaco": [r"프랑스", r"\bfrance\b", r"benelux", r"모나코", r"\bmonaco\b"],
        "latin_america": [r"남미", r"라틴아메리카", r"latin america"],
        "global": [r"글로벌", r"전세계", r"worldwide", r"global"],
    }

    for region, patterns in region_patterns.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return region

    return None


# ---------------------------------------------------------
# 4. 검색 대상 doc_type 결정
# ---------------------------------------------------------
def get_allowed_doc_types(intent: str) -> List[str]:
    if intent == "coverage":
        return ["benefit_guide", "tob"]
    if intent == "preauth":
        return ["benefit_guide", "preauth_form", "tob"]
    if intent == "claim":
        return ["benefit_guide", "claim_form"]
    return ["benefit_guide", "tob"]


# ---------------------------------------------------------
# 5. 다국어 검색 쿼리 생성
#    - 한글 원문
#    - 영어 자연어
#    - 영어 키워드형
# ---------------------------------------------------------
def make_search_queries(question: str, intent: str, detected_region: Optional[str]) -> List[str]:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""
You are generating search queries for retrieving passages from English insurance documents.

User question:
{question}

Intent:
{intent}

Detected region:
{detected_region or "none"}

Return strict JSON only in this format:
{{
  "queries": [
    "query1",
    "query2",
    "query3"
  ]
}}

Rules:
- queries must be short and retrieval-friendly
- include one English natural-language query
- include one English keyword-style query
- include one bilingual or Korean-friendly query if helpful
- keep insurance terms precise:
  - 사전승인 -> pre-authorisation / preauthorization
  - 직접청구 -> direct billing
  - 청구 -> claim / reimbursement
  - 보장한도 -> benefit limit / coverage limit
- if region exists, include the English region name
- do not explain anything
"""

    queries = [question.strip()]

    try:
        result = llm.invoke(prompt).content.strip()
        data = json.loads(result)
        llm_queries = data.get("queries", [])
        for q in llm_queries:
            if isinstance(q, str) and q.strip():
                queries.append(q.strip())
    except Exception:
        fallback_queries = build_fallback_queries(question, intent, detected_region)
        queries.extend(fallback_queries)

    # 중복 제거
    deduped = []
    seen = set()
    for q in queries:
        normalized = q.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(q)

    return deduped[:4]


def build_fallback_queries(question: str, intent: str, detected_region: Optional[str]) -> List[str]:
    region_text = detected_region.replace("_", " ") if detected_region else ""

    if intent == "preauth":
        return [
            f"{region_text} pre-authorisation required before planned hospitalisation".strip(),
            f"{region_text} preauthorisation direct billing inpatient".strip(),
        ]
    if intent == "claim":
        return [
            f"{region_text} claim reimbursement required documents".strip(),
            f"{region_text} invoice receipt claim form".strip(),
        ]
    return [
        f"{region_text} coverage benefits limits exclusions".strip(),
        f"{region_text} inpatient outpatient maternity benefit limit".strip(),
    ]


# ---------------------------------------------------------
# 6. 문서 고유키
# ---------------------------------------------------------
def doc_unique_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk_idx"),
        doc.metadata.get("doc_type"),
        doc.metadata.get("region"),
    )


# ---------------------------------------------------------
# 7. 간단 rerank
# ---------------------------------------------------------
def score_document(question: str, doc: Document, intent: str, detected_region: Optional[str]) -> int:
    score = 0
    q = question.lower()
    content = doc.page_content.lower()
    metadata = doc.metadata

    # 지역 가중치
    if detected_region and metadata.get("region") == detected_region:
        score += 6
    if metadata.get("region") == "global":
        score += 2

    # 문서 타입 가중치
    if intent == "preauth" and metadata.get("doc_type") in ["preauth_form", "benefit_guide", "tob"]:
        score += 4
    elif intent == "claim" and metadata.get("doc_type") in ["claim_form", "benefit_guide"]:
        score += 4
    elif intent == "coverage" and metadata.get("doc_type") in ["benefit_guide", "tob"]:
        score += 4

    # 키워드 일치 가중치
    keyword_groups = [
        ["pre-authorisation", "preauthorization", "preauth", "사전승인"],
        ["direct billing", "직접청구"],
        ["claim", "reimbursement", "청구", "환급"],
        ["invoice", "receipt", "영수증", "서류"],
        ["inpatient", "hospitalisation", "hospitalization", "입원"],
        ["outpatient", "외래"],
        ["maternity", "출산"],
        ["benefit limit", "coverage limit", "limit", "한도"],
        ["exclusion", "제외", "면책"],
    ]

    for group in keyword_groups:
        if any(k in q for k in group) and any(k in content for k in group):
            score += 3

    # 페이지 내용 길이 최소 보정
    score += min(len(content) // 300, 3)

    return score


# ---------------------------------------------------------
# 8. 문서 검색
#    - 다중 질의
#    - MMR 검색
#    - dedup
#    - rerank
# ---------------------------------------------------------
def retrieve_documents(question: str):
    vectordb = get_vectorstore()

    intent = classify_intent(question)
    detected_region = detect_region(question)
    allowed_doc_types = get_allowed_doc_types(intent)

    regions = ["global"]
    if detected_region and detected_region != "global":
        regions.append(detected_region)

    queries = make_search_queries(question, intent, detected_region)

    all_docs: List[Document] = []
    seen = set()

    search_filter = {
        "$and": [
            {"doc_type": {"$in": allowed_doc_types}},
            {"region": {"$in": regions}}
        ]
    }

    for q in queries:
        try:
            docs = vectordb.max_marginal_relevance_search(
                q,
                k=6,
                fetch_k=20,
                filter=search_filter,
            )
        except Exception:
            docs = vectordb.similarity_search(
                q,
                k=6,
                filter=search_filter,
            )

        for d in docs:
            key = doc_unique_key(d)
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    ranked_docs = sorted(
        all_docs,
        key=lambda d: score_document(question, d, intent, detected_region),
        reverse=True
    )

    return ranked_docs[:10], intent, detected_region, regions, queries


# ---------------------------------------------------------
# 9. 문서 컨텍스트 생성
#    - search_tags는 LLM 답변 문맥에서는 제거
# ---------------------------------------------------------
def strip_search_tags(text: str) -> str:
    if "[search_tags]" in text:
        return text.split("[search_tags]")[0].strip()
    return text.strip()


def build_context(docs: List[Document]) -> str:
    context_parts = []

    for d in docs:
        source = d.metadata.get("source")
        region = d.metadata.get("region")
        page = d.metadata.get("page")
        doc_type = d.metadata.get("doc_type")
        year = d.metadata.get("doc_year")
        content = strip_search_tags(d.page_content)

        context_parts.append(
            f"[문서: {source} | region: {region} | type: {doc_type} | year: {year} | page: {page}]\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# 10. 답변 생성
# ---------------------------------------------------------
def generate_answer(question: str) -> Tuple[str, list]:
    docs, intent, detected_region, regions, queries = retrieve_documents(question)
    context = build_context(docs)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    region_text = detected_region if detected_region else "없음 (global 공통 문서 기준 검색)"

    prompt = f"""
너는 Allianz 보험 문서 기반 안내 챗봇이다.
반드시 아래 문맥에 근거해서만 답변해야 한다.
문맥에 없는 내용은 추측하지 마라.
모르면 "문서상 확인 불가"라고 답해라.
법률적/의학적 확정 판단처럼 말하지 말고 "문서 기준 안내" 형식으로 답해라.
답변은 반드시 한국어로 작성하라.

검색 설정:
- 질문 의도: {intent}
- 감지된 지역: {region_text}
- 실제 검색 region 목록: {regions}
- 검색에 사용된 질의: {queries}

답변 규칙:
1. 지역이 감지되면 그 지역 문서를 최우선으로 반영하라.
2. 공통 규칙은 global 문서를 보조 근거로 사용하라.
3. 지역이 감지되지 않았다면 global 기준으로 답하라.
4. 출처는 반드시 문서명과 페이지를 적어라.
5. 검색은 참고용일 뿐이며, 실제 결론은 문맥 근거에 맞게 보수적으로 작성하라.
6. 동일한 내용이 지역 문서와 global 문서에서 다르면 지역 문서를 우선한다.

답변 형식:
1. 결론
2. 지역 기준 근거
3. 공통 규칙
4. 필요한 절차 또는 주의사항
5. 출처

질문:
{question}

문맥:
{context}
"""

    result = llm.invoke(prompt)
    return result.content, docs


# ---------------------------------------------------------
# 11. 디버깅용 단독 실행
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_question = "싱가포르에서 입원 치료 전에 사전승인이 필요한가요?"
    answer, docs = generate_answer(sample_question)

    print("=== ANSWER ===")
    print(answer)
    print("\n=== RETRIEVED DOCS ===")
    for doc in docs:
        print(doc.metadata)