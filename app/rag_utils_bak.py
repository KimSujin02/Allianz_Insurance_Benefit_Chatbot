from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(BASE_DIR / "vectordb")
COLLECTION_NAME = "allianz_care"

# 1. 벡터스토어 연결
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},   # GPU 있으면 "cuda"
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# 2. 입력 언어 감지 (간단 rule-based)
#    - 최종 언어 결정은 normalize_question() 결과를 우선 사용
def detect_language(text: str) -> str:
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return "zh"
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return "ja"
    if any('\uac00' <= c <= '\ud7a3' for c in text):
        return "ko"
    return "en"

# 3. 질문 표준화
# - language / intent / region / english_query를 생성
# - 다국어 입력을 내부 공통 표현으로 맞춤
# - llm이 사용자의 질문을 바탕으로 rag 내에서 검색할 질의를 생성하기 위한 함수
def normalize_question(question: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""
You are a multilingual insurance query normalizer.

Your job is to analyze a user's insurance question and return a standardized JSON object.

Return STRICT JSON only in the following format:
{{
  "language": "ko|en|zh|ja|es|fr|de|other",
  "intent": "coverage|preauth|claim",
  "region": "singapore|dubai_northern_emirates|lebanon|indonesia|vietnam|hong_kong|china|switzerland|uk|france_benelux_monaco|latin_america|global|none",
  "english_query": "A concise English search query for retrieving relevant passages from English insurance documents",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Rules:
- "intent":
  - preauth = prior approval, pre-authorisation, admission approval, hospital approval, direct billing preparation
  - claim = claims, reimbursement, invoice, receipt, required documents, refund
  - coverage = benefits, limits, exclusions, waiting periods, whether covered
- "region" must be one of the allowed enum values above
- if no region is mentioned, use "none"
- "english_query" must be natural and retrieval-friendly
- "keywords" must be short English retrieval keywords
- Do not explain anything
- Output JSON only

User question:
{question}
"""

    fallback_language = detect_language(question)

    try:
        # llm 응답 받기
        raw = llm.invoke(prompt).content.strip()
        data = json.loads(raw)

        # llm 응답에 대한 처리
        language = data.get("language", fallback_language)
        intent = data.get("intent", "coverage")
        region = data.get("region", "none")
        english_query = data.get("english_query", question)
        keywords = data.get("keywords", [])

        # 이 조건에 들어갈 일이 있나?
        # 혹시 LLM이 enum 값을 잘못 반환하는 경우를 대비
        if language not in {"ko", "en", "zh", "ja", "es", "fr", "de", "other"}:
            language = fallback_language

        if intent not in {"coverage", "preauth", "claim"}:
            intent = "coverage"

        # 허용된 지역 + 글로벌 외에는 모두 none으로 처리
        allowed_regions = {
            "singapore",
            "dubai_northern_emirates",
            "lebanon",
            "indonesia",
            "vietnam",
            "hong_kong",
            "china",
            "switzerland",
            "uk",
            "france_benelux_monaco",
            "latin_america",
            "global",
            "none",
        }
        if region not in allowed_regions:
            region = "none"

        if not isinstance(keywords, list):
            keywords = []

        return {
            "language": language,
            "intent": intent,
            "region": region,
            "english_query": english_query.strip(),
            "keywords": [str(k).strip() for k in keywords if str(k).strip()],
        }

    # LLM이 JSON을 제대로 반환하지 못하는 경우, fallback으로 간단히 규칙 기반으로 감지
    except Exception:
        return fallback_normalize_question(question, fallback_language)

# 3-1. 질문 표준화 fallback
# - LLM 실패 시 사용되는 대체 함수
def fallback_normalize_question(question: str, language: str) -> Dict[str, Any]:
    q = question.lower()

    preauth_terms = [
        "사전승인", "입원 전 승인", "pre-author", "preauthor", "pre-auth",
        "prior approval", "approval before", "hospital approval", "direct billing"
    ]
    claim_terms = [
        "청구", "환급", "보험금", "영수증", "서류", "claim",
        "reimbursement", "invoice", "receipt", "refund"
    ]

    if any(t in q for t in preauth_terms):
        intent = "preauth"
    elif any(t in q for t in claim_terms):
        intent = "claim"
    else:
        intent = "coverage"

    # 지역 감지
    region = detect_region_fallback(question)
    # 영어 fallback 질의 생성
    english_query = build_fallback_english_query(question, intent, region)

    return {
        "language": language,
        "intent": intent,
        "region": region if region else "none",
        "english_query": english_query,
        "keywords": [],
    }

# 4. 지역 fallback 감지
# - LLM 실패 시 최소한의 보조용
# - 다국어 alias를 조금 더 넓게 반영
def detect_region_fallback(question: str) -> Optional[str]:
    q = question.lower()

    region_patterns = {
        "singapore": [
            r"싱가포르", r"\bsingapore\b", r"新加坡", r"シンガポール"
        ],
        "dubai_northern_emirates": [
            r"두바이", r"북부에미리트", r"\bdubai\b", r"northern emirates",
            r"\buae\b", r"迪拜", r"ドバイ"
        ],
        "lebanon": [
            r"레바논", r"\blebanon\b", r"黎巴嫩", r"レバノン"
        ],
        "indonesia": [
            r"인도네시아", r"\bindonesia\b", r"印度尼西亚", r"インドネシア"
        ],
        "vietnam": [
            r"베트남", r"\bvietnam\b", r"越南", r"ベトナム"
        ],
        "hong_kong": [
            r"홍콩", r"hong kong", r"\bhk\b", r"香港"
        ],
        "china": [
            r"중국", r"\bchina\b", r"中?国", r"中国", r"中國", r"中화권", r"中国大陆"
        ],
        "switzerland": [
            r"스위스", r"\bswitzerland\b", r"\bsuisse\b", r"瑞士", r"スイス"
        ],
        "uk": [
            r"영국", r"\buk\b", r"united kingdom", r"\bengland\b",
            r"britain", r"英国", r"イギリス"
        ],
        "france_benelux_monaco": [
            r"프랑스", r"\bfrance\b", r"benelux", r"모나코", r"\bmonaco\b",
            r"法国", r"摩纳哥", r"フランス", r"モナコ"
        ],
        "latin_america": [
            r"남미", r"라틴아메리카", r"latin america", r"拉丁美洲", r"中南米"
        ],
        "global": [
            r"글로벌", r"전세계", r"worldwide", r"global", r"全球", r"全世界"
        ],
    }

    for region, patterns in region_patterns.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return region

    return None

# 5. 검색 대상 doc_type 결정
def get_allowed_doc_types(intent: str) -> List[str]:
    if intent == "coverage":    # 보험 적용 범위인 경우
        return ["benefit_guide", "tob"] # 보험 약관과 혜택 가이드가 모두 관련 정보 포함할 수 있도록 허용
    if intent == "preauth":     # 사전 신청
        return ["benefit_guide", "preauth_form", "tob"]
    if intent == "claim":       # 청구
        return ["benefit_guide", "claim_form"]
    return ["benefit_guide", "tob"]

# 6. 검색 질의 생성
#    - 원문 질문 : 사용자 입력
#    - 표준 영어 질의 : LLM이 생성한 영어 검색 질의
#    - keyword 질의 : 
#   - fallback 질의
def make_search_queries(normalized: Dict[str, Any], original_question: str) -> List[str]:
    region = normalized["region"]
    intent = normalized["intent"]
    english_query = normalized["english_query"]
    keywords = normalized.get("keywords", [])

    # 원문 질문과 LLM이 생성한 영어 질의
    queries = [original_question.strip(), english_query.strip()]
    
    # keyword 기반 질의
    # 정상적으로 llm이 생성해준 경우와 그렇지 않은 경우 모두 대비
    keyword_query = build_keyword_query(intent, region, keywords)
    if keyword_query:
        queries.append(keyword_query)
    
    # LLM이 영어 질의나 키워드 질의를 제대로 생성하지 못하는 경우를 대비해, fallback으로 간단한 규칙 기반 질의 추가
    fallback_queries = build_fallback_queries(intent, region)
    queries.extend(fallback_queries)

    # deduped : 중복 제거 + 공백 제거
    # LLM이 의도한 검색 질의를 제대로 생성하지 못하는 경우를 대비해, 원문 질문과 간단한 규칙 기반 질의를 모두 포함하되, 중복은 제거
    deduped = []
    seen = set()
    for q in queries:
        nq = q.lower().strip()
        if nq and nq not in seen:
            seen.add(nq)
            deduped.append(q.strip())

    return deduped[:5]

# 6-1. keyword 질의 생성
def build_keyword_query(intent: str, region: str, keywords: List[str]) -> str:
    region_text = "" if region in {"none", "global"} else region.replace("_", " ")
    base = " ".join(keywords[:5]).strip()

    # base가 있는 경우 : region + keywords 조합
    # 정상적으로 LLM이 keywords를 생성한 경우, 이를 활용해 검색 질의를 만들어줌
    if base:
        return f"{region_text} {base}".strip()

    # base가 없는 경우 : intent와 region 기반의 간단한 템플릿으로 생성
    if intent == "preauth":
        return f"{region_text} pre-authorisation inpatient hospitalisation approval".strip()
    if intent == "claim":
        return f"{region_text} claim reimbursement invoice receipt documents".strip()
    # intent == "coverage"인 경우
    return f"{region_text} coverage benefits limits exclusions".strip()

# 6-2. fallback 질의 생성
# - LLM이 영어 질의를 제대로 생성하지 못하는 경우 대비
def build_fallback_queries(intent: str, region: str) -> List[str]:
    region_text = "" if region in {"none", "global"} else region.replace("_", " ")

    if intent == "preauth":
        return [
            f"{region_text} pre-authorisation required before inpatient treatment".strip(),
            f"{region_text} planned hospitalisation prior approval".strip(),
        ]
    if intent == "claim":
        return [
            f"{region_text} claim reimbursement required documents".strip(),
            f"{region_text} invoice receipt claim form".strip(),
        ]
    return [
        f"{region_text} coverage benefits limits exclusions".strip(),
        f"{region_text} inpatient outpatient benefit limit".strip(),
    ]

# 6-1. fallback 영어 질의 생성
# - LLM이 영어 질의를 제대로 생성하지 못하는 경우 대비
# - 간단한 규칙 기반으로 영어 검색 질의 생성
#   - 지역과 의도에 따라 기본적인 검색 질의 템플릿을 만들어줌
def build_fallback_english_query(question: str, intent: str, region: Optional[str]) -> str:
    region_text = "" if not region or region == "none" else region.replace("_", " ")

    if intent == "preauth":
        return f"Is pre-authorisation required before inpatient treatment {f'in {region_text}' if region_text else ''}?".strip()
    if intent == "claim":
        return f"What documents are required to submit a claim {f'in {region_text}' if region_text else ''}?".strip()
    return f"What is covered under the insurance plan {f'in {region_text}' if region_text else ''}?".strip()

# 7. 문서 고유키
def doc_unique_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk_idx"),
        doc.metadata.get("doc_type"),
        doc.metadata.get("region"),
    )

# 8. 간단 rerank
#    - 영어 기준 키워드로 점수화
def score_document(question: str, doc: Document, intent: str, detected_region: Optional[str]) -> int:
    score = 0
    q = question.lower()
    content = doc.page_content.lower()
    metadata = doc.metadata

    if detected_region and metadata.get("region") == detected_region:
        score += 6
    if metadata.get("region") == "global":
        score += 2

    if intent == "preauth" and metadata.get("doc_type") in ["preauth_form", "benefit_guide", "tob"]:
        score += 4
    elif intent == "claim" and metadata.get("doc_type") in ["claim_form", "benefit_guide"]:
        score += 4
    elif intent == "coverage" and metadata.get("doc_type") in ["benefit_guide", "tob"]:
        score += 4

    keyword_groups = [
        ["pre-authorisation", "preauthorization", "prior approval", "preauth"],
        ["direct billing"],
        ["claim", "reimbursement", "refund"],
        ["invoice", "receipt", "documents"],
        ["inpatient", "hospitalisation", "hospitalization", "admission"],
        ["outpatient"],
        ["maternity", "pregnancy"],
        ["benefit limit", "coverage limit", "limit"],
        ["exclusion"],
    ]

    for group in keyword_groups:
        if any(k in q for k in group) and any(k in content for k in group):
            score += 3

    score += min(len(content) // 300, 3)
    return score

# 9. 문서 검색
def retrieve_documents(question: str):
    vectordb = get_vectorstore()

    # 문서에 넣을 검색 질의 생성
    normalized = normalize_question(question)
    
    # intent : coverage / preauth / claim
    intent = normalized["intent"]
    # 지역
    detected_region = None if normalized["region"] == "none" else normalized["region"]
    # 검색 대상 doc_type 결정
    allowed_doc_types = get_allowed_doc_types(intent)

    # rag 문서 검색 : global 문서 + 감지된 지역 문서 동시에 검색하게
    regions = ["global"]
    if detected_region and detected_region != "global":
        regions.append(detected_region)
    
    # 검색 질의 생성 - 원문 질문 + 영어 변환 질문 + keyword 기반 질문 + fallback 질문
    queries = make_search_queries(normalized, question)

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
        key=lambda d: score_document(normalized["english_query"], d, intent, detected_region),
        reverse=True
    )

    return ranked_docs[:10], normalized, regions, queries

# 10. 문서 컨텍스트 생성
def strip_search_tags(text: str) -> str:
    if "[search_tags]" in text:
        return text.split("[search_tags]")[0].strip()
    return text.strip()

# 문서 컨텍스트 생성 - 검색된 문서들을 LLM이 이해하기 좋은 형태로 가공
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
            f"[Document: {source} | region: {region} | type: {doc_type} | year: {year} | page: {page}]\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)

# 11. 답변 언어 맵
LANGUAGE_NAME_MAP = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "other": "the same language as the user's question",
}

# 12. 답변 생성
def generate_answer(question: str) -> Tuple[str, list]:
    docs, normalized, regions, queries = retrieve_documents(question)
    context = build_context(docs)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    language_code = normalized["language"]
    answer_language = LANGUAGE_NAME_MAP.get(language_code, "English")
    intent = normalized["intent"]
    detected_region = normalized["region"]
    region_text = detected_region if detected_region != "none" else "none (global documents only)"
    english_query = normalized["english_query"]

    prompt = f"""
You are an Allianz insurance document-based assistant.

You must answer ONLY based on the provided context.
Do not guess any information that is not supported by the context.
If the answer cannot be confirmed from the documents, say that it cannot be confirmed from the documents.
Do not present the answer as a legal or medical final judgment. Present it as document-based insurance guidance.

IMPORTANT LANGUAGE RULE:
- You must answer in {answer_language}.
- The answer language must match the user's question language.
- Do not switch languages unless necessary for official document names or insurance terms.

Search settings:
- detected language: {language_code}
- intent: {intent}
- detected region: {region_text}
- actual searched regions: {regions}
- English normalized query: {english_query}
- search queries used: {queries}

Answer rules:
1. If a region-specific document exists, prioritize it.
2. Use global documents as supplementary evidence.
3. If region is not detected, answer based on global documents.
4. Cite the source document names and pages.
5. Be conservative and evidence-based.
6. If region-specific and global documents conflict, prioritize the region-specific document.

Answer format:
1. Conclusion
2. Region-specific basis
3. General/global rule
4. Procedure or notes
5. Sources

User question:
{question}

Context:
{context}
"""

    result = llm.invoke(prompt)
    return result.content, docs

# 13. 디버깅용 단독 실행
if __name__ == "__main__":
    sample_questions = [
        "싱가포르에서 입원 치료 전에 사전승인이 필요한가요?",
        "Is prior approval required for inpatient treatment in Singapore?",
        "住院治疗前在新加坡需要预先批准吗？",
        "シンガポールで入院治療の前に事前承認は必要ですか？",
    ]

    for sample_question in sample_questions:
        print("=" * 100)
        print("QUESTION:", sample_question)
        answer, docs = generate_answer(sample_question)
        print("=== ANSWER ===")
        print(answer)
        print("\n=== RETRIEVED DOCS ===")
        for doc in docs:
            print(doc.metadata)