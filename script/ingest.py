from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
DB_DIR = BASE_DIR / "vectordb"
COLLECTION_NAME = "allianz_care"

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


# 1. 파일 목록
FILES: List[Dict[str, Any]] = [
    # 글로벌 공통
    {
        "path": DATA_DIR / "DOC-Care-IBG-EN-1125_개인고객용혜택가이드.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
    },
    {
        "path": DATA_DIR / "care-tob-en_보장금액.pdf",
        "doc_type": "tob",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
    },
    {
        "path": DATA_DIR / "FRM-PreAuth-EN-0825_사전승인신청서.pdf",
        "doc_type": "preauth_form",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
    },
    {
        "path": DATA_DIR / "FRM-PCF-EN-1125_사후보험청구서.pdf",
        "doc_type": "claim_form",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
    },

    # 지역별 코퍼스
    {
        "path": DATA_DIR / "DOC-Singapore-IBG-EN-0126_싱가포르.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2026,
        "region": "singapore",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-Dubai-Northern-Emirates-EN-0126_두바이.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2026,
        "region": "dubai_northern_emirates",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-LEBANON-IBG-EN-0725_레바논.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2025,
        "region": "lebanon",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-Indonesia-en-UK-1123_인도네시아.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2024,
        "region": "indonesia",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-Vietnam-en-UK-0823_베트남.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2023,
        "region": "vietnam",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-HongKong-en-UK-2024_홍콩.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2023,
        "region": "hong_kong",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-AZJD-en-UK-0824_중국.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2024,
        "region": "china",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-SUISSE-IBG-KPT-EN-0624_스위스.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2022,
        "region": "switzerland",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-CARE-UK-EN-1125_영국.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2025,
        "region": "uk",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-IBG-FP-en-UK-1223_프랑스.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2024,
        "region": "france_benelux_monaco",
        "product_family": "regional",
    },
    {
        "path": DATA_DIR / "DOC-Global-IBG-EN-0524_남미.pdf",
        "doc_type": "benefit_guide",
        "doc_year": 2024,
        "region": "latin_america",
        "product_family": "regional",
    },
]

# 2. 텍스트 정리
def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# 3. PDF 페이지 읽기
# 이 함수가 반환하는 값의 타입이 List[tuple[int, str]] 라는 뜻.
# 즉, (페이지 번호, 페이지 텍스트) 형태의 리스트를 반환한다는 의미.
def read_pdf_pages(pdf_path: Path) -> List[tuple[int, str]]: 
    if not pdf_path.exists():
        print(f"[WARN] 파일이 없습니다: {pdf_path}")
        return []

    pages: List[tuple[int, str]] = []
    doc = fitz.open(pdf_path)

    try:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()

    return pages

# 4. 공통 메타데이터 생성
# 인덱싱된 각 문서 조각이 어떤 문서에서 왔는지,
# 어떤 유형의 문서인지, 어느 지역과 관련된 것인지 등의 정보를 담아줌.
def build_common_metadata(
    file_info: Dict[str, Any],
    source_name: str,
    page_num: int,
    chunk_idx: int | None = None,
    section: str | None = None,
) -> Dict[str, Any]:    # 리턴형식 : Dict[str, Any]
    metadata = {
        "source": source_name,
        "doc_type": file_info["doc_type"],
        "doc_year": file_info["doc_year"],
        "region": file_info["region"],
        "product_family": file_info["product_family"],
        "page": page_num,
        "insurer": "Allianz",
    }

    # chunk_idx가 있는 경우 : metadata에 chunk_idx 추가
    # chunk_idx는 같은 페이지 내에서 여러 개의 텍스트 조각이 나올 때, 각 조각을 구분하기 위한 인덱스입니다.
    if chunk_idx is not None:
        metadata["chunk_idx"] = chunk_idx

    # section이 있는 경우 : metadata에 section 추가
    # section은 문서 내에서 특정 섹션이나 제목을 나타내는 문자열입니다.
    # 예를 들어, "Coverage Details" 같은 섹션명이 될 수 있습니다.
    if section:
        metadata["section"] = section

    return metadata

# 5. 검색 보조 태그
REGION_ALIASES = {
    "global": ["global", "worldwide", "전세계", "글로벌", "공통"],
    "singapore": ["singapore", "싱가포르"],
    "dubai_northern_emirates": ["dubai", "northern emirates", "두바이", "북부에미리트", "uae", "아랍에미리트"],
    "lebanon": ["lebanon", "레바논"],
    "indonesia": ["indonesia", "인도네시아"],
    "vietnam": ["vietnam", "베트남"],
    "hong_kong": ["hong kong", "hk", "홍콩"],
    "china": ["china", "중국", "중화권"],
    "switzerland": ["switzerland", "suisse", "스위스"],
    "uk": ["uk", "united kingdom", "england", "britain", "영국"],
    "france_benelux_monaco": ["france", "benelux", "monaco", "프랑스", "모나코", "베네룩스"],
    "latin_america": ["latin america", "남미", "라틴아메리카"],
}

DOC_TYPE_ALIASES = {
    "benefit_guide": [
        "benefit guide", "coverage guide", "benefits", "혜택 가이드", "보장 안내", "보장", "혜택"
    ],
    "tob": [
        "table of benefits", "schedule of benefits", "benefit limits",
        "보장금액", "보장표", "한도표", "한도", "limit"
    ],
    "preauth_form": [
        "pre-authorisation form", "preauthorization form", "preauth form",
        "사전승인 신청서", "사전승인", "입원 전 승인", "직접청구 준비"
    ],
    "claim_form": [
        "claim form", "reimbursement form",
        "보험금 청구서", "청구서", "환급 청구", "사후 청구"
    ],
}

INSURANCE_SEARCH_TAGS = [
    "coverage", "covered", "benefit", "limit", "co-payment", "copay",
    "deductible", "waiting period", "exclusion", "outpatient", "inpatient",
    "maternity", "cancer", "chronic condition", "pre-existing condition",
    "pre-authorisation", "preauthorization", "planned hospitalisation",
    "direct billing", "claim", "reimbursement", "invoice", "receipt",
    "서류", "청구", "환급", "직접청구", "사전승인", "보장", "혜택",
    "한도", "면책", "제외사항", "외래", "입원", "출산", "기왕증"
]

# 검색 태그 빌드
def build_search_tags(file_info: Dict[str, Any]) -> str:
    region_aliases = REGION_ALIASES.get(file_info["region"], [])
    doc_type_aliases = DOC_TYPE_ALIASES.get(file_info["doc_type"], [])

    return "\n".join([
        "[search_tags]",
        f"region: {' | '.join(region_aliases)}",
        f"doc_type: {' | '.join(doc_type_aliases)}",
        f"product_family: {file_info['product_family']}",
        f"insurer: Allianz 알리안츠",
        "keywords: " + ", ".join(INSURANCE_SEARCH_TAGS),
    ])


def enrich_text_for_multilingual_search(text: str, file_info: Dict[str, Any]) -> str:
    tags = build_search_tags(file_info)
    return f"{text}\n\n{tags}"


# 6. Benefit Guide 청킹
def chunk_benefit_guide(
    pages: List[tuple[int, str]],
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []

    for page_num, text in pages:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

        for idx, para in enumerate(paragraphs):
            metadata = build_common_metadata(
                file_info=file_info,
                source_name=source_name,
                page_num=page_num,
                chunk_idx=idx,
            )
            docs.append(
                Document(
                    page_content=enrich_text_for_multilingual_search(para, file_info),
                    metadata=metadata,
                )
            )

    return docs


# 7. TOB 청킹
def chunk_tob(
    pages: List[tuple[int, str]],
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []

    for page_num, text in pages:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        buffer: List[str] = []
        chunk_idx = 0

        for line in lines:
            buffer.append(line)
            joined = " ".join(buffer)

            if len(joined) > 350:
                metadata = build_common_metadata(
                    file_info=file_info,
                    source_name=source_name,
                    page_num=page_num,
                    chunk_idx=chunk_idx,
                )
                docs.append(
                    Document(
                        page_content=enrich_text_for_multilingual_search(joined, file_info),
                        metadata=metadata,
                    )
                )
                buffer = []
                chunk_idx += 1

        if buffer:
            metadata = build_common_metadata(
                file_info=file_info,
                source_name=source_name,
                page_num=page_num,
                chunk_idx=chunk_idx,
            )
            docs.append(
                Document(
                    page_content=enrich_text_for_multilingual_search(" ".join(buffer), file_info),
                    metadata=metadata,
                )
            )

    return docs


# 8. Form 문서 청킹
def chunk_form(
    pages: List[tuple[int, str]],
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []

    for page_num, text in pages:
        blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 40]

        for idx, block in enumerate(blocks):
            metadata = build_common_metadata(
                file_info=file_info,
                source_name=source_name,
                page_num=page_num,
                chunk_idx=idx,
            )
            docs.append(
                Document(
                    page_content=enrich_text_for_multilingual_search(block, file_info),
                    metadata=metadata,
                )
            )

    return docs


# 9. 문서 빌드
def build_documents() -> List[Document]:
    all_docs: List[Document] = []

    for file_info in FILES:
        path: Path = file_info["path"]
        pages = read_pdf_pages(path)    # 파일에서 페이지 읽기

        if not pages:
            continue

        source_name = path.name
        doc_type = file_info["doc_type"]

        print(
            f"[INFO] 처리 중: {source_name} | type={doc_type} | "
            f"region={file_info['region']} | year={file_info['doc_year']}"
        )

        if doc_type == "benefit_guide":
            docs = chunk_benefit_guide(pages, source_name, file_info)
        elif doc_type == "tob":
            docs = chunk_tob(pages, source_name, file_info)
        elif doc_type in ["preauth_form", "claim_form"]:
            docs = chunk_form(pages, source_name, file_info)
        else:
            print(f"[WARN] 지원하지 않는 doc_type: {doc_type}")
            continue

        all_docs.extend(docs)

    return all_docs


# 10. Vector DB 저장
def index_documents(documents: List[Document]) -> None:
    if not documents:
        print("[WARN] 인덱싱할 문서가 없습니다.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectordb.persist()

    print(f"[DONE] Indexed {len(documents)} chunks into {DB_DIR}")


# 11. 실행
def main() -> None:
    documents = build_documents()
    index_documents(documents)


if __name__ == "__main__":
    main()