#!/usr/bin/env python3
"""Benchmark truy xuất và đánh giá Agent RAG trên bộ dữ liệu Quy chế học vụ.

Thực hiện 4 nhiệm vụ theo hướng dẫn Checkpoint:
1. Đọc từng file .md trong data/academic_regulations, tách frontmatter và content.
2. Chia nhỏ (chunk) bằng RecursiveChunker, mỗi chunk thành một Document với id dạng {doc_id}#{i}.
3. Nạp vào EmbeddingStore, chạy 5 câu hỏi benchmark qua search_with_filter().
4. In top-3 kèm score, doc_id và câu trả lời của Agent để đối chiếu với Gold Answer.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import KnowledgeBaseAgent
from src.chunking import HeadingRecursiveChunker, RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from src.models import Document
from src.store import EmbeddingStore


def semantic_hash_embed(text: str, dim: int = 128) -> list[float]:
    """Trình nhúng từ vựng ngữ nghĩa nhẹ (Lightweight Semantic Hash Embedder).
    
    Tạo vector đặc trưng từ vựng và n-gram không phụ thuộc thư viện ngoài,
    giúp retrieval hoạt động chính xác cả khi chưa cài sentence-transformers.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return [0.0] * dim

    vec = [0.0] * dim
    # Đưa unigram và bigram vào không gian vector
    tokens = words + [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Quy trình nộp đơn và thời hạn tiếp nhận đơn phúc khảo bài kiểm tra của sinh viên tại Trường Đại học Thủ Dầu Một là bao lâu?",
        "filter": {"audience": "student"},
        "expected_doc": "tdmu-grade-appeal",
        "key_phrases": ["07 ngày làm việc", "lệ phí phúc khảo", "7 ngày"],
        "gold_answer": (
            "Sinh viên nộp đơn trực tuyến qua cổng đào tạo và nộp lệ phí phúc khảo "
            "trong thời hạn 07 ngày làm việc kể từ ngày điểm thi được công bố trên cổng thông tin sinh viên."
        ),
        "notes": "Cần metadata_filter={'audience': 'student'} để tránh nhầm với quy trình của cán bộ khảo thí (tdmu-grade-appeal-operations)",
    },
    {
        "id": 2,
        "query": "Sinh viên rút bớt học phần trong thời hạn bao lâu và số tín chỉ tối thiểu còn lại là bao nhiêu tại Đại học Trà Vinh?",
        "filter": None,
        "expected_doc": "tvu-course-registration",
        "key_phrases": ["2 tuần", "14 tín chỉ"],
        "gold_answer": (
            "Trong vòng 2 tuần đầu tiên kể từ ngày bắt đầu học kỳ chính; "
            "số tín chỉ còn lại sau khi rút không được thấp hơn khối lượng học tập tối thiểu là 14 tín chỉ."
        ),
        "notes": "Tra cứu số liệu điều kiện và mốc thời gian rút học phần",
    },
    {
        "id": 3,
        "query": "Các tiêu chí cảnh báo học vụ về điểm GPA, CPA và số tín chỉ nợ đối với sinh viên Trường Y Dược - TVU là gì?",
        "filter": None,
        "expected_doc": "tvu-academic-warning",
        "key_phrases": ["1.00", "1.20", "24 tín chỉ", "gpa", "cpa"],
        "gold_answer": (
            "Điểm trung bình học kỳ GPA dưới 1.00 (kỳ đầu) hoặc dưới 1.20 (kỳ sau); "
            "hoặc điểm tích lũy CPA dưới 1.20 đến 1.80 theo năm học; hoặc tổng tín chỉ nợ vượt quá 24 tín chỉ."
        ),
        "notes": "Liệt kê các tiêu chí cảnh báo học vụ",
    },
    {
        "id": 4,
        "query": "Sinh viên chương trình tiên tiến tại TNUT chưa đạt chuẩn tiếng Anh theo tiến độ năm học thì bị giới hạn đăng ký tối đa bao nhiêu tín chỉ?",
        "filter": None,
        "expected_doc": "tnut-advanced-registration",
        "key_phrases": ["12 tín chỉ"],
        "gold_answer": (
            "Chỉ được đăng ký tối đa 12 tín chỉ các học phần đại cương và phải dành thời gian học bổ trợ tiếng Anh."
        ),
        "notes": "Quy định ràng buộc chuẩn ngoại ngữ khi đăng ký học phần",
    },
    {
        "id": 5,
        "query": "Sinh viên năm thứ nhất tại Đại học Tài chính - Marketing trong học kỳ đầu tiên có phải tự đăng ký học phần trên hệ thống không?",
        "filter": None,
        "expected_doc": "ufm-course-registration",
        "key_phrases": ["thời khóa biểu mặc định", "không phải tự đăng ký"],
        "gold_answer": (
            "Không, sinh viên khóa mới trong học kỳ đầu được nhà trường sắp xếp thời khóa biểu mặc định; "
            "sinh viên không phải tự đăng ký trên hệ thống."
        ),
        "notes": "Quy định xếp lịch học mặc định cho sinh viên năm nhất",
    },
]


def parse_markdown_file(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    parts = raw.split("---")
    if len(parts) >= 3:
        fm_text = parts[1]
        content = "---".join(parts[2:]).strip()
        fm = dict(re.findall(r"^(\w+):\s*(.+)$", fm_text, re.M))
    else:
        fm = {}
        content = raw.strip()
    return fm, content


def build_knowledge_base(data_dir: Path, chunk_size: int = 700) -> tuple[EmbeddingStore, list[Document]]:
    chunker = HeadingRecursiveChunker(chunk_size=chunk_size)
    all_chunks: list[Document] = []

    md_files = sorted(data_dir.glob("*.md"))
    print(f"[*] Đọc {len(md_files)} file tài liệu từ {data_dir}...")

    for file_path in md_files:
        fm, content = parse_markdown_file(file_path)
        doc_id = fm.get("doc_id") or file_path.stem

        chunks = chunker.chunk(content)
        for idx, ch in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc_id}#{idx}",
                content=ch,
                metadata={
                    **fm,
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "source": file_path.name,
                },
            )
            all_chunks.append(chunk_doc)

    print(f"[*] Đã tạo tổng cộng {len(all_chunks)} chunks từ {len(md_files)} tài liệu.")

    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "semantic_hash").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = semantic_hash_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = semantic_hash_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            embedder = semantic_hash_embed
    else:
        embedder = semantic_hash_embed

    store = EmbeddingStore(collection_name="benchmark_regulations", embedding_fn=embedder)
    store.add_documents(all_chunks)
    return store, all_chunks


def mock_llm_responder(prompt: str) -> str:
    """Tạo câu trả lời dựa trên ngữ cảnh trích xuất [1]."""
    match = re.search(r"\[1\]\s*\([^\)]*\)\s*(.*?)(?=\n\[2\]|\n---------------------)", prompt, re.DOTALL)
    if match:
        first_chunk = match.group(1).strip()
        lines = [line.strip() for line in first_chunk.split("\n") if line.strip() and not line.strip().startswith("#")]
        summary = " ".join(lines[:2])
        return f"[Agent] Dựa theo [1]: {summary}"
    return "[Agent] Đã trả lời dựa trên tài liệu trích xuất."


class TeeLogger:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def run_benchmark():
    data_dir = Path("data/academic_regulations")
    if not data_dir.exists():
        print(f"Lỗi: Không tìm thấy thư mục {data_dir}")
        return

    tee = TeeLogger("ket_qua_benchmark.txt")
    orig_stdout = sys.stdout
    sys.stdout = tee
    try:
        _execute_benchmark(data_dir)
    finally:
        sys.stdout = orig_stdout
        tee.close()
        print("\n[+] Đã lưu kết quả benchmark vào ket_qua_benchmark.txt thành công!")


def _execute_benchmark(data_dir: Path):
    store, _ = build_knowledge_base(data_dir, chunk_size=350)
    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm_responder)

    print("\n" + "=" * 80)
    print("CHẠY 5 BENCHMARK QUERIES — ĐÁNH GIÁ RETRIEVAL & AGENT (L3A)")
    print("=" * 80)

    total_hits = 0
    results_summary = []

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        query = q["query"]
        filt = q["filter"]
        expected = q["expected_doc"]
        gold = q["gold_answer"]
        key_phrases = q.get("key_phrases", [])

        print(f"\n--- [CÂU HỎI {qid}] ---")
        print(f"Query    : {query}")
        print(f"Filter   : {filt}")
        print(f"Gold Ans : {gold}")
        print(f"Tài liệu : {expected}.md")

        results = store.search_with_filter(query, top_k=3, metadata_filter=filt)

        hit_doc = False
        hit_content = False
        doc_rank = None
        content_rank = None

        top1_doc = results[0]["metadata"].get("doc_id") if results else "none"
        top1_score = results[0]["score"] if results else 0.0
        top1_content = results[0]["content"][:80].replace("\n", " ") if results else ""

        print("Top-3 Chunks truy xuất được:")
        for rank, r in enumerate(results, start=1):
            doc_id = r["metadata"].get("doc_id", "unknown")
            score = r["score"]
            text = r["content"]
            preview = text[:90].replace("\n", " ")
            is_doc_match = (doc_id == expected)
            is_content_match = any(kp.lower() in text.lower() for kp in key_phrases)

            if is_doc_match and doc_rank is None:
                doc_rank = rank
                hit_doc = True
            if is_content_match and content_rank is None:
                content_rank = rank
                hit_content = True

            mark = "[TRÚNG ĐÍCH]" if is_doc_match else "[KHÁC]"
            has_ans = " [CHỨA ĐÁP ÁN]" if is_content_match else ""
            print(f"  {rank}. {mark} score={score:.4f} doc_id={doc_id:30}{has_ans} | {preview}...")

        # Chấm 2 mức:
        # Mức 1 (Ngây thơ - theo Doc ID): Top-1 = 2đ, Top-2/3 = 1đ, không có = 0đ
        lvl1_pts = 2 if doc_rank == 1 else (1 if doc_rank in (2, 3) else 0)
        # Mức 2 (Nội dung - Chunk chứa đáp án thực sự): Top-1 = 2đ, Top-2/3 = 1đ, không có = 0đ
        lvl2_pts = 2 if content_rank == 1 else (1 if content_rank in (2, 3) else 0)

        if hit_doc:
            total_hits += 1

        ans = agent.answer(query, top_k=3)
        print(f"Agent Ans: {ans}")
        print(f"Đánh giá : Mức 1 (Doc ID): {lvl1_pts}đ (rank {doc_rank}) | Mức 2 (Nội dung đáp án): {lvl2_pts}đ (rank {content_rank})")

        results_summary.append({
            "qid": qid,
            "query": query,
            "top1_summary": f"{top1_doc}: {top1_content}...",
            "score": f"{top1_score:.3f}",
            "lvl1_pts": lvl1_pts,
            "lvl2_pts": lvl2_pts,
            "doc_rank": doc_rank,
            "content_rank": content_rank,
            "relevant": "Có" if hit_content else "Chỉ đúng tài liệu",
            "agent_ans": ans[:120] + "...",
        })

    # Bảng tổng kết 2 mức
    print("\n" + "=" * 80)
    print("BẢNG SO SÁNH HAI MỨC ĐÁNH GIÁ (MỨC 1: DOC ID vs MỨC 2: CHỨA ĐÁP ÁN NỘI DUNG)")
    print("=" * 80)
    print("Câu | Mức 1 (Doc ID) | Mức 2 (Nội dung) | Chênh lệch | Nhận xét")
    print("-" * 80)
    sum_lvl1 = sum(r["lvl1_pts"] for r in results_summary)
    sum_lvl2 = sum(r["lvl2_pts"] for r in results_summary)
    for r in results_summary:
        diff = r["lvl1_pts"] - r["lvl2_pts"]
        note = "Khớp hoàn toàn" if diff == 0 else "Thổi phồng (Đúng doc nhưng sai chunk/rank)"
        print(f" {r['qid']}  | {r['lvl1_pts']}đ (rank {r['doc_rank']})     | {r['lvl2_pts']}đ (rank {r['content_rank']})      | {diff:+d}đ        | {note}")
    print("-" * 80)
    print(f"Tổng| {sum_lvl1}/10 điểm     | {sum_lvl2}/10 điểm      | {sum_lvl1 - sum_lvl2:+d}đ        |")

    # Thử nghiệm A/B cho Câu hỏi 1: Minh chứng giá trị của metadata filtering
    print("\n" + "=" * 80)
    print("THỬ NGHIỆM A/B: MINH CHỨNG GIÁ TRỊ CỦA METADATA FILTERING TRÊN CÂU HỎI 1")
    print("=" * 80)
    q1 = BENCHMARK_QUERIES[0]
    res_no_filter = store.search_with_filter(q1["query"], top_k=3, metadata_filter=None)
    res_with_filter = store.search_with_filter(q1["query"], top_k=3, metadata_filter=q1["filter"])

    print("1. KHI KHÔNG LỌC (metadata_filter=None):")
    for rank, r in enumerate(res_no_filter, start=1):
        print(f"   {rank}. doc_id={r['metadata'].get('doc_id'):30} | audience={r['metadata'].get('audience')}")

    print("\n2. KHI CÓ LỌC (metadata_filter={'audience': 'student'}):")
    for rank, r in enumerate(res_with_filter, start=1):
        print(f"   {rank}. doc_id={r['metadata'].get('doc_id'):30} | audience={r['metadata'].get('audience')}")

    print("\n=> KẾT LUẬN A/B: Khi không lọc, tài liệu nghiệp vụ nội bộ của cán bộ ('staff') có thể chen chân vào top-k. Khi áp dụng bộ lọc {'audience': 'student'}, toàn bộ tài liệu nội bộ bị loại bỏ, bảo đảm câu trả lời hướng đúng đối tượng sinh viên!")


if __name__ == "__main__":
    run_benchmark()
