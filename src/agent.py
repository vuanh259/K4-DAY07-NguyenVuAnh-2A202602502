from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức hiện đang rỗng, không thể trả lời câu hỏi."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks = []
        for i, r in enumerate(results, start=1):
            source = (
                r.get("metadata", {}).get("source")
                or r.get("metadata", {}).get("doc_id")
                or r.get("id")
                or f"chunk_{i}"
            )
            content = r.get("content", "").strip()
            context_blocks.append(f"[{i}] (Nguồn: {source})\n{content}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"Dưới đây là thông tin ngữ cảnh được trích xuất từ cơ sở tri thức:\n"
            f"---------------------\n"
            f"{context_text}\n"
            f"---------------------\n"
            f"Yêu cầu:\n"
            f"1. Chỉ sử dụng thông tin trong ngữ cảnh trên để trả lời câu hỏi, tuyệt đối không suy đoán hay bịa đặt ngoài ngữ cảnh.\n"
            f"2. Trích dẫn số thứ tự nguồn [1], [2], ... tương ứng với thông tin bạn sử dụng để đảm bảo tính truy vết (Source Traceability).\n"
            f"3. Nếu ngữ cảnh không đủ thông tin để trả lời, hãy nêu rõ là không tìm thấy trong tài liệu.\n\n"
            f"Câu hỏi: {question}\n"
            f"Trả lời:"
        )
        return self.llm_fn(prompt)
