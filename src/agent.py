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

    def answer(
        self, question: str, top_k: int = 3, metadata_filter: dict | None = None
    ) -> str:
        if metadata_filter:
            results = self.store.search_with_filter(
                question, top_k=top_k, metadata_filter=metadata_filter
            )
        else:
            results = self.store.search(question, top_k=top_k)

        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_blocks: list[str] = []
        for idx, item in enumerate(results, start=1):
            doc_id = item.get("metadata", {}).get("doc_id", item.get("id", f"doc_{idx}"))
            context_blocks.append(f"[{idx}] (Nguồn: {doc_id}):\n{item['content']}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là một trợ lý học vụ thông minh và cẩn trọng.\n"
            "Hãy trả lời câu hỏi dưới đây CHỈ DỰA TRÊN ngữ cảnh được cung cấp. "
            "Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nêu rõ rằng quy định không đề cập.\n"
            "Khi trích xuất thông tin, hãy ghi rõ nguồn tham chiếu [1], [2] tương ứng.\n\n"
            f"--- NGỮ CẢNH ---\n{context_str}\n\n"
            f"--- CÂU HỎI ---\n{question}\n\n"
            "--- CÂU TRẢ LỜI ---"
        )

        return self.llm_fn(prompt)
