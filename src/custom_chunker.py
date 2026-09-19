import re
from .chunking import RecursiveChunker


class HeadingAwareContextChunker:
    """
    Tách tài liệu có cấu trúc theo Header Markdown (## Điều, ### Mục),
    đồng thời áp dụng kỹ thuật Context Injection (gắn tiền tố tiêu đề)
    để tối ưu hóa điểm số truy xuất ngữ nghĩa (Retrieval).
    """

    def __init__(self, max_chunk_size: int = 500, overlap: int = 0) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self._fallback_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # 1. Trích xuất tiêu đề cấp 1 (# Tiêu đề)
        doc_title_match = re.search(r"^#\s+(.+)$", text, re.M)
        doc_title = doc_title_match.group(1).strip() if doc_title_match else ""

        # 2. Tách tài liệu theo các header ## hoặc ###
        sections = [s.strip() for s in re.split(r"(?=\n#{2,3}\s+)", text) if s.strip()]
        chunks: list[str] = []

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Bỏ tiêu đề H1 nếu nó đứng hoàn toàn riêng lẻ
            if doc_title and sec_clean == f"# {doc_title}":
                continue

            # Trích xuất tiêu đề của section (## hoặc ###)
            sec_title_match = re.search(r"^#{2,3}\s+(.+)$", sec_clean, re.M)
            sec_title = sec_title_match.group(1).strip() if sec_title_match else ""

            # Tạo tiền tố ngữ cảnh (Context Prefix)
            context_prefix = (
                f"[{doc_title} > {sec_title}]: "
                if doc_title and sec_title
                else (f"[{doc_title}]: " if doc_title else "")
            )

            # Nếu section vừa vặn kích thước
            if len(sec_clean) + len(context_prefix) <= self.max_chunk_size:
                chunks.append(f"{context_prefix}{sec_clean}")
            else:
                # Nếu section dài hơn max_chunk_size, chia nhỏ bằng recursive nhưng vẫn gắn context prefix
                sub_parts = self._fallback_chunker.chunk(sec_clean)
                for part in sub_parts:
                    p = part.strip()
                    if p:
                        chunks.append(f"{context_prefix}{p}")

        return chunks if chunks else self._fallback_chunker.chunk(text)

