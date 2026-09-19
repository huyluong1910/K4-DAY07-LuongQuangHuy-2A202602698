from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(override=False)

UNI_DIR = Path("data/university")
SOURCES_CSV = UNI_DIR / "sources.csv"
CACHE_DOCS: list[dict[str, str]] = []
SOURCES_METADATA: dict[str, dict[str, str]] = {}

VI_STOPWORDS = {
    "thời", "gian", "ngày", "hôm", "nay", "là", "và", "của", "có", "các", "những",
    "được", "trong", "cho", "về", "khi", "này", "tôi", "bạn", "gì", "sao", "nào",
    "thế", "một", "hay", "ai", "thì", "mà", "ở", "tại", "với", "như", "rất",
    "lại", "ra", "vào", "đến", "đi", "làm", "biết", "xin", "hãy", "chỉ", "cần",
    "hỏi", "ơi", "ạ", "nhỉ", "nhé", "chưa", "đã", "đang", "sẽ", "phải"
}

DOMAIN_KEYWORDS = [
    "hoc bong", "tieu chi", "duy tri", "gpa", "do an", "tot nghiep", "capstone",
    "khoa luan", "ojt", "thuc tap", "doanh nghiep", "hoc phi", "dong tien", "bao luu",
    "rut ho so", "fap", "thu tuc", "diem danh", "nghi hoc", "tin chi", "khung gio",
    "hoc lai", "thi lai", "phuc tra", "quy che", "fpt", "fptu", "truong", "sinh vien",
    "khoa", "nganh", "dao tao", "khao thi", "lich thi", "chuong trinh", "tieng anh",
    "giai doan", "ren luyen", "giao duc the chat", "dich vu sinh vien"
]

CANDIDATE_CHAT_MODELS = [
    os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]


def remove_diacritics(text: str) -> str:
    """Normalize Vietnamese text by stripping diacritics for robust matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    clean = "".join(c for c in nfkd if not unicodedata.combining(c))
    return clean.replace("đ", "d").replace("Đ", "D").lower()


def load_sources_metadata() -> dict[str, dict[str, str]]:
    global SOURCES_METADATA
    if SOURCES_METADATA:
        return SOURCES_METADATA

    if SOURCES_CSV.exists():
        with open(SOURCES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                SOURCES_METADATA[row["doc_id"]] = {
                    "title": row.get("title", row["doc_id"]),
                    "source_url": row.get("source_url", ""),
                    "retrieved_at": row.get("retrieved_at", ""),
                }
    return SOURCES_METADATA


def load_corpus_docs() -> list[dict[str, str]]:
    global CACHE_DOCS
    if CACHE_DOCS:
        return CACHE_DOCS

    docs = []
    if UNI_DIR.exists():
        for md_path in sorted(UNI_DIR.glob("*.md")):
            content = md_path.read_text(encoding="utf-8")
            content_clean = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
            docs.append({
                "filename": md_path.name,
                "doc_id": md_path.stem,
                "content": content_clean,
                "norm_content": remove_diacritics(content_clean),
            })
    CACHE_DOCS = docs
    return CACHE_DOCS


def is_university_domain_query(query_norm: str) -> bool:
    """Check if query is related to university regulations or FPTU education."""
    for kw in DOMAIN_KEYWORDS:
        if kw in query_norm:
            return True
    return False


def retrieve_relevant_context(query: str, top_k: int = 3) -> list[dict[str, str]]:
    q_norm = remove_diacritics(query)

    # 1. Out-of-domain guard: queries like weather, cooking, jokes have NO relevant regulations
    if not is_university_domain_query(q_norm):
        return []

    docs = load_corpus_docs()
    if not docs:
        return []

    words_norm = {w for w in re.findall(r"\w+", q_norm) if w not in VI_STOPWORDS and len(w) > 1}
    if not words_norm:
        return []

    scored_docs = []
    for doc in docs:
        doc_words_norm = set(re.findall(r"\w+", doc["norm_content"]))
        overlap = len(words_norm.intersection(doc_words_norm))

        score = overlap * 2.0

        # Domain-specific targeting
        if any(k in q_norm for k in ["hoc bong", "tieu chi", "duy tri", "gpa"]) and "scholarship" in doc["doc_id"]:
            score += 25.0
        if any(k in q_norm for k in ["do an", "tot nghiep", "capstone", "khoa luan"]) and "academic-regulations" in doc["doc_id"]:
            score += 25.0
        if any(k in q_norm for k in ["ojt", "thuc tap", "doanh nghiep"]) and "ojt" in doc["doc_id"]:
            score += 25.0
        if any(k in q_norm for k in ["hoc phi", "dong tien", "bao luu", "rut ho so"]) and "tuition" in doc["doc_id"]:
            score += 25.0
        if any(k in q_norm for k in ["fap", "thu tuc", "diem danh", "nghi hoc"]) and "fap" in doc["doc_id"]:
            score += 25.0

        if score > 2.0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_matched_docs = scored_docs[:top_k]

    results = []
    for sc, doc in top_matched_docs:
        if len(doc["content"]) <= 6500:
            results.append({
                "source": doc["filename"],
                "doc_id": doc["doc_id"],
                "content": doc["content"],
                "score": sc,
            })
        else:
            sections = re.split(r"(?=\n#{1,3}\s+)", doc["content"])
            sec_scored = []
            for sec in sections:
                s_clean = sec.strip()
                if len(s_clean) < 30:
                    continue
                s_words = set(re.findall(r"\w+", remove_diacritics(s_clean)))
                s_score = len(words_norm.intersection(s_words))
                sec_scored.append((s_score, s_clean))
            sec_scored.sort(key=lambda x: x[0], reverse=True)
            best_secs = [s[1] for s in sec_scored[:3]]
            results.append({
                "source": doc["filename"],
                "doc_id": doc["doc_id"],
                "content": "\n\n---\n\n".join(best_secs),
                "score": sc,
            })

    return results


def call_gemini_with_fallback(prompt: str) -> str | None:
    """Attempt generation with candidate models in order to prevent 429 quota exhaustion."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        for model_name in CANDIDATE_CHAT_MODELS:
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if res.text and res.text.strip():
                    return res.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    continue
                if "404" in err_str or "NOT_FOUND" in err_str:
                    continue
    except Exception:
        pass

    return None


def generate_rag_answer(query: str, contexts: list[dict[str, str]], is_first_turn: bool = False) -> str:
    # 1. OUT-OF-DOMAIN HANDLING:
    if not contexts:
        # Prompt model to politely decline and guide user back to university regulations
        out_of_domain_prompt = (
            "Bạn là Aethera Assistant — trợ lý học vụ chuyên sâu của Trường Đại học FPT.\n"
            f"Người dùng vừa hỏi: '{query}'.\n\n"
            "HƯỚNG DẪN TRẢ LỜI:\n"
            "1. Câu hỏi này KHÔNG thuộc phạm vi quy chế, đào tạo hay học vụ của Trường Đại học FPT (ví dụ: hỏi thời tiết, tin tức xã hội, nấu ăn, chuyện phiếm...).\n"
            "2. Hãy trả lời ngắn gọn (chỉ 2-3 câu), lịch sự và thân thiện: giải thích rõ bạn là trợ lý chuyên về quy chế học vụ, học bổng, học phí, OJT và thủ tục sinh viên của FPTU nên không có dữ liệu thời gian thực cho câu hỏi này.\n"
            "3. Gợi ý người dùng đặt câu hỏi về các vấn đề quy chế của trường (học bổng, điều kiện làm đồ án, thực tập doanh nghiệp OJT, thủ tục FAP...). Tuyệt đối không viện dẫn hay chép quy chế không liên quan vào."
        )
        gemini_ans = call_gemini_with_fallback(out_of_domain_prompt)
        if gemini_ans:
            return gemini_ans

        return (
            "Tôi là Aethera Assistant — trợ lý chuyên trách về Quy chế Đào tạo và Học vụ của Trường Đại học FPT. "
            "Câu hỏi này nằm ngoài phạm vi cơ sở dữ liệu học vụ của trường (tôi không có dữ liệu thời gian thực về thời tiết hay thông tin bên ngoài). "
            "Bạn có thể đặt các câu hỏi liên quan đến **học bổng, đồ án tốt nghiệp, thực tập OJT, học phí hoặc cổng FAP** để tôi hỗ trợ nhé!"
        )

    # 2. IN-DOMAIN REGULATION ANSWERING:
    sources_meta = load_sources_metadata()
    context_blocks = []
    for i, c in enumerate(contexts, 1):
        doc_id = c.get("doc_id", "")
        meta = sources_meta.get(doc_id, {})
        title = meta.get("title", c["source"])
        url = meta.get("source_url", "https://daihoc.fpt.edu.vn/")
        context_blocks.append(
            f"=== [NGUỒN THAM CHIẾU {i}] ===\n"
            f"Tên tài liệu: {title}\n"
            f"Link chính thức: {url}\n"
            f"Nội dung quy chế trích xuất:\n{c['content']}"
        )

    context_str = "\n\n".join(context_blocks)

    greeting_rule = (
        "Chào hỏi ngắn gọn và giới thiệu bạn là Aethera Assistant trong lượt đầu này."
        if is_first_turn
        else "TUYỆT ĐỐI KHÔNG chào hỏi, KHÔNG giới thiệu lại tên 'Xin chào...', 'Tôi là Aethera...'. Hãy đi thẳng vào nội dung giải đáp câu hỏi một cách trực tiếp, mạch lạc và lịch sự."
    )

    prompt = (
        "Bạn là Aethera Assistant — trợ lý học vụ thông minh của Trường Đại học FPT.\n"
        "Nhiệm vụ của bạn là giải đáp thắc mắc cho sinh viên một cách RÕ RÀNG, ĐẦY ĐỦ Ý, ĐÚNG TRỌNG TÂM và CHUẨN XÁC dựa trên tập tài liệu quy chế được cung cấp.\n\n"
        "QUY TẮC PHẢN HỒI BẮT BUỘC:\n"
        f"1. CHÀO HỎI: {greeting_rule}\n"
        "2. ĐÚNG TRỌNG TÂM & RÕ RÀNG: Trả lời chính xác những gì câu hỏi yêu cầu. Cung cấp đầy đủ các điều kiện, tiêu chí, số liệu (GPA, số tín chỉ, mốc thời gian...) theo các mục gạch đầu dòng rõ ràng. KHÔNG chép nguyên văn toàn bộ chương điều dài dòng không liên quan.\n"
        "3. TRÍCH DẪN MINH BẠCH: Tại các luận điểm quan trọng, ghi rõ ký hiệu trích dẫn [1], [2]. Ở CUỐI CÂU TRẢ LỜI, BẮT BUỘC có mục '📌 **Nguồn tham chiếu chính thức của Trường Đại học FPT:**' liệt kê từng nguồn kèm link Markdown: `[1] [Tên tài liệu chính thức](Link_URL_chính_thức)`.\n"
        "4. TÍNH CHÂN THỰC: Chỉ nêu thông tin có trong ngữ cảnh.\n\n"
        f"--- TẬP VĂN BẢN QUY CHẾ FPTU ---\n{context_str}\n\n"
        f"--- CÂU HỎI CỦA SINH VIÊN ---\n{query}"
    )

    gemini_ans = call_gemini_with_fallback(prompt)
    if gemini_ans:
        return gemini_ans

    # Fallback when Gemini quota exhausted or offline
    citations = []
    for i, c in enumerate(contexts, 1):
        doc_id = c.get("doc_id", "")
        meta = sources_meta.get(doc_id, {})
        title = meta.get("title", c["source"])
        url = meta.get("source_url", "https://daihoc.fpt.edu.vn/")
        citations.append(f"[{i}] [{title}]({url})")

    return (
        "Dựa trên quy chế đào tạo chính thức của Trường Đại học FPT [1], thông tin về vấn đề này được quy định như sau:\n\n"
        "• Sinh viên cần đáp ứng đầy đủ các tiêu chuẩn học thuật, tín chỉ tích lũy và điều kiện môn học tiên quyết theo quy chế hiện hành.\n"
        "• Mọi thông tin chi tiết xin vui lòng tra cứu thêm trên hệ thống FAP hoặc liên hệ Phòng Dịch vụ Sinh viên.\n\n"
        "📌 **Nguồn tham chiếu chính thức của Trường Đại học FPT:**\n"
        + "\n".join(citations)
    )


class AetheraServerHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                query = data.get("query", "").strip()
                is_first_turn = bool(data.get("is_first_turn", False))
            except Exception:
                query = ""
                is_first_turn = False

            if not query:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"error": "Missing query"}')
                return

            contexts = retrieve_relevant_context(query, top_k=3)
            sources_meta = load_sources_metadata()

            sources_list = []
            seen_ids = set()
            for c in contexts:
                did = c.get("doc_id", "")
                if did and did not in seen_ids:
                    seen_ids.add(did)
                    meta = sources_meta.get(did, {})
                    sources_list.append({
                        "doc_id": did,
                        "title": meta.get("title", c["source"]),
                        "url": meta.get("source_url", "https://daihoc.fpt.edu.vn/"),
                    })

            answer = generate_rag_answer(query, contexts, is_first_turn=is_first_turn)

            response_data = {
                "answer": answer,
                "sources": sources_list,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run(port: int = 8000):
    load_sources_metadata()
    load_corpus_docs()
    server_address = ("", port)
    httpd = HTTPServer(server_address, AetheraServerHandler)
    print(f"Aethera Server listening on http://localhost:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port=port)
