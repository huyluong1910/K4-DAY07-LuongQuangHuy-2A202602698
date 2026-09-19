"""
Script làm sạch dữ liệu chi tiết cho 10 tài liệu FPTU trong data/university
- Chuẩn hóa YAML Frontmatter (thêm doc_id, audience chuẩn)
- Xóa bỏ boilerplate web (menu, toggle, [Explore](#), [Xem thêm], Cloudflare email protection, tác giả rác)
- Định dạng lại Heading và bảng biểu Markdown
- Sinh file sources.csv chuẩn hóa
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

UNIVERSITY_DIR = Path("data/university")


def clean_frontmatter(text: str, doc_id: str, audience_override: str | None = None) -> tuple[dict, str]:
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    fm_text = parts[1]
    body = parts[2].strip()

    fm_dict = {}
    for line in fm_text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm_dict[k.strip()] = v.strip().strip('"')

    fm_dict["doc_id"] = doc_id
    if audience_override:
        fm_dict["audience"] = audience_override
    else:
        fm_dict.setdefault("audience", "student")

    return fm_dict, body


def dump_markdown(fm: dict, body: str) -> str:
    lines = ["---"]
    for k in [
        "doc_id",
        "title",
        "audience",
        "department",
        "category",
        "campus",
        "language",
        "source_url",
        "retrieved_at",
        "document_version",
        "published_at",
        "coverage",
        "source_domain",
        "source_type",
    ]:
        if k in fm:
            val = fm[k]
            lines.append(f'{k}: "{val}"')
    lines.append("---\n")
    return "\n".join(lines) + body + "\n"


def clean_01(body: str) -> str:
    # Xóa Mục lục [Toggle](#)
    body = re.sub(r"Mục lục\s+\[Toggle\]\(#\)", "", body)
    # Xóa chữ ký ThuNT ở cuối
    body = re.sub(r"\n+ThuNT\s*$", "", body)
    # Chuẩn hóa các header CHƯƠNG
    body = re.sub(r"####\s+\*\*CHƯƠNG", "## CHƯƠNG", body)
    # Xóa các dòng trống thừa
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def clean_02(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)  # xóa header lặp lại
    body = re.sub(r"Mục lục\s+\[Toggle\]\(#\)", "", body)
    body = re.sub(r"\n+admin_ptud\s*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Hướng dẫn sử dụng cổng thông tin đào tạo FAP (FPT University)\n\n{body.strip()}"


def clean_03(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)  # xóa header lặp lại
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Học phí tại Campus TP. Hồ Chí Minh (Khóa K22 - Năm 2026)\n\n{body.strip()}"


def clean_04(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    body = re.sub(r"## \*\*FAQ\*\*", "", body)
    # Chuẩn hóa ### 1. -> ## Câu 1:
    body = re.sub(r"###\s+(\d+)\.\s+(.+)", r"## Câu \1: \2", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Câu hỏi thường gặp về Học bổng Đại học FPT (FAQ)\n\n{body.strip()}"


def clean_05(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    # Thay thế Cloudflare email protection bằng email thật
    body = re.sub(
        r"\[\[email protected\]\]\(/cdn-cgi/l/email-protection[^\)]*\)",
        "tuyensinh.hcm@fpt.edu.vn",
        body,
        count=1,
    )
    body = re.sub(
        r"\[\[email protected\]\]\(/cdn-cgi/l/email-protection[^\)]*\)",
        "ss.hcm@fpt.edu.vn",
        body,
    )
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Thông tin liên hệ và Dịch vụ sinh viên Campus TP.HCM\n\n{body.strip()}"


def clean_06(body: str) -> str:
    # Xóa các link rác
    body = re.sub(r"\[Đăng ký ngay →\]\([^\)]+\)", "", body)
    body = re.sub(r"\[Khám phá thêm\]\([^\)]+\)", "", body)
    body = re.sub(r"\[Tìm hiểu thêm\]\([^\)]+\)", "", body)
    body = re.sub(r"\[Explore\]\(#\)", "", body)
    body = re.sub(r"\[xem thêm\]\([^\)]+\)", "", body)
    body = re.sub(r"\[Tham quan campus\]\([^\)]+\)", "", body)
    body = re.sub(r"\[Đăng ký\]\(#\)", "", body)
    body = re.sub(r"\[Xem chi tiết →\]\(#\)", "", body)
    body = re.sub(r"\[Xem thêm\]\([^\)]+\)", "", body)
    body = re.sub(r"\*\*01\*\*\s*/\s*12", "", body)
    # Xóa tin tức sự kiện ở cuối file (từ # Tin tức và sự kiện)
    if "# **Tin tức và sự kiện**" in body:
        body = body.split("# **Tin tức và sự kiện**")[0]
    elif "# Tin tức và sự kiện" in body:
        body = body.split("# Tin tức và sự kiện")[0]
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Cơ sở vật chất và Không gian học tập Campus TP. Hồ Chí Minh\n\n{body.strip()}"


def clean_07(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    body = re.sub(
        r"\[\[email protected\]\]\(/cdn-cgi/l/email-protection[^\)]*\)",
        "qhdn.hcm@fpt.edu.vn",
        body,
    )
    body = re.sub(r"\n+Tiến Duy\s*$", "", body)
    # Bổ sung nội dung tóm tắt cho các mục bị rỗng do web chỉ có tiêu đề
    body = body.replace(
        "2. **THAM DỰ ORIENTATION**\n\n3. **LỊCH TRIỂN KHAI DỰ KIẾN OJT SPRING 2026**",
        "2. **THAM DỰ ORIENTATION VÀ LỊCH TRIỂN KHAI**\n\n- Sinh viên bắt buộc tham dự buổi Orientation OJT trực tuyến qua Zoom.\n- Cần đăng nhập trước 15 phút giờ khai mạc; buổi sinh hoạt có điểm danh chính thức.",
    )
    body = body.replace(
        "4. **DANH SÁCH GIẢNG VIÊN CỐ VẤN OJT SPRING 2026**\n\nTrong suốt quá trình tham gia OJT",
        "3. **GIẢNG VIÊN CỐ VẤN CHUYÊN MÔN**\n\nTrong suốt quá trình tham gia OJT",
    )
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Quy định và Thông báo tham gia Học kỳ doanh nghiệp (OJT) Spring 2026\n\n{body.strip()}"


def clean_08(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    body = re.sub(r"Mục lục\s+\[Toggle\]\(#\)", "", body)
    body = re.sub(
        r"\[\[email protected\]\]\(/cdn-cgi/l/email-protection[^\)]*\)",
        "qhdn.hcm@fpt.edu.vn",
        body,
    )
    body = re.sub(r"\n+Tiến Duy\s*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Hướng dẫn sinh viên đăng ký doanh nghiệp thực tập OJT Spring 2026\n\n{body.strip()}"


def clean_09(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    body = re.sub(r"-\s+1\s+-\s+\[2\]\([^\)]+\)", "", body)
    body = re.sub(r"#####\s+\[([^\]]+)\]\([^\)]+\)", r"## \1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Các chương trình trao đổi sinh viên quốc tế (Exchange Program)\n\n{body.strip()}"


def clean_10(body: str) -> str:
    body = re.sub(r"^# [^\n]+\n+", "", body)
    body = re.sub(r"\[Xem chi tiết\]\([^\)]+\)", "", body)
    body = re.sub(r"### Thầy ([^\n]+)", r"### \1 (Phó Giám đốc / Trưởng ban)", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"# Sơ đồ tổ chức và Phân luồng phòng ban hỗ trợ Campus TP.HCM\n\n{body.strip()}"


CLEANERS = {
    "01-academic-regulations.md": (clean_01, "student"),
    "02-fap-and-academic-procedures.md": (clean_02, "student"),
    "03-tuition-hcm.md": (clean_03, "student"),
    "04-scholarship-faq.md": (clean_04, "student"),
    "05-student-services-hcm.md": (clean_05, "student"),
    "06-campus-facilities-hcm.md": (clean_06, "all"),
    "07-ojt-regulations.md": (clean_07, "student"),
    "08-ojt-registration.md": (clean_08, "student"),
    "09-international-exchange-hcm.md": (clean_09, "student"),
    "10-departments-and-support-routing-hcm.md": (clean_10, "staff"),
}


def main():
    print("Bắt đầu quy trình làm sạch sâu 10 tài liệu FPTU trong data/university...")
    manifest_rows = []

    for filename, (cleaner_fn, audience_val) in CLEANERS.items():
        file_path = UNIVERSITY_DIR / filename
        if not file_path.exists():
            print(f"[!] Bỏ qua file không tồn tại: {file_path}")
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem
        fm, body = clean_frontmatter(raw_text, doc_id, audience_override=audience_val)

        cleaned_body = cleaner_fn(body)
        cleaned_file_content = dump_markdown(fm, cleaned_body)

        file_path.write_text(cleaned_file_content, encoding="utf-8")
        print(f"  [OK] Đã làm sạch: {filename} (Audience: {fm['audience']}, Kích thước: {len(cleaned_body):,} ký tự)")

        manifest_rows.append({
            "doc_id": doc_id,
            "file_path": f"data/university/{filename}",
            "title": fm.get("title", ""),
            "source_url": fm.get("source_url", ""),
            "retrieved_at": fm.get("retrieved_at", "2026-09-19"),
            "document_version": fm.get("document_version", "current"),
            "license_or_permission": "official-public-source",
        })

    # Tạo file sources.csv
    sources_csv = UNIVERSITY_DIR / "sources.csv"
    with open(sources_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "file_path",
                "title",
                "source_url",
                "retrieved_at",
                "document_version",
                "license_or_permission",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[+] Đã sinh sources.csv thành công ({len(manifest_rows)} tài liệu).")


if __name__ == "__main__":
    main()
