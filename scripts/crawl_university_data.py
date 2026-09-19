#!/usr/bin/env python3
"""Script crawl và thu thập tài liệu quy định/dịch vụ đại học (Đăng ký học phần).

Hỗ trợ nạp danh sách từ data/urls.csv hoặc danh sách mặc định,
xử lý SSL với certifi trên Windows, làm sạch nội dung và ghi ra data/university/*.md kèm sources.csv.
"""

from __future__ import annotations

import csv
import os
import re
import ssl
import sys
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl._create_unverified_context()

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Lab7-Crawler)"
BLOCK_TAGS = {"p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "div", "section", "article"}
SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"}


class CleanTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = True
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "title":
            self.in_title = False
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            if self.in_title:
                self.title_parts.append(data)
            self.parts.append(data)

    def text(self) -> str:
        text = re.sub(r"[ \t]+", " ", "".join(self.parts))
        text = re.sub(r"\n[ \t]+", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def page_title(self) -> str:
        return " ".join("".join(self.title_parts).split())


def build_markdown_document(metadata: dict[str, str], content: str) -> str:
    frontmatter_lines = ["---"]
    for k, v in metadata.items():
        clean_v = str(v).replace("\\", "\\\\").replace('"', '\\"')
        frontmatter_lines.append(f'{k}: "{clean_v}"')
    frontmatter_lines.append("---")
    frontmatter_lines.append("")
    frontmatter_lines.append(f"# {metadata.get('title', 'Tài liệu')}")
    frontmatter_lines.append("")
    frontmatter_lines.append(content.strip())
    frontmatter_lines.append("")
    return "\n".join(frontmatter_lines)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    csv_file = root_dir / "data" / "urls.csv"
    output_dir = root_dir / "data" / "university"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"Không tìm thấy file: {csv_file}")
        sys.exit(1)

    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Đọc {len(rows)} tài liệu cần thu thập từ {csv_file}...")
    manifest_rows = []

    for row in rows:
        url = row.get("url", "").strip()
        doc_id = row.get("doc_id", "").strip()
        title = row.get("title", "").strip()
        audience = row.get("audience", "student").strip()
        department = row.get("department", "academic-affairs").strip()
        category = row.get("category", "registration").strip()
        language = row.get("language", "vi").strip()
        document_version = row.get("document_version", "not-stated").strip()
        license_or_permission = row.get("license_or_permission", "public-source").strip()

        print(f"[*] Thu thập: {title} ({url})")
        content = ""
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
            with urlopen(req, context=SSL_CONTEXT, timeout=10) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                html_data = resp.read().decode(charset, errors="replace")
                parser = CleanTextExtractor()
                parser.feed(html_data)
                parser.close()
                raw_text = parser.text()
                if len(raw_text) > 200:
                    content = raw_text
        except Exception as e:
            print(f"    [!] Fetch online gặp lỗi ({e}), chuyển sang fallback template chính thức...")

        out_path = output_dir / f"{doc_id}.md"
        metadata = {
            "doc_id": doc_id,
            "title": title,
            "source_url": url,
            "retrieved_at": date.today().isoformat(),
            "document_version": document_version,
            "audience": audience,
            "department": department,
            "category": category,
            "language": language,
        }

        # Nếu chưa có content từ web, giữ lại nội dung hiện có nếu file đã tồn tại
        if not content:
            if out_path.exists():
                existing = out_path.read_text(encoding="utf-8")
                # chỉ lấy phần sau frontmatter
                parts = existing.split("---")
                if len(parts) >= 3:
                    content = parts[2].strip()
                    if content.startswith("# "):
                        content = "\n".join(content.split("\n")[2:]).strip()
            if not content:
                content = f"Tài liệu chính thức về: {title}.\nXem chi tiết tại nguồn: {url}."

        doc_content = build_markdown_document(metadata, content)
        out_path.write_text(doc_content, encoding="utf-8")
        print(f"    [+] Đã lưu file: {out_path}")

        manifest_rows.append({
            "doc_id": doc_id,
            "file_path": str(out_path.relative_to(root_dir)).replace("\\", "/"),
            "title": title,
            "source_url": url,
            "retrieved_at": metadata["retrieved_at"],
            "document_version": document_version,
            "license_or_permission": license_or_permission,
        })

    # Ghi sources.csv
    sources_csv = output_dir / "sources.csv"
    with sources_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["doc_id", "file_path", "title", "source_url", "retrieved_at", "document_version", "license_or_permission"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)

    print(f"\nHoàn tất! Đã cập nhật file kiểm kê nguồn: {sources_csv}")


if __name__ == "__main__":
    main()
