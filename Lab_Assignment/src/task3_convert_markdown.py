"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    from markitdown import MarkItDown
except Exception:  # pragma: no cover - optional dependency fallback
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"
CONVERSION_REPORT_PATH = METADATA_DIR / "conversion_report.json"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown() if MarkItDown else None

    report = []
    for old_output in output_dir.glob("*.md"):
        old_output.unlink()

    selected_files = select_legal_files_for_conversion(legal_dir)
    for filepath in selected_files:
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            text, method, ocr_required = extract_document_text(filepath, md)
            if not text.strip():
                text = (
                    f"# {filepath.stem}\n\n"
                    "Không trích xuất được text layer từ file này. "
                    "Cần OCR trước khi dùng làm context RAG.\n"
                )
                ocr_required = True
            output_path.write_text(text.strip() + "\n", encoding="utf-8")
            report.append({
                "input": str(filepath),
                "output": str(output_path),
                "method": method,
                "ocr_used": False,
                "ocr_required": ocr_required,
                "chars": len(text),
            })
            print(f"  ✓ Saved: {output_path} ({len(text)} chars, method={method}, ocr_required={ocr_required})")
    write_conversion_report("legal", report)


def select_legal_files_for_conversion(legal_dir: Path) -> list[Path]:
    """Prefer official DOC/DOCX for text extraction; keep PDFs as raw evidence."""
    files = [path for path in sorted(legal_dir.iterdir()) if path.is_file() and not path.name.startswith(".")]
    by_stem: dict[str, list[Path]] = {}
    for path in files:
        by_stem.setdefault(path.stem, []).append(path)

    selected = []
    for stem, candidates in sorted(by_stem.items()):
        docs = [path for path in candidates if path.suffix.lower() in {".doc", ".docx"}]
        pdfs = [path for path in candidates if path.suffix.lower() == ".pdf"]
        if docs:
            selected.extend(sorted(docs))
        else:
            selected.extend(sorted(pdfs))
    return selected


def extract_document_text(filepath: Path, md=None) -> tuple[str, str, bool]:
    text = ""
    if md is not None:
        try:
            text = md.convert(str(filepath)).text_content
            if text.strip():
                return text, "markitdown", False
        except Exception:
            pass

    suffix = filepath.suffix.lower()
    if suffix == ".docx":
        try:
            text = extract_docx_text(filepath)
            if text.strip():
                return text, "docx_xml", False
        except Exception:
            pass

    if suffix in {".doc", ".docx"}:
        text = extract_with_textutil(filepath)
        if text.strip():
            return text, "textutil", False

    if suffix == ".pdf":
        text = extract_pdf_text(filepath)
        if text.strip():
            return text, "pdftotext", False
        return "", "pdf_no_text_layer", True

    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    return text, "plain_text_fallback", False


def extract_with_textutil(filepath: Path) -> str:
    if not shutil.which("textutil"):
        return ""
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(filepath)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout if result.returncode == 0 else ""


def extract_pdf_text(filepath: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(filepath))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    if not shutil.which("pdftotext"):
        return ""
    result = subprocess.run(
        ["pdftotext", "-layout", str(filepath), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout if result.returncode == 0 else ""


def extract_docx_text(filepath: Path) -> str:
    """Extract plain text from a DOCX without requiring external dependencies."""
    with zipfile.ZipFile(filepath) as docx:
        xml_bytes = docx.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        if parts:
            paragraphs.append("".join(parts))
    return "\n\n".join(paragraphs)


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"
            header = f"# {data.get('title', 'Unknown')}\n\n"
            if data.get("publisher"):
                header += f"**Publisher:** {data.get('publisher')}\n"
            if data.get("published_date"):
                header += f"**Published:** {data.get('published_date')}\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            content = header + data.get("content_markdown", "")
            output_path.write_text(content.strip() + "\n", encoding="utf-8")
            report.append({
                "input": str(filepath),
                "output": str(output_path),
                "method": "json_content_markdown",
                "ocr_used": False,
                "ocr_required": False,
                "chars": len(content),
            })
            print(f"  ✓ Saved: {output_path}")
    write_conversion_report("news", report)


def write_conversion_report(section: str, records: list[dict]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    report = {"legal": [], "news": [], "updated_at": datetime.now(timezone.utc).isoformat()}
    if CONVERSION_REPORT_PATH.exists():
        try:
            report = json.loads(CONVERSION_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    report[section] = records
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    CONVERSION_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
