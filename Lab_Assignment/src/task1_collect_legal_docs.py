"""Task 1 — collect official legal documents from source-of-truth portals."""

from __future__ import annotations

import json
import re
import shutil
from html import unescape
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"
SOURCES_PATH = METADATA_DIR / "sources.json"


LEGAL_SOURCES = [
    {
        "id": "bo_luat_hinh_su_2015",
        "title": "Luật số 100/2015/QH13 Bộ luật Hình sự",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2015-11-27",
        "effective_date": "2016-07-01",
        "source_url": "https://congbao.chinhphu.vn/van-ban/luat-so-100-2015-qh13-18442.htm",
        "preferred_slug": "bo-luat-hinh-su-2015",
    },
    {
        "id": "luat_phong_chong_ma_tuy_2021",
        "title": "Luật Phòng, chống ma túy 2021",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2021-03-30",
        "effective_date": "2022-01-01",
        "source_url": "https://congbao.chinhphu.vn/thuoc-tinh-van-ban-so-73-2021-qh14-33659?cbid=35651",
        "preferred_slug": "luat-phong-chong-ma-tuy-2021",
    },
    {
        "id": "nghi_dinh_57_2022_danh_muc_chat_ma_tuy",
        "title": "Nghị định 57/2022/NĐ-CP quy định các danh mục chất ma túy và tiền chất",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2022-08-25",
        "effective_date": "2022-08-25",
        "source_url": "https://congbao.chinhphu.vn/so-do-van-ban-so-57-2022-nd-cp-37734?cbid=41623",
        "preferred_slug": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy",
    },
    {
        "id": "nghi_dinh_105_2021_huong_dan_luat_phong_chong_ma_tuy",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng, chống ma túy",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2021-12-04",
        "effective_date": "2022-01-01",
        "source_url": "https://congbao.chinhphu.vn/so-do-van-ban-so-105-2021-nd-cp-34944?cbid=37821",
        "preferred_slug": "nghi-dinh-105-2021-huong-dan-luat-phong-chong-ma-tuy",
    },
    {
        "id": "nghi_dinh_116_2021_cai_nghien_quan_ly_sau_cai",
        "title": "Nghị định 116/2021/NĐ-CP về cai nghiện và quản lý sau cai nghiện ma túy",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2021-12-21",
        "effective_date": "2022-01-01",
        "source_url": "https://congbao.chinhphu.vn/noi-dung-van-ban-so-116-2021-nd-cp-36404?cbid=39135",
        "preferred_slug": "nghi-dinh-116-2021-cai-nghien-quan-ly-sau-cai",
    },
    {
        "id": "nghi_dinh_90_2024_sua_doi_danh_muc_chat_ma_tuy",
        "title": "Nghị định 90/2024/NĐ-CP sửa đổi danh mục chất ma túy và tiền chất",
        "publisher": "Công báo Chính phủ",
        "issued_date": "2024-07-17",
        "effective_date": "2024-07-17",
        "source_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-90-2024-nd-cp-42369/51055.htm",
        "preferred_slug": "nghi-dinh-90-2024-sua-doi-danh-muc-chat-ma-tuy",
    },
]


def setup_directory() -> None:
    """Create data/landing/legal/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Legal landing dir ready: {DATA_DIR}")


def archive_existing_data() -> None:
    files = [path for path in DATA_DIR.iterdir() if path.is_file() and not path.name.startswith(".")]
    if not files:
        return
    archive_dir = DATA_DIR.parent.parent / "demo_backup" / datetime.now().strftime("%Y%m%d_%H%M%S") / "legal"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.move(str(path), archive_dir / path.name)
    print(f"✓ Archived {len(files)} previous legal files to {archive_dir}")


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; Day08RAG/1.0; +https://localhost)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def discover_downloads(source_url: str) -> list[dict]:
    """Find Công báo download URLs from the canonical detail page."""
    response = requests.get(source_url, headers=_headers(), timeout=30)
    response.raise_for_status()
    html_text = response.text
    hrefs = re.findall(r'href=["\']([^"\']*(?:tai-ve-van-ban|api/download/stream)[^"\']*)["\']', html_text)
    downloads = []
    seen = set()
    for href in hrefs:
        url = urljoin(source_url, unescape(href).replace("&amp;", "&"))
        if url in seen:
            continue
        seen.add(url)
        lowered = url.lower()
        extension = (
            "pdf" if ".pdf" in lowered or "format=pdf" in lowered
            else "doc" if ".doc" in lowered or "format=doc" in lowered
            else "bin"
        )
        downloads.append({"url": url, "extension": extension})
    return downloads


def download_file(url: str, filepath: Path) -> None:
    response = requests.get(url, headers=_headers(), timeout=60)
    response.raise_for_status()
    if len(response.content) < 1024:
        raise RuntimeError(f"Downloaded file too small: {len(response.content)} bytes")
    filepath.write_bytes(response.content)


def update_sources_metadata(legal_records: list[dict]) -> None:
    existing = {"legal": [], "news": []}
    if SOURCES_PATH.exists():
        try:
            existing = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["legal"] = legal_records
    SOURCES_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Updated metadata: {SOURCES_PATH}")


def collect_legal_documents(archive_existing: bool = True) -> list[dict]:
    setup_directory()
    if archive_existing:
        archive_existing_data()

    records = []
    for source in LEGAL_SOURCES:
        print(f"Discovering downloads: {source['title']}")
        try:
            downloads = discover_downloads(source["source_url"])
        except Exception as exc:
            print(f"  ! Failed to discover downloads: {exc}")
            continue

        downloaded_files = []
        ext_counts: dict[str, int] = {}
        for item in downloads:
            if item["extension"] not in {"pdf", "doc"}:
                continue
            ext_counts[item["extension"]] = ext_counts.get(item["extension"], 0) + 1
            suffix = f"_part{ext_counts[item['extension']]}" if sum(1 for d in downloads if d["extension"] == item["extension"]) > 1 else ""
            filepath = DATA_DIR / f"{source['preferred_slug']}{suffix}.{item['extension']}"
            try:
                download_file(item["url"], filepath)
                downloaded_files.append({
                    "path": str(filepath),
                    "download_url": item["url"],
                    "file_type": item["extension"],
                    "bytes": filepath.stat().st_size,
                })
                print(f"  ✓ {filepath.name} ({filepath.stat().st_size} bytes)")
            except Exception as exc:
                print(f"  ! Failed {item['extension']}: {exc}")

        if downloaded_files:
            records.append({
                **source,
                "type": "legal",
                "source_of_truth": "official_government_gazette",
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "files": downloaded_files,
                "ocr_used": False,
                "ocr_note": "Official DOC/PDF downloaded from Công báo; OCR is only needed if text extraction fails.",
            })

    update_sources_metadata(records)
    return records


if __name__ == "__main__":
    collect_legal_documents()
