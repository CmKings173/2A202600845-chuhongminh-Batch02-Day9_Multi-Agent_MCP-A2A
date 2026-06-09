"""Task 2 — crawl real news articles with source metadata."""

from __future__ import annotations

import asyncio
import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
METADATA_DIR = Path(__file__).parent.parent / "data" / "metadata"
SOURCES_PATH = METADATA_DIR / "sources.json"


ARTICLE_URLS = [
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
    "https://cuoi.tuoitre.vn/chu-bin-bi-tam-giu-vi-lien-quan-den-mai-thi-thuy-20240606194643806.htm",
    "https://tuoitre.vn/on-ao-showbiz-viet-nam-2024-dam-vinh-hung-kien-cao-nam-em-bi-phat-doi-tu-cua-nam-thu-20241228091939857.htm",
    "https://tuoitre.vn/vu-miu-le-long-nhat-son-ngoc-minh-nghe-si-phai-giu-hinh-anh-chin-chu-tren-san-khau-lan-ngoai-doi-2026052112085492.htm",
    "https://tuoitre.vn/miu-le-bi-khoi-to-tam-giam-cong-ty-quan-ly-khang-dinh-khong-bao-che-20260516231928239.htm",
    "https://tuoitre.vn/ca-si-long-nhat-va-son-ngoc-minh-mua-ma-tuy-tu-dau-20260522101239371.htm",
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def archive_existing_data() -> None:
    files = [path for path in DATA_DIR.iterdir() if path.is_file() and not path.name.startswith(".")]
    if not files:
        return
    archive_dir = DATA_DIR.parent.parent / "demo_backup" / datetime.now().strftime("%Y%m%d_%H%M%S") / "news"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.move(str(path), archive_dir / path.name)
    print(f"✓ Archived {len(files)} previous news files to {archive_dir}")


def _headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (compatible; Day08RAG/1.0; +https://localhost)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _strip_tags(value: str) -> str:
    value = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>", " ", value)
    value = re.sub(r"(?is)<br\s*/?>", "\n", value)
    value = re.sub(r"(?is)</p>|</h[1-6]>|</div>|</li>", "\n", value)
    value = re.sub(r"(?is)<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n\n", value)
    return value.strip()


def _meta_content(html_text: str, prop: str) -> str:
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+name=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(prop)}["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{re.escape(prop)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.I)
        if match:
            return html.unescape(match.group(1)).strip()
    return ""


def _extract_article_body(html_text: str) -> str:
    candidates = []
    for pattern in [
        r'(?is)<article[^>]*>(.*?)</article>',
        r'(?is)<div[^>]+class=["\'][^"\']*(?:detail|content|article|fck|main)[^"\']*["\'][^>]*>(.*?)</div>',
    ]:
        candidates.extend(re.findall(pattern, html_text))
    if not candidates:
        candidates = [html_text]
    texts = [_strip_tags(candidate) for candidate in candidates]
    return max(texts, key=len).strip()


def _publisher_from_url(url: str) -> str:
    host = urlparse(url).netloc.replace("www.", "")
    if "vnexpress" in host:
        return "VnExpress"
    if "tuoitre" in host:
        return "Tuổi Trẻ"
    if "thanhnien" in host:
        return "Thanh Niên"
    return host


async def crawl_article(url: str) -> dict:
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    html_text = response.text
    title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
    title = _meta_content(html_text, "og:title") or (_strip_tags(title_match.group(1)) if title_match else "Unknown")
    description = _meta_content(html_text, "og:description") or _meta_content(html_text, "description")
    published = _meta_content(html_text, "article:published_time")
    body = _extract_article_body(html_text)
    if len(body) < 500:
        raise RuntimeError(f"Extracted article body too short: {len(body)} chars")

    publisher = _publisher_from_url(url)
    markdown = f"# {title}\n\n"
    markdown += f"**Publisher:** {publisher}\n"
    markdown += f"**Source:** {url}\n"
    if published:
        markdown += f"**Published:** {published}\n"
    markdown += "\n---\n\n"
    if description:
        markdown += f"> {description}\n\n"
    markdown += body

    return {
        "url": url,
        "publisher": publisher,
        "title": title,
        "published_date": published,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
        "ocr_used": False,
    }


def update_sources_metadata(news_records: list[dict]) -> None:
    existing = {"legal": [], "news": []}
    if SOURCES_PATH.exists():
        try:
            existing = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["news"] = news_records
    SOURCES_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Updated metadata: {SOURCES_PATH}")


async def crawl_all(archive_existing: bool = True) -> list[dict]:
    setup_directory()
    if archive_existing:
        archive_existing_data()

    records = []
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")
        records.append({
            "id": f"news_{i:02d}",
            "type": "news",
            "publisher": article.get("publisher"),
            "title": article.get("title"),
            "source_url": article.get("url"),
            "published_date": article.get("published_date"),
            "crawled_at": article.get("date_crawled"),
            "path": str(filepath),
            "ocr_used": False,
        })

    update_sources_metadata(records)
    return records


if __name__ == "__main__":
    asyncio.run(crawl_all())
