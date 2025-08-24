# server.py
from __future__ import annotations
import os
import re
import html
import io
import base64
import requests
from typing import List, Dict
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from mcp.server.fastmcp import FastMCP



mcp = FastMCP("WebTools")

# --- Yardımcılar -------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = html.unescape(text)
  
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Araçlar (Tools) --------------------------------------------------------

@mcp.tool()
def web_search(query: str, max_results: int = 5, region: str = "tr-tr", safesearch: str = "moderate") -> List[Dict[str, str]]:
    """Web araması yapar (API anahtarsız). Sonuç: title, url, snippet.
    region: ddg bölge kodu (ör. "tr-tr"). safesearch: off|moderate|strict
    """
    if not query or not query.strip():
        return []

    out: List[Dict[str, str]] = []
    # DuckDuckGo Search (no key)
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results, region=region, safesearch=safesearch):
            out.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo"
            })
    return out


@mcp.tool()
def http_get(url: str, max_chars: int = 8000, timeout: float = 12.0, user_agent: str = "MCP-WebTools/1.0") -> Dict[str, str]:
    """URL'den içerik çeker ve sadeleştirilmiş metin döner.
    Çok büyük sayfaları keser (max_chars)."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http(s) URLs are allowed")

    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Basit ayrıştırma (istenirse trafilatura/readability ile iyileştirilebilir)
    soup = BeautifulSoup(resp.text, "html.parser")
    title = _clean_text(soup.title.text) if soup.title else ""
    # script/style kaldır
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = _clean_text(soup.get_text(" "))
    if len(text) > max_chars:
        text = text[:max_chars] + " …(truncated)"

    return {
        "url": url,
        "title": title,
        "text": text
    }


@mcp.tool()
def http_get_chunks(url: str, chunk_size: int = 3000, overlap: int = 200, timeout: float = 12.0) -> Dict[str, List[str]]:
    """Uzun HTML içeriklerini parçalayarak döner (chunk'lara ayırır)."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Only http(s) URLs are allowed")

    headers = {"User-Agent": "MCP-WebTools/1.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = _clean_text(soup.get_text(" "))

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  

    return {
        "url": url,
        "chunks": chunks,
        "chunk_count": len(chunks)
    }


@mcp.tool()
def text_to_pdf(text: str, title: str = "Document") -> Dict[str, str]:
    """Verilen metni PDF'e dönüştürür ve aynı klasöre kaydeder."""
    
    FONT_PATH = "C:\mcp\dejavu\DejaVuSans.ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
    
    filename = f"{title.replace(' ', '_')}.pdf"
    filepath = os.path.join(os.path.dirname(__file__), filename)

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    c.setFont("DejaVu", 14)
    c.drawString(50, height - 50, title)

    c.setFont("DejaVu", 11)
    y = height - 80
    for line in text.split("\n"):
        wrapped = [line[i:i+90] for i in range(0, len(line), 90)]
        for part in wrapped:
            if y < 50:  # sayfa sonu
                c.showPage()
                c.setFont("DejaVu", 11)
                y = height - 50
            c.drawString(50, y, part)
            y -= 14

    c.save()

    return {
        "filename": filename,
        "saved_to": filepath,
        "status": "ok"
    }


if __name__ == "__main__":
    # stdio üzerinden çalıştır (Claude Desktop, Cursor, Inspector ile uyumlu)
    mcp.run()
