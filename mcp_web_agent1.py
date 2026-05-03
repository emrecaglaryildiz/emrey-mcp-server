# -*- coding: utf-8 -*-
"""
MCP Web Agent — Tam Cozum
=========================
Arama Zinciri : Brave Search API → DuckDuckGo (ücretsiz fallback)
Sayfa Okuma   : Jina Reader (hızlı) → nodriver (JS/Cloudflare fallback)

Gereksinimler:
    pip install "mcp[cli]" httpx beautifulsoup4 nodriver

Opsiyonel (stealth için):
    pip install playwright playwright-stealth
    playwright install chromium

Ortam Değişkeni (Brave için):
    export BRAVE_API_KEY="your_key_here"
    Brave ücretsiz API: https://api.search.brave.com
"""

import asyncio
import os
import re
import urllib.parse
import httpx
from mcp.server.fastmcp import FastMCP

# --- NODRIVER (opsiyonel, yoksa devre dışı) ---
try:
    import nodriver as uc
    NODRIVER_AVAILABLE = True
except ImportError:
    NODRIVER_AVAILABLE = False

# --- MCP SUNUCUSU ---
mcp = FastMCP("WebBrowserAgent")

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")

# ─────────────────────────────────────────────
# YARDIMCI: Metin Temizleme
# ─────────────────────────────────────────────

def clean_text(text: str, max_chars: int = 15000) -> str:
    """Fazla boşlukları ve tekrar eden satırları temizler, token limitine kırpar."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()[:max_chars]


# ─────────────────────────────────────────────
# ARAMA 1: Brave Search API
# ─────────────────────────────────────────────

async def search_brave(query: str, count: int = 10) -> list[dict] | None:
    """
    Brave Search API ile arama yapar.
    API key yoksa veya hata olursa None döner (fallback tetiklenir).
    """
    if not BRAVE_API_KEY:
        return None

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params={"q": query, "count": count, "search_lang": "tr"},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("web", {}).get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                    "source": "brave",
                }
                for r in results
            ]
    except Exception as e:
        print(f"[Brave] Hata: {e}")
        return None


# ─────────────────────────────────────────────
# ARAMA 2: DuckDuckGo (ücretsiz fallback)
# ─────────────────────────────────────────────

async def search_duckduckgo(query: str, count: int = 10) -> list[dict] | None:
    """
    DuckDuckGo HTML arama sayfasını parse eder.
    API key gerektirmez, tamamen ücretsizdir.
    """
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for result in soup.select(".result")[:count]:
            title_tag = result.select_one(".result__title a")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            href = title_tag.get("href", "")
            # DuckDuckGo redirect URL'lerini çöz
            if "uddg=" in href:
                try:
                    href = urllib.parse.unquote(
                        urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0]
                    )
                except Exception:
                    pass

            results.append({
                "title": title_tag.get_text(strip=True),
                "url": href,
                "description": snippet_tag.get_text(strip=True) if snippet_tag else "",
                "source": "duckduckgo",
            })

        return results if results else None

    except Exception as e:
        print(f"[DuckDuckGo] Hata: {e}")
        return None


# ─────────────────────────────────────────────
# SAYFA OKUMA 1: Jina Reader (hızlı, ücretsiz)
# ─────────────────────────────────────────────

async def fetch_jina(url: str) -> str | None:
    """
    Jina Reader API — URL'nin önüne r.jina.ai/ ekleyerek
    temiz Markdown döndürür. API key gerekmez.
    """
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/markdown, text/plain, */*",
    }
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(jina_url, headers=headers)
            resp.raise_for_status()
            if len(resp.text.strip()) < 200:
                return None  # Boş/bloke edilmiş sayfa → fallback
            return clean_text(resp.text)
    except Exception as e:
        print(f"[Jina] Hata: {e}")
        return None


# ─────────────────────────────────────────────
# SAYFA OKUMA 2: nodriver (JS/Cloudflare fallback)
# ─────────────────────────────────────────────

async def fetch_nodriver(url: str) -> str | None:
    """
    nodriver (undetected Chromium) ile sayfayı açar.
    Cloudflare ve JS-heavy siteler için fallback.
    """
    if not NODRIVER_AVAILABLE:
        return "nodriver yüklü değil. 'pip install nodriver' çalıştırın."

    try:
        browser = await uc.start(headless=True)
        page = await browser.get(url)
        await asyncio.sleep(4)  # JS render için bekle

        content = await page.get_content()
        await browser.stop()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            tag.decompose()

        return clean_text(soup.get_text(separator='\n', strip=True))

    except Exception as e:
        print(f"[nodriver] Hata: {e}")
        return None


# ─────────────────────────────────────────────
# MCP TOOL: web_search
# ─────────────────────────────────────────────

@mcp.tool()
async def web_search(query: str, count: int = 10) -> str:
    """
    Web'de arama yapar.
    Önce Brave Search API dener (BRAVE_API_KEY env var gerekli).
    Başarısız olursa DuckDuckGo ile devam eder.

    Args:
        query: Arama sorgusu
        count: Sonuç sayısı (varsayılan: 10)

    Returns:
        Arama sonuçlarının başlık + URL + açıklama listesi
    """
    print(f"[web_search] Sorgu: {query}")

    # 1. Brave dene
    results = await search_brave(query, count)
    provider = "Brave Search"

    # 2. Brave başarısız → DuckDuckGo
    if not results:
        print("[web_search] Brave başarısız, DuckDuckGo deneniyor...")
        results = await search_duckduckgo(query, count)
        provider = "DuckDuckGo"

    if not results:
        return "Arama başarısız oldu. Her iki sağlayıcı da sonuç döndürmedi."

    # Sonuçları formatla
    lines = [f"**{provider} Sonuçları** — '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   URL: {r['url']}")
        if r.get("description"):
            lines.append(f"   {r['description']}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MCP TOOL: web_fetch
# ─────────────────────────────────────────────

@mcp.tool()
async def web_fetch(url: str, force_browser: bool = False) -> str:
    """
    Verilen URL'nin içeriğini temiz Markdown olarak döndürür.
    Önce Jina Reader dener (hızlı), başarısız olursa nodriver kullanır.

    Args:
        url          : Okunacak web sayfasının adresi
        force_browser: True ise Jina'yı atlar, doğrudan nodriver kullanır

    Returns:
        Sayfanın temiz Markdown içeriği
    """
    print(f"[web_fetch] URL: {url}")

    # 1. Jina Reader dene (hızlı, ücretsiz)
    if not force_browser:
        content = await fetch_jina(url)
        if content:
            return f"*Kaynak: Jina Reader*\n\n{content}"
        print("[web_fetch] Jina başarısız, nodriver deneniyor...")

    # 2. nodriver fallback
    content = await fetch_nodriver(url)
    if content:
        return f"*Kaynak: nodriver (Headless Chromium)*\n\n{content}"

    return f"Sayfa okunamadı: {url}"


# ─────────────────────────────────────────────
# MCP TOOL: web_search_and_read
# ─────────────────────────────────────────────

@mcp.tool()
async def web_search_and_read(query: str, max_pages: int = 3) -> str:
    """
    Arama yapıp ilk N sonucun içeriğini okuyarak tek seferde döndürür.
    LLM'e hem arama sonuçlarını hem sayfa içeriklerini verir.

    Args:
        query    : Arama sorgusu
        max_pages: Okunacak maksimum sayfa sayısı (varsayılan: 3)

    Returns:
        Arama özeti + her sayfanın içeriği
    """
    print(f"[web_search_and_read] Sorgu: {query}, max_pages: {max_pages}")

    # Önce arama yap
    results = await search_brave(query, max_pages)
    provider = "Brave Search"
    if not results:
        results = await search_duckduckgo(query, max_pages)
        provider = "DuckDuckGo"

    if not results:
        return "Arama sonucu bulunamadı."

    output = [f"## Arama Sonuçları ({provider}): '{query}'\n"]

    # Her sayfayı oku
    for i, r in enumerate(results[:max_pages], 1):
        output.append(f"---\n### {i}. {r['title']}")
        output.append(f"**URL:** {r['url']}")
        if r.get("description"):
            output.append(f"**Özet:** {r['description']}\n")

        # Sayfa içeriğini çek
        content = await fetch_jina(r["url"])
        if not content:
            content = await fetch_nodriver(r["url"])

        if content:
            # Her sayfa için 4000 karakter yeter
            output.append(f"**İçerik:**\n{content[:4000]}")
        else:
            output.append("*(Sayfa içeriği okunamadı)*")

        output.append("")

    return "\n".join(output)


# ─────────────────────────────────────────────
# BAŞLANGIÇ
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Windows cp1254 encoding sorununu coz
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("MCP Web Agent baslatiliyor...")
    print(f"  Brave API : {'[OK] Aktif' if BRAVE_API_KEY else '[--] Yok (DuckDuckGo kullanilacak)'}")
    print(f"  nodriver  : {'[OK] Yuklu' if NODRIVER_AVAILABLE else '[--] Yuklu degil (pip install nodriver)'}")
    print(f"  Jina      : [OK] Aktif (API key gerektirmez)")
    mcp.run()
