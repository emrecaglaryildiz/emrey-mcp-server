#python3.12<=
#pip install mcp httpx beautifulsoup4 markdownify nodriver

"""
WebBrowserAgent MCP Server
--------------------------
Arama   : Brave Search API  → DuckDuckGo (fallback)
Okuma   : Jina Reader        → nodriver  (fallback, JS-heavy siteler)

Kurulum:
  pip install mcp httpx beautifulsoup4 markdownify nodriver

Env:
  BRAVE_API_KEY=your_key_here   (https://api.search.brave.com - 2000/ay ücretsiz)
"""

import asyncio
import os
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md_converter
from mcp.server.fastmcp import FastMCP

# ─── nodriver (opsiyonel) ────────────────────────────────────────────────────
try:
    import nodriver as uc
    NODRIVER_AVAILABLE = True
except ImportError:
    NODRIVER_AVAILABLE = False

# ─── Sabitler ────────────────────────────────────────────────────────────────
BRAVE_API_KEY  = os.getenv("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDG_ENDPOINT   = "https://html.duckduckgo.com/html/"
JINA_READER    = "https://r.jina.ai/"
MAX_CHARS      = 15000

mcp = FastMCP("WebBrowserAgent")


# ═══════════════════════════════════════════════════════════════════════════════
#  ARAÇ 1: web_search  —  Brave → DuckDuckGo fallback
# ═══════════════════════════════════════════════════════════════════════════════

async def _brave_search(query: str, num: int) -> str | None:
    """Brave Search API ile arama. Başarısız olursa None döner."""
    if not BRAVE_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                BRAVE_ENDPOINT,
                headers={
                    "X-Subscription-Token": BRAVE_API_KEY,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                },
                params={"q": query, "count": num, "search_lang": "tr", "country": "TR"},
            )
            r.raise_for_status()
            data = r.json()
            items = data.get("web", {}).get("results", [])
            if not items:
                return None
            lines = []
            for i, item in enumerate(items, 1):
                lines.append(
                    f"{i}. **{item.get('title', 'Başlık yok')}**\n"
                    f"   URL: {item.get('url', '')}\n"
                    f"   {item.get('description', '').strip()}"
                )
            return "\n\n".join(lines)
    except Exception as e:
        print(f"[Brave] hata: {e}")
        return None


async def _ddg_search(query: str, num: int) -> str:
    """DuckDuckGo HTML arama (API key gerektirmez)."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                DDG_ENDPOINT,
                data={"q": query, "kl": "tr-tr"},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                follow_redirects=True,
            )
        soup = BeautifulSoup(r.text, "html.parser")
        results = soup.select(".result__body")[:num]
        if not results:
            return "DuckDuckGo'dan sonuç alınamadı."
        lines = []
        for i, result in enumerate(results, 1):
            title_el = result.select_one(".result__title")
            url_el   = result.select_one(".result__url")
            snip_el  = result.select_one(".result__snippet")
            title = title_el.get_text(strip=True) if title_el else "Başlık yok"
            url   = url_el.get_text(strip=True)   if url_el   else ""
            snip  = snip_el.get_text(strip=True)  if snip_el  else ""
            lines.append(f"{i}. **{title}**\n   URL: {url}\n   {snip}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"DuckDuckGo hatası: {e}"


@mcp.tool()
async def web_search(query: str, num_results: int = 10) -> str:
    """
    Web'de arama yapar.
    Önce Brave Search API dener (BRAVE_API_KEY varsa),
    başarısız olursa otomatik DuckDuckGo'ya geçer.
    Bot koruması yoktur.
    """
    print(f"[Search] Sorgu: {query}")

    # Brave dene
    result = await _brave_search(query, num_results)
    if result:
        print("[Search] Brave Search → başarılı")
        return f"[Kaynak: Brave Search]\n\n{result}"

    # Fallback: DuckDuckGo
    print("[Search] Brave başarısız → DuckDuckGo fallback")
    result = await _ddg_search(query, num_results)
    return f"[Kaynak: DuckDuckGo (fallback)]\n\n{result}"


# ═══════════════════════════════════════════════════════════════════════════════
#  ARAÇ 2: web_scrape  —  Jina Reader → nodriver fallback
# ═══════════════════════════════════════════════════════════════════════════════

async def _jina_read(url: str) -> str | None:
    """
    Jina Reader ile URL'yi LLM-dostu Markdown'a çevirir.
    API key olmadan 20 RPM (ücretsiz, kurulum yok).
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(
                f"{JINA_READER}{url}",
                headers={
                    "Accept": "text/plain",
                    "User-Agent": "Mozilla/5.0",
                    "X-Return-Format": "markdown",
                },
                follow_redirects=True,
            )
            if r.status_code == 200 and len(r.text.strip()) > 100:
                return r.text[:MAX_CHARS]
            return None
    except Exception as e:
        print(f"[Jina] hata: {e}")
        return None


async def _nodriver_read(url: str) -> str:
    """
    nodriver ile JS-heavy sayfaları okur (Cloudflare vb. aşar).
    Jina başarısız olursa devreye girer.
    """
    if not NODRIVER_AVAILABLE:
        return "nodriver kurulu değil. `pip install nodriver` ile yükleyin."
    try:
        browser = await uc.start(headless=True)
        page    = await browser.get(url)
        await asyncio.sleep(3)
        content = await page.get_content()
        await browser.stop()

        soup = BeautifulSoup(content, "html.parser")
        for el in soup(["script","style","nav","footer","header","iframe","noscript"]):
            el.decompose()
        clean = md_converter(str(soup), strip=["a"])
        return clean[:MAX_CHARS]
    except Exception as e:
        return f"nodriver hatası: {e}"


@mcp.tool()
async def web_scrape(url: str) -> str:
    """
    Verilen URL'nin içeriğini Markdown formatında döndürür.
    Önce Jina Reader dener (hızlı, ücretsiz, kurulum yok).
    Başarısız olursa nodriver ile bot korumasını aşarak okur.
    """
    print(f"[Scrape] URL: {url}")

    # Jina dene
    result = await _jina_read(url)
    if result:
        print("[Scrape] Jina Reader → başarılı")
        return f"[Kaynak: Jina Reader]\n\n{result}"

    # Fallback: nodriver
    print("[Scrape] Jina başarısız → nodriver fallback")
    result = await _nodriver_read(url)
    return f"[Kaynak: nodriver (fallback)]\n\n{result}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
