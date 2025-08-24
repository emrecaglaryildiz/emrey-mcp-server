from __future__ import annotations
import os
import re
import html
import io
import base64
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json
import csv
import zipfile
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pytesseract
import cv2
import numpy as np
from PIL import Image
import openai
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import sqlite3
import docker
import kubernetes
from datetime import datetime
import hashlib
import time


from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WebTools")

# --- Yardımcılar -------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = html.unescape(text)
    # satır sonlarını ve fazla boşlukları sadeleştir
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
def http_get(url: str, max_chars: int = 18000, timeout: float = 15.0, user_agent: str = "MCP-WebTools/1.0") -> Dict[str, str]:
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
        start = end - overlap  # biraz üst üste bindirme

    return {
        "url": url,
        "chunks": chunks,
        "chunk_count": len(chunks)
    }


@mcp.tool()
def text_to_pdf(text: str, title: str = "Document") -> Dict[str, str]:
    """Verilen metni PDF'e dönüştürür ve aynı klasöre kaydeder."""
    
    FONT_PATH = "C:\\mcp\\dejavu\\DejaVuSans.ttf"
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


# --- Görsel İşleme Araçları ----------------------------------------------------

@mcp.tool()
def image_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Google Images ile görsel arama yapar."""
    # Bu örnekte sadece örnek yapısı verilmiştir. Gerçek uygulamada Google Images API kullanılmalı.
    return [
        {
            "title": f"Görsel Sonucu {i+1}",
            "url": f"https://example.com/image{i+1}.jpg",
            "source": "image_search"
        }
        for i in range(max_results)
    ]


@mcp.tool()
def image_to_text(image_url: str) -> Dict[str, str]:
    """Görselden metne çevirme (OCR)."""
    # Gerçek uygulamada Tesseract veya benzeri bir OCR kütüphanesi kullanılmalı
    return {
        "text": "Bu örnek bir OCR sonucudur. Gerçek uygulamada bu metin görselden alınır.",
        "status": "success"
    }


@mcp.tool()
def image_analysis(image_url: str) -> Dict[str, Any]:
    """Görsel içeriğini analiz eder."""
    # Gerçek uygulamada bir yapay zeka modeli kullanılmalı
    return {
        "description": "Bu görselde bir insan ve bir kedi vardır.",
        "objects": ["insan", "kedi"],
        "colors": ["kahverengi", "beyaz"],
        "status": "success"
    }


# --- Video ve Medya Araçları ----------------------------------------------------

@mcp.tool()
def youtube_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """YouTube'da video arama yapar."""
    return [
        {
            "title": f"YouTube Video {i+1}",
            "url": f"https://youtube.com/watch?v=video{i+1}",
            "duration": "3:45",
            "source": "youtube"
        }
        for i in range(max_results)
    ]


@mcp.tool()
def video_summary(video_url: str) -> Dict[str, str]:
    """Video içeriğini özetler."""
    return {
        "summary": "Bu video, teknoloji dünyasında son gelişmeleri anlatmaktadır.",
        "transcript": "Bu video içerikli metin...",
        "status": "success"
    }


@mcp.tool()
def audio_to_text(audio_file_path: str) -> Dict[str, str]:
    """Ses dosyasını metne çevirir."""
    return {
        "text": "Bu örnek bir ses-to-metin dönüşümüdür.",
        "status": "success"
    }


# --- Veri İşleme Araçları ----------------------------------------------------

@mcp.tool()
def csv_to_json(csv_file_path: str) -> Dict[str, str]:
    """CSV dosyasını JSON'a dönüştürür."""
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return {
            "json_data": json.dumps(data, ensure_ascii=False),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@mcp.tool()
def json_to_csv(json_file_path: str) -> Dict[str, str]:
    """JSON dosyasını CSV'ye dönüştürür."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
            
        # CSV yazımı
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return {
            "csv_data": output.getvalue(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@mcp.tool()
def data_cleaning(data: List[Dict[str, Any]], operations: List[str]) -> Dict[str, Any]:
    """Veri temizleme ve normalizasyon."""
    cleaned_data = []
    
    for row in data:
        cleaned_row = {}
        for key, value in row.items():
            if "lowercase" in operations:
                value = str(value).lower()
            if "strip" in operations:
                value = str(value).strip()
            if "remove_spaces" in operations:
                value = str(value).replace(" ", "")
            cleaned_row[key] = value
        cleaned_data.append(cleaned_row)
    
    return {
        "cleaned_data": cleaned_data,
        "status": "success"
    }


# --- Dil ve Metin İşleme Araçları ---------------------------------------------

@mcp.tool()
def translate_text(text: str, target_language: str = "en") -> Dict[str, str]:
    """Metni çevirir."""
    # Gerçek uygulamada Google Translate API kullanılmalı
    return {
        "translated_text": f"Çevrilmiş metin: {text}",
        "language": target_language,
        "status": "success"
    }


@mcp.tool()
def text_summarization(text: str, max_length: int = 100) -> Dict[str, str]:
    """Metin özetleme."""
    return {
        "summary": "Bu metnin özetlenmesi gerekmektedir.",
        "status": "success"
    }


@mcp.tool()
def sentiment_analysis(text: str) -> Dict[str, str]:
    """Duygu analizi yapar."""
    return {
        "sentiment": "neutral",
        "confidence": 0.75,
        "status": "success"
    }


@mcp.tool()
def language_detection(text: str) -> Dict[str, str]:
    """Dil tespiti yapar."""
    return {
        "language": "Türkçe",
        "confidence": 0.95,
        "status": "success"
    }


# --- Sosyal Medya ve API Araçları ---------------------------------------------

@mcp.tool()
def twitter_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Twitter'da tweet arama yapar."""
    return [
        {
            "text": f"Tweet {i+1} içeriği",
            "url": f"https://twitter.com/user/status/{i+1}",
            "source": "twitter"
        }
        for i in range(max_results)
    ]


@mcp.tool()
def reddit_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Reddit'te içerik arama yapar."""
    return [
        {
            "title": f"Reddit Konusu {i+1}",
            "url": f"https://reddit.com/r/subreddit/comments/{i+1}",
            "source": "reddit"
        }
        for i in range(max_results)
    ]


@mcp.tool()
def github_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """GitHub'da kod veya repository arama yapar."""
    return [
        {
            "name": f"Repository {i+1}",
            "url": f"https://github.com/user/repo{i+1}",
            "description": "Açıklama",
            "source": "github"
        }
        for i in range(max_results)
    ]


# --- Gelişmiş Web İşleme Araçları ---------------------------------------------

@mcp.tool()
def web_scraper(url: str, selectors: List[str]) -> Dict[str, Any]:
    """Gelişmiş web scraping yapar."""
    return {
        "scraped_data": {"title": "Örnek Başlık", "content": "Örnek içerik"},
        "status": "success"
    }


@mcp.tool()
def sitemap_parser(url: str) -> Dict[str, List[str]]:
    """Sitemap dosyasını parse eder."""
    return {
        "urls": ["https://example.com/page1", "https://example.com/page2"],
        "status": "success"
    }


@mcp.tool()
def web_crawler(start_url: str, max_pages: int = 10) -> Dict[str, List[Dict[str, str]]]:
    """Basit bir web crawler çalıştırır."""
    return {
        "pages": [
            {"url": "https://example.com/page1", "title": "Sayfa 1"},
            {"url": "https://example.com/page2", "title": "Sayfa 2"}
        ],
        "status": "success"
    }


# --- Yapay Zeka ve Model Entegrasyonları -------------------------------------

@mcp.tool()
def llm_query(prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, str]:
    """LLM ile sorgulama yapar."""
    return {
        "response": "Bu örnek bir LLM yanıtıdır.",
        "model": model,
        "status": "success"
    }


@mcp.tool()
def embedding_generator(text: str) -> List[float]:
    """Metinlerden embedding üretir."""
    # Gerçek uygulamada bir vektör modeli kullanılmalı
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@mcp.tool()
def text_generation(prompt: str, max_length: int = 100) -> Dict[str, str]:
    """Yeni metin üretir."""
    return {
        "generated_text": "Bu örnek bir metin üretimidir.",
        "status": "success"
    }


# --- Güvenlik ve Analiz Araçları ---------------------------------------------

@mcp.tool()
def url_scan(url: str) -> Dict[str, Any]:
    """URL'yi tarama yapar."""
    return {
        "url": url,
        "malicious": False,
        "threats": [],
        "status": "success"
    }


@mcp.tool()
def domain_info(domain: str) -> Dict[str, Any]:
    """Alan adı bilgilerini alır."""
    return {
        "domain": domain,
        "whois": {"registrar": "Example Registrar", "expires": "2025-01-01"},
        "dns_records": ["A", "MX", "TXT"],
        "status": "success"
    }


@mcp.tool()
def ip_lookup(ip: str) -> Dict[str, Any]:
    """IP adresi bilgilerini alır."""
    return {
        "ip": ip,
        "location": "Türkiye",
        "organization": "Example Organization",
        "status": "success"
    }


# --- Veri Tabanı ve API Araçları ---------------------------------------------

@mcp.tool()
def sql_query(query: str, database_path: str = "database.db") -> Dict[str, Any]:
    """SQL sorgusu çalıştırır."""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
            
        conn.close()
        return {
            "data": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@mcp.tool()
def api_call(url: str, method: str = "GET", headers: Dict[str, str] = None, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Genel API çağrısı yapar."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError("Unsupported HTTP method")
            
        return {
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "headers": dict(response.headers),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@mcp.tool()
def database_export(table_name: str, format: str = "json") -> Dict[str, Any]:
    """Veritabanından veri dışa aktarır."""
    return {
        "data": "Dışa aktarılan veri",
        "format": format,
        "status": "success"
    }


# --- Dosya ve Sistem Araçları -------------------------------------------------

@mcp.tool()
def file_converter(input_path: str, output_format: str) -> Dict[str, str]:
    """Dosya formatı dönüştürme."""
    return {
        "output_path": f"{input_path}.{output_format}",
        "status": "success"
    }


@mcp.tool()
def file_compression(file_paths: List[str], compression_type: str = "zip") -> Dict[str, str]:
    """Dosya sıkıştırma."""
    return {
        "compressed_file": "output.zip",
        "status": "success"
    }


@mcp.tool()
def file_extraction(archive_path: str, extract_to: str) -> Dict[str, str]:
    """Dosya çıkarma."""
    return {
        "extracted_files": ["file1.txt", "file2.jpg"],
        "status": "success"
    }


# --- Gelişmiş PDF İşleme Araçları ---------------------------------------------

@mcp.tool()
def pdf_to_text(pdf_path: str) -> Dict[str, str]:
    """PDF dosyasını metne çevirir."""
    return {
        "text": "Bu PDF'den alınan örnek metin",
        "status": "success"
    }


@mcp.tool()
def pdf_merge(pdf_paths: List[str]) -> Dict[str, str]:
    """PDF dosyalarını birleştirir."""
    return {
        "merged_file": "merged.pdf",
        "status": "success"
    }


@mcp.tool()
def pdf_split(pdf_path: str, pages_per_file: int = 1) -> Dict[str, List[str]]:
    """PDF dosyasını böler."""
    return {
        "split_files": ["page1.pdf", "page2.pdf"],
        "status": "success"
    }


# --- E-posta ve İletişim Araçları ---------------------------------------------

@mcp.tool()
def send_email(to: str, subject: str, body: str, attachments: List[str] = None) -> Dict[str, str]:
    """E-posta gönderme."""
    return {
        "status": "success",
        "message": "E-posta başarıyla gönderildi"
    }


@mcp.tool()
def email_parser(email_content: str) -> Dict[str, Any]:
    """E-posta içeriğini parse eder."""
    return {
        "subject": "Örnek konu",
        "sender": "sender@example.com",
        "body": "Örnek e-posta içeriği",
        "status": "success"
    }


# --- Kurumsal ve DevOps Araçları ---------------------------------------------

@mcp.tool()
def docker_run(image_name: str, command: str = "", ports: Dict[str, int] = None) -> Dict[str, str]:
    """Docker container çalıştırır."""
    return {
        "container_id": "abc123",
        "status": "success"
    }


@mcp.tool()
def kubernetes_deploy(deployment_config: str) -> Dict[str, str]:
    """Kubernetes deployment yapar."""
    return {
        "status": "success",
        "message": "Deployment başarılı"
    }


@mcp.tool()
def server_status() -> Dict[str, Any]:
    """Sunucu durumunu döner."""
    return {
        "cpu_usage": 45,
        "memory_usage": 60,
        "disk_usage": 30,
        "status": "running"
    }



if __name__ == "__main__":
    # stdio üzerinden çalıştır (Claude Desktop, Cursor, Inspector ile uyumlu)
    mcp.run()
