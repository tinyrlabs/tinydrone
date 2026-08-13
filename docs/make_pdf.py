#!/usr/bin/env python3
"""DRONE_REHBERI.md → PDF (akademik siyah-beyaz tema, Liberation Serif)"""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_DIR = "/usr/share/fonts/truetype/liberation"
pdfmetrics.registerFont(TTFont("LSerif", f"{FONT_DIR}/LiberationSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LSerif-B", f"{FONT_DIR}/LiberationSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LSerif-I", f"{FONT_DIR}/LiberationSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LMono", f"{FONT_DIR}/LiberationMono-Regular.ttf"))

OUT = "/home/ubuntu/projects/tinydrone/docs/DRONE_REHBERI.pdf"
SRC = "/home/ubuntu/projects/tinydrone/docs/DRONE_REHBERI.md"

BLACK = colors.HexColor("#111111")
GRAY = colors.HexColor("#444444")
LIGHT = colors.HexColor("#f2f2f2")

S = {}
S["body"] = ParagraphStyle("body", fontName="LSerif", fontSize=10, leading=14,
                           alignment=TA_JUSTIFY, textColor=BLACK)
S["h1"] = ParagraphStyle("h1", fontName="LSerif-B", fontSize=18, leading=22,
                         spaceBefore=6, spaceAfter=10, textColor=BLACK)
S["h2"] = ParagraphStyle("h2", fontName="LSerif-B", fontSize=14, leading=18,
                         spaceBefore=14, spaceAfter=6, textColor=BLACK)
S["h3"] = ParagraphStyle("h3", fontName="LSerif-B", fontSize=11.5, leading=15,
                         spaceBefore=10, spaceAfter=4, textColor=GRAY)
S["code"] = ParagraphStyle("code", fontName="LMono", fontSize=8.5, leading=11,
                           backColor=LIGHT, borderPadding=6, spaceBefore=4,
                           spaceAfter=6, textColor=BLACK)
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=16,
                             bulletIndent=4, spaceAfter=2)
S["tbl"] = ParagraphStyle("tbl", fontName="LSerif", fontSize=8.5, leading=11,
                          textColor=BLACK)

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def parse_md(md):
    lines = md.split("\n")
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # Title
        if ln.startswith("# ") and not ln.startswith("## "):
            out.append(("h1", ln[2:].strip())); i += 1; continue
        if ln.startswith("## "):
            out.append(("h2", ln[3:].strip())); i += 1; continue
        if ln.startswith("### "):
            out.append(("h3", ln[4:].strip())); i += 1; continue
        # Code block
        if ln.strip().startswith("```"):
            i += 1; buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            out.append(("code", "\n".join(buf)))
            i += 1; continue
        # Table
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|", lines[i+1]):
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i]); i += 1
            rows = []
            for r in buf:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                rows.append(cells)
            out.append(("table", rows[0], rows[2:]))  # header, skip separator
            continue
        # Bullet
        if re.match(r"^\s*[-*] ", ln):
            out.append(("bullet", re.sub(r"^\s*[-*] ", "", ln).strip()))
            i += 1; continue
        if re.match(r"^\s*\d+\. ", ln):
            out.append(("bullet", re.sub(r"^\s*\d+\. ", "", ln).strip()))
            i += 1; continue
        # Bold-only line (section intro)
        if ln.strip() and not ln.startswith("---") and ln.strip() != "":
            out.append(("p", ln.strip()))
        i += 1
    return out

def build():
    md = open(SRC).read()
    items = parse_md(md)

    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            topMargin=2.0*cm, bottomMargin=2.0*cm,
                            title="TINYDRONE — Gömülü Otonom Hedef Tespit Drone Rehberi",
                            author="Tiny R Labs")

    story = []
    for item in items:
        kind = item[0]
        if kind == "h1":
            story.append(Paragraph(esc(item[1]), S["h1"]))
            story.append(Spacer(1, 4))
        elif kind == "h2":
            story.append(Paragraph(esc(item[1]), S["h2"]))
        elif kind == "h3":
            story.append(Paragraph(esc(item[1]), S["h3"]))
        elif kind == "p":
            # inline code + bold handling
            t = esc(item[1])
            t = re.sub(r"`([^`]+)`", r'<font face="LMono" size="8.5">\1</font>', t)
            t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
            story.append(Paragraph(t, S["body"]))
            story.append(Spacer(1, 3))
        elif kind == "bullet":
            t = esc(item[1])
            t = re.sub(r"`([^`]+)`", r'<font face="LMono" size="8.5">\1</font>', t)
            t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
            story.append(Paragraph(t, S["bullet"], bulletText="•"))
        elif kind == "code":
            story.append(Paragraph(esc(item[1]).replace("\n", "<br/>"), S["code"]))
        elif kind == "table":
            header, rows = item[1], item[2]
            data = [[Paragraph(esc(c), S["tbl"]) for c in header]]
            for r in rows:
                data.append([Paragraph(esc(c), S["tbl"]) for c in r])
            n = len(header)
            tbl = Table(data, colWidths=[(17.5 - 0.4) * cm / max(n, 1)] * n)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))

    doc.build(story)
    print(f"PDF oluşturuldu: {OUT}")

build()
