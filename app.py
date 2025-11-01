
# app.py
# -*- coding: utf-8 -*-
"""
Streamlit-приложение для извлечения спецификаций и таблиц из PDF,
включая СКАНЫ (через OCR Tesseract).

Основные возможности:
- Предпросмотр текста (если PDF-текстовый).
- Извлечение таблиц с помощью pdfplumber (для текстовых PDF).
- OCR-режим для сканов: PyMuPDF -> raster -> OpenCV -> Tesseract -> таблица.
- Специальные хелперы под «Спецификацию БС-1» (Поз., Обозначение, Наименование, Кол., Масса ед., Примечание).

Зависимости (см. requirements.txt):
streamlit, pymupdf, pdfplumber, pandas, opencv-python-headless, pytesseract
И понадобится установленный бинарник Tesseract в системе (пример: sudo apt-get install tesseract-ocr tesseract-ocr-rus).
"""
from __future__ import annotations
import io
import re
from typing import List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
import numpy as np

import fitz  # PyMuPDF
import pdfplumber

# OCR/Computer Vision
import cv2
import pytesseract

# ---------- Константы/регексы ----------
NAME_HEADER_REGEX = re.compile(r"(наимен|описан|item|description)", re.IGNORECASE)
SPEC_HEADER_CANDIDATES = ["Поз", "Обозначение", "Наименование", "Кол", "Масса", "Примеч"]

# ---------- Утилиты ----------
def page_count(pdf_bytes: bytes) -> int:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count

def parse_page_ranges(spec: str, total_pages: int) -> List[int]:
    pages: List[int] = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            a, b = max(1, a), min(total_pages, b)
            pages.extend(list(range(a, b + 1)))
        else:
            try:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.append(p)
            except ValueError:
                pass
    # uniq & keep order
    seen = set()
    result = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result

def extract_text_from_page(pdf_bytes: bytes, page_index0: int) -> str:
    # пытаемся через PyMuPDF, если текст есть
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index0)
        text = page.get_text("text")
    return text or ""

# ---------- ТАБЛИЦЫ: текстовый PDF через pdfplumber ----------
def extract_tables_textual(pdf_bytes: bytes, page_numbers_1based: List[int]) -> Tuple[pd.DataFrame, Dict[int, list]]:
    """
    Возвращает общий DataFrame из всех найденных таблиц на указанных страницах,
    а также "raw" словарь: {page_number: [dataframes...]}
    """
    all_tables: List[pd.DataFrame] = []
    raw: Dict[int, list] = {}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p1 in page_numbers_1based:
            idx = p1 - 1
            if not (0 <= idx < len(pdf.pages)):
                continue
            page = pdf.pages[idx]
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            raw[p1] = []
            for tbl in tables:
                df = pd.DataFrame(tbl)
                # переносим первую строку как заголовки, если похоже на заголовок
                if not df.empty and any(df.iloc[0].astype(str).str.contains(NAME_HEADER_REGEX, na=False)):
                    df.columns = [str(x).strip() for x in df.iloc[0]]
                    df = df.iloc[1:].reset_index(drop=True)
                raw[p1].append(df)
                if not df.empty:
                    all_tables.append(df)

    if not all_tables:
        return pd.DataFrame(), raw

    # Склеиваем, приведя ширину столбцов по максимуму
    wide = max(len(df.columns) for df in all_tables)
    norm = []
    for df in all_tables:
        df2 = df.copy()
        if len(df2.columns) < wide:
            df2 = df2.reindex(columns=list(range(wide)))
        norm.append(df2)
    merged = pd.concat(norm, ignore_index=True, sort=False)
    return merged, raw

# ---------- ТАБЛИЦЫ: OCR для СКАНОВ ----------
def render_page_to_image(pdf_bytes: bytes, page_index0: int, zoom: float = 3.0) -> np.ndarray:
    """
    Рендерим страницу PDF в изображение (RGB numpy).
    zoom 2.0-4.0 обычно достаточно.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc.load_page(page_index0)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.h, pix.w, 3)
        return img

def find_table_cells(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Поиск ячеек таблицы: морфология по линиям + контуры.
    Возвращает список bbox (x,y,w,h) ячеек.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 11)
    # ищем горизонтальные и вертикальные линии
    horizontal = thr.copy()
    vertical = thr.copy()
    horizontalsize = max(10, img.shape[1] // 80)
    verticalsize = max(10, img.shape[0] // 80)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    mask = cv2.add(horizontal, vertical)

    # контуры предполагаемых ячеек
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 5000:  # отсев мелочи/мусора
            continue
        boxes.append((x,y,w,h))

    # сортировка сверху-вниз, затем слева-направо
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def group_boxes_into_rows(boxes: List[Tuple[int,int,int,int]], y_tol: int = 12) -> List[List[Tuple[int,int,int,int]]]:
    """
    Группируем ячейки в строки по близости по Y.
    """
    rows: List[List[Tuple[int,int,int,int]]] = []
    for b in boxes:
        x,y,w,h = b
        placed = False
        for row in rows:
            # сравниваем с первым элементом строки
            ry = row[0][1]
            if abs(y - ry) <= y_tol:
                row.append(b); placed = True; break
        if not placed:
            rows.append([b])
    # внутри строки сортируем по X
    for r in rows:
        r.sort(key=lambda b: b[0])
    # фильтруем строки с малым числом ячеек
    rows = [r for r in rows if len(r) >= 3]
    return rows

def ocr_cell(img: np.ndarray, box: Tuple[int,int,int,int]) -> str:
    x,y,w,h = box
    crop = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    # для улучшения распознавания
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(bw, lang="rus+eng", config="--psm 6")
    return re.sub(r"\s+", " ", txt).strip()

def ocr_table_from_page(pdf_bytes: bytes, page_index0: int) -> pd.DataFrame:
    img = render_page_to_image(pdf_bytes, page_index0, zoom=3.0)
    boxes = find_table_cells(img)
    rows = group_boxes_into_rows(boxes)

    if not rows:
        return pd.DataFrame()

    # OCR всех ячеек
    text_rows = []
    for r in rows:
        text_row = [ocr_cell(img, b) for b in r]
        text_rows.append(text_row)

    # эвристика: выбираем строку-заголовок, похожую на набор столбцов спецификации
    header_idx = 0
    best_score = -1
    for i, r in enumerate(text_rows[:10]):
        joined = " ".join(r)
        score = sum(any(re.search(re.escape(word), c, re.IGNORECASE) for c in r) for word in SPEC_HEADER_CANDIDATES)
        if score > best_score:
            best_score = score; header_idx = i

    header = text_rows[header_idx]
    # нормализуем заголовки
    def norm_col(c: str) -> str:
        c = c.lower()
        if "обозн" in c: return "Обозначение"
        if "наимен" in c or "описан" in c: return "Наименование"
        if "масса" in c: return "Масса ед., кг"
        if "кол" in c: return "Кол."
        if "примеч" in c: return "Примечание"
        if "поз" in c: return "Поз."
        return c.strip().title()
    columns = [norm_col(c) for c in header]

    # данные: всё после header_idx
    body = text_rows[header_idx+1:]
    # приводим ширину строк к ширине заголовка
    W = len(columns)
    fixed = []
    for r in body:
        if len(r) < W:
            r = r + [""]*(W-len(r))
        elif len(r) > W:
            r = r[:W]
        fixed.append(r)

    df = pd.DataFrame(fixed, columns=columns)
    # чистим шумовые строки
    # удаляем пустые
    df = df[~(df.astype(str).apply(lambda s: s.str.strip()=="").all(axis=1))].reset_index(drop=True)
    return df

# ---------- Специализированный фильтр под «Спецификацию БС-1» ----------
def filter_spec_bs1(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # оставляем столбцы если есть
    keep = [c for c in df.columns if any(k.lower() in c.lower() for k in ["поз", "обозн", "наимен", "кол", "масса", "примеч"])]
    if keep:
        df = df[keep]
    # уберём очевидные заголовочные повторения в теле
    mask_header_like = df.apply(lambda r: any(word in " ".join(map(str, r)).lower() for word in ["поз", "наимен", "обозн"]), axis=1)
    df = df[~mask_header_like].reset_index(drop=True)
    return df

# ---------- UI ----------
st.set_page_config(page_title="PDF → Таблицы/Спецификация", layout="wide")
st.title("Импорт спецификаций из PDF (включая сканы)")

st.sidebar.subheader("Настройки")
ocr_mode = st.sidebar.toggle("OCR-режим (для сканов)", value=True)
st.sidebar.caption("Если PDF текстовый — можно выключить OCR и использовать pdfplumber.")

uploaded = st.file_uploader("Загрузите PDF", type=["pdf"])

if not uploaded:
    st.info("Загрузите PDF файл со спецификацией (например, лист с «Спецификация БС-1»).")
    st.stop()

pdf_bytes = uploaded.read()
total_pages = page_count(pdf_bytes)

st.sidebar.markdown(f"**Страниц**: {total_pages}")
page_ranges = st.sidebar.text_input("Страницы (пример: 1,3,5-7)", value="1")
pages = parse_page_ranges(page_ranges or "1", total_pages)

tab_text, tab_tables = st.tabs(["Текст/превью", "Таблицы/спецификация"])

with tab_text:
    preview_text = []
    for p in pages[:3]:  # немного для превью
        preview_text.append(f"--- Стр. {p} ---\n" + (extract_text_from_page(pdf_bytes, p-1) or "[текст не найден, вероятно скан]"))
    st.text_area("Предпросмотр текста (выдержки)", "\n\n".join(preview_text)[:4000], height=260)

with tab_tables:
    if not ocr_mode:
        st.write("Извлечение таблиц через pdfplumber (текстовый PDF).")
        df, raw = extract_tables_textual(pdf_bytes, pages)
    else:
        st.write("OCR-извлечение таблицы со страниц (сканы).")
        frames = []
        for p in pages:
            with st.spinner(f"OCR страница {p}…"):
                df = ocr_table_from_page(pdf_bytes, p-1)
                frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    df_spec = filter_spec_bs1(df)

    if df_spec.empty:
        st.warning("Не удалось распознать таблицу спецификации. Попробуйте: включить OCR, повысить качество PDF, или сузить диапазон до листа со спецификацией.")
    else:
        st.success(f"Найдено строк: {len(df_spec)}")
        st.dataframe(df_spec, use_container_width=True)

        csv = df_spec.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Скачать CSV спецификации", data=csv, file_name="spec_extracted.csv", mime="text/csv")

st.caption("Подходит для листов типа «Спецификация БС-1». Для лучших результатов установите Tesseract с русским языковым пакетом.")
