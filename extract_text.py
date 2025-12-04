# extract_text.py
from typing import Tuple
import os

def load_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

def load_pdf(path: str) -> str:
    # using pypdf
    from pypdf import PdfReader
    reader = PdfReader(path)
    texts = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text()
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
    return "\n".join(texts)

def load_file(path: str) -> Tuple[str, str]:
    """
    Return (text, file_type)
    file_type = 'pdf' or 'docx' or 'txt'
    """
    path = os.path.abspath(path)
    if path.lower().endswith(".pdf"):
        return load_pdf(path), "pdf"
    elif path.lower().endswith(".docx"):
        return load_docx(path), "docx"
    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "txt"
    else:
        raise ValueError("Unsupported file type. Use .pdf, .docx or .txt")
