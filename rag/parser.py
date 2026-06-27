"""
rag/parser.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Takes a raw PDF (as bytes) and splits it into two lists:
    1. text_chunks  — list of dicts, each holding a passage of text
    2. image_chunks — list of dicts, each holding a PIL Image of a full page

WHY TWO SEPARATE LISTS:
  Text and images need different embedding models (BGE vs CLIP).
  Keeping them separate makes it easy to embed each with the right model
  and store them in their own FAISS index.

CHUNK METADATA:
  Every chunk — text or image — carries the same metadata keys:
    type     : "text" or "image"  ← this is how the LLM later knows which is which
    source   : original filename
    page     : page number (1-indexed)
    content  : the actual text (for text chunks) or a label (for image chunks)
    image    : None (for text) or a PIL Image object (for image chunks)

  The "type" field flows all the way through the pipeline into the prompt,
  so Groq knows to describe an image source differently from a text source.
─────────────────────────────────────────────────────────────────────────────
"""

import fitz          # PyMuPDF — reads PDFs
from PIL import Image
import io


def parse_pdf(file_bytes: bytes, filename: str) -> tuple[list[dict], list[dict]]:
    """
    Parse a PDF into text chunks and page images.

    Args:
        file_bytes : raw bytes of the uploaded PDF
        filename   : original filename, used in metadata for citations

    Returns:
        text_chunks  : list of text chunk dicts
        image_chunks : list of page image dicts
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    text_chunks  = []
    image_chunks = []

    for page_num, page in enumerate(doc, start=1):

        # ── TEXT EXTRACTION ──────────────────────────────────────────────
        # get_text("text") returns plain text from the page, preserving
        # line breaks but stripping formatting like bold/italic.
        raw_text = page.get_text("text").strip()

        if raw_text:
            # Split page text into overlapping chunks.
            # Overlap ensures a sentence split across two chunks isn't lost —
            # both chunks contain it, so retrieval can find it either way.
            chunks = _split_text(raw_text, chunk_size=300, overlap=50)

            for chunk_idx, chunk_text in enumerate(chunks):
                text_chunks.append({
                    "type"    : "text",          # ← used later to tell LLM this is text
                    "source"  : filename,
                    "page"    : page_num,
                    "chunk_id": chunk_idx,       # which chunk within the page
                    "content" : chunk_text,      # the actual passage
                    "image"   : None             # no image for text chunks
                })

        # ── IMAGE EXTRACTION (full page render) ──────────────────────────
        # Instead of extracting embedded images (logos, photos), we render
        # the ENTIRE page as a PNG at 150 DPI.
        #
        # Why render the full page?
        #   - Charts and tables aren't "embedded images" — they're drawn
        #     as vector graphics by PyMuPDF. Extracting embedded images
        #     would miss them entirely.
        #   - A full-page render captures everything: charts, tables,
        #     diagrams, figures, and any embedded photos.
        #
        # 150 DPI is a good balance — clear enough for CLIP to understand
        # the visual content, small enough to not slow down processing.

        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        image_chunks.append({
            "type"   : "image",          # ← used later to tell LLM this is visual
            "source" : filename,
            "page"   : page_num,
            "content": f"[Visual content — page {page_num} of {filename}]",
            # ↑ This text label is what the LLM sees in the prompt.
            #   The actual PIL image is used only for CLIP embedding
            #   and for displaying the thumbnail in the Streamlit UI.
            "image"  : pil_image         # PIL Image — used by CLIP embedder
        })

    doc.close()
    return text_chunks, image_chunks


def _split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size = 300 words  → roughly 400 tokens (safe under BGE's 512-token limit)
    overlap    = 50  words  → the last 50 words of chunk N are the
                               first 50 words of chunk N+1

    Example with chunk_size=5, overlap=2:
      words = [A B C D E F G H]
      chunk 1 = [A B C D E]
      chunk 2 = [D E F G H]   ← D and E repeated from chunk 1
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap   # step forward by (chunk_size - overlap)

    return chunks