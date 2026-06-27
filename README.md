# Multimodal RAG — CLIP + BGE + FAISS + Groq

A multimodal Retrieval-Augmented Generation (RAG) app that lets users upload a PDF and ask questions — getting answers with inline citations pointing to specific pages. Text and images are embedded separately using purpose-built models and retrieved from dual FAISS indexes, with cross-modal score normalization to rank results fairly across modalities.

**Live demo →** [multi-modal-rag.streamlit.app](https://multi-modal-rag-wqhuxf3aw29t9jpgpa6p5y.streamlit.app/)

---

## How it works

### Ingestion pipeline (runs once per PDF, zero API calls)

```
Upload PDF
    ↓
PyMuPDF — extract text chunks (300 words, 50-word overlap)
         + render each page as PNG at 150 DPI
    ↓
BGE  (bge-small-en-v1.5)      →  384-dim vectors  →  FAISS index #1 (text)
CLIP (ViT-B/32)                →  512-dim vectors  →  FAISS index #2 (images)
```

### Query pipeline (one Groq API call per question)

```
User question
    ↓
BGE  embeds query  →  search FAISS #1  →  top 3 text chunks
CLIP embeds query  →  search FAISS #2  →  top 2 page images
    ↓
Cross-modal score normalization → merge → rank
    ↓
Groq llama-3.3-70b generates answer with [Source N] citations
    ↓
Streamlit displays answer + sources expander (text preview / page image thumbnail)
```

### Why dual indexes?

BGE and CLIP produce scores on different scales. Without normalization, text chunks would always outrank image pages — not because they're more relevant, but because BGE produces higher raw numbers. Each group is normalized independently to `[0, 1]` before merging, so results compete fairly regardless of which model produced them.

---

## Stack

| Component | Tool | Purpose |
|---|---|---|
| Text embedding | `BAAI/bge-small-en-v1.5` | 384-dim, 512-token limit, purpose-built for retrieval |
| Image embedding | `openai/clip-vit-base-patch32` | 512-dim, aligns text + image in same vector space |
| Vector store | `faiss-cpu` — two `IndexFlatIP` indexes | Exact cosine similarity search, no GPU needed |
| PDF parsing | `PyMuPDF` | Text extraction + full-page image rendering |
| LLM | `Groq llama-3.3-70b-versatile` | Fast inference, free tier, strong citation following |
| UI | `Streamlit` | File upload, chat interface, sources expander |

---

## Project structure

```
multimodal-rag/
├── app.py                  # Streamlit UI — upload, chat, sources display
├── rag/
│   ├── parser.py           # PyMuPDF: text chunks + page image rendering
│   ├── embedder.py         # BGE (text) + CLIP (image) model loaders and embed functions
│   ├── indexer.py          # Build dual FAISS indexes, search, score normalization, merge
│   └── generator.py        # Prompt builder + Groq API call
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # API key (never committed)
```

---

## Key design decisions

**BGE for text, not CLIP** — CLIP has a 77-token hard limit on text inputs. A 300-word chunk is ~400 tokens — CLIP would silently truncate 80% of every chunk. BGE has a 512-token limit and is trained specifically for asymmetric retrieval (short query → long document), making it the right tool for text.

**Full-page rendering instead of embedded image extraction** — Charts and tables in PDFs are usually drawn as vector graphics, not stored as embedded images. Extracting embedded images would miss them entirely. Rendering the full page at 150 DPI captures everything: charts, tables, diagrams, and any embedded photos.

**In-memory FAISS, no persistent store** — Each user session gets its own isolated FAISS index in `st.session_state`. No cross-session persistence means no data leakage between users and no external database dependency.

**Zero ingestion API cost** — BGE and CLIP run locally. The only API call is to Groq at query time — one call per user question. A 25-page PDF indexes in under 45 seconds with no API usage.

---

## Limitations

- **Text-only LLM** — Groq's `llama-3.3-70b` cannot see images. When an image page is retrieved, the model is told "a visual exists on page N" and cites it. The user sees the actual page thumbnail in the Sources expander.
- **PDF only** — currently supports PDF uploads. DOCX and PPTX support can be added via `python-docx` / `python-pptx` parsers.
- **In-memory storage** — indexes are lost when the session ends. Users need to re-upload for a new session.
- **Scanned PDFs** — PyMuPDF cannot extract text from scanned/image-only PDFs without OCR. Text-based PDFs only.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/mukulsagvekar/multi-modal-rag.git
cd multimodal-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads BGE (~33MB) and CLIP (~150MB) from HuggingFace. Subsequent runs load from local cache instantly.

### 3. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) — free, no credit card required.

### 4. Add your API key

```toml
# .streamlit/secrets.toml
[groq]
api_key = "gsk_..."
```

> Add `.streamlit/secrets.toml` to your `.gitignore` — never commit this file.

### 5. Run

```bash
streamlit run app.py
```
---

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (secrets.toml excluded via .gitignore)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, file `app.py`
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   [groq]
   api_key = "gsk_..."
   ```
5. Deploy

Models download on first cold start (~60s). Subsequent loads are fast.

---

## License

MIT
