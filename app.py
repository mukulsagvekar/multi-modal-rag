"""
app.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  The main Streamlit UI. Orchestrates the full pipeline:

  1. LOAD MODELS   : BGE + CLIP loaded once, cached in memory
  2. UPLOAD        : User uploads PDF via sidebar
  3. INGEST        : Parse → Embed → Build dual FAISS indexes
  4. CHAT          : User asks questions → search → merge → generate → display

SESSION STATE KEYS:
  text_index   : FAISS index for BGE text vectors
  image_index  : FAISS index for CLIP image vectors
  text_chunks  : list of text chunk dicts (metadata + content)
  image_chunks : list of image chunk dicts (metadata + PIL images)
  messages     : chat history list [{role, content, sources}]
  indexed_file : filename of the currently indexed PDF

WHY SESSION STATE:
  Streamlit reruns the entire script on every interaction (button click,
  chat input, etc). Without session_state, the FAISS indexes would be
  rebuilt from scratch on every rerun — taking 30-60 seconds each time.
  session_state persists data across reruns within the same browser session.
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
from rag.parser import parse_pdf
from rag.embedder import load_bge, load_clip
from rag.indexer import build_text_index, build_image_index, search_and_merge
from rag.generator import generate_answer

# ── PAGE CONFIG ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multimodal RAG — CLIP + BGE + FAISS + Groq")
st.caption("Upload a PDF → ask questions → get cited answers from text and images")

# ── LOAD MODELS (cached — runs once per session) ──────────────────────────

# These lines trigger the @st.cache_resource functions in embedder.py.
# First load: downloads models + loads into memory (~30s on first ever run).
# Subsequent loads: instant (served from HuggingFace cache on disk).

bge_model               = load_bge()
clip_model, clip_proc   = load_clip()

# ── SESSION STATE INITIALISATION ──────────────────────────────────────────

for key, default in {
    "text_index"  : None,
    "image_index" : None,
    "text_chunks" : None,
    "image_chunks": None,
    "messages"    : [],
    "indexed_file": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR: UPLOAD + INGESTION ───────────────────────────────────────────

with st.sidebar:
    st.header("📁 Upload Document")
    st.caption("PDF files only. Images are extracted as full-page renders.")

    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        help="Upload any PDF. Text and images are indexed separately."
    )

    if uploaded_file is not None:
        # Only re-index if a NEW file is uploaded (not on every rerun)
        if st.session_state.indexed_file != uploaded_file.name:
            if st.button("🔍 Process & Index", type="primary", use_container_width=True):

                file_bytes = uploaded_file.read()

                # ── STEP 1: PARSE ──────────────────────────────────────
                # Split PDF into text chunks + page images
                with st.spinner("📖 Parsing PDF..."):
                    text_chunks, image_chunks = parse_pdf(file_bytes, uploaded_file.name)

                st.toast(f"Parsed {len(text_chunks)} text chunks + {len(image_chunks)} page images")

                # ── STEP 2: BUILD TEXT INDEX (BGE) ─────────────────────
                # Embed all text chunks with BGE → store in FAISS #1
                with st.spinner(f"🔤 Embedding {len(text_chunks)} text chunks with BGE..."):
                    text_index, text_chunks = build_text_index(text_chunks, bge_model)

                # ── STEP 3: BUILD IMAGE INDEX (CLIP) ───────────────────
                # Embed all page images with CLIP → store in FAISS #2
                with st.spinner(f"🖼️ Embedding {len(image_chunks)} page images with CLIP..."):
                    image_index, image_chunks = build_image_index(
                        image_chunks, clip_model, clip_proc
                    )

                # ── STEP 4: STORE IN SESSION STATE ─────────────────────
                st.session_state.text_index   = text_index
                st.session_state.image_index  = image_index
                st.session_state.text_chunks  = text_chunks
                st.session_state.image_chunks = image_chunks
                st.session_state.indexed_file = uploaded_file.name
                st.session_state.messages     = []   # reset chat for new file

                st.success(f"✅ Indexed: {uploaded_file.name}")
        else:
            st.success(f"✅ Active: {st.session_state.indexed_file}")

    # Show index stats if ready
    if st.session_state.text_index is not None:
        st.divider()
        st.markdown("**Index stats**")
        col1, col2 = st.columns(2)
        col1.metric("Text chunks", len(st.session_state.text_chunks))
        col2.metric("Image pages", len(st.session_state.image_chunks))

        st.divider()
        st.markdown("**Retrieval settings**")
        top_k_text  = st.slider("Text chunks to retrieve",  1, 6, 3)
        top_k_image = st.slider("Image pages to retrieve",  1, 4, 2)
    else:
        top_k_text  = 3
        top_k_image = 2

    st.divider()
    st.markdown("""
**Stack:**
- 🔤 BGE `bge-small-en-v1.5` — text retrieval
- 🖼️ CLIP `ViT-B/32` — image retrieval
- 🗄️ FAISS `IndexFlatIP` × 2
- 🤖 Groq `llama-3.3-70b-versatile`
    """)

# ── MAIN AREA: CHAT INTERFACE ─────────────────────────────────────────────

if st.session_state.text_index is None:
    # No document indexed yet — show instructions
    st.info("👈 Upload a PDF in the sidebar and click **Process & Index** to start.")

    with st.expander("How it works"):
        st.markdown("""
**Ingestion pipeline** (runs once per PDF):
1. **Parse** — PyMuPDF extracts text chunks (300 words) and renders each page as a PNG
2. **Embed text** — BGE `bge-small-en-v1.5` converts each text chunk to a 384-dim vector
3. **Embed images** — CLIP `ViT-B/32` converts each page image to a 512-dim vector
4. **Index** — Two separate FAISS indexes store the vectors (text=384-dim, image=512-dim)

**Query pipeline** (runs on every question):
1. **Embed query (text)** — BGE embeds your question → search FAISS text index → top 3 chunks
2. **Embed query (image)** — CLIP embeds your question → search FAISS image index → top 2 pages
3. **Merge** — Scores normalized to [0,1] independently, then merged and ranked
4. **Generate** — Retrieved chunks sent to Groq `llama-3.3-70b` with citation instructions
5. **Answer** — Groq returns an answer with [Source N] citations pointing to specific pages
        """)
else:
    # ── DISPLAY CHAT HISTORY ──────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources expander for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} retrieved)"):
                    for i, chunk in enumerate(msg["sources"], 1):

                        # Citation header
                        score_pct = f"{chunk['normalized_score']*100:.0f}%"
                        st.markdown(
                            f"**[Source {i}]** `{chunk['source']}` — "
                            f"Page {chunk['page']} | "
                            f"Type: `{chunk['type']}` | "
                            f"Relevance: {score_pct}"
                        )

                        if chunk["type"] == "text":
                            # Show text preview
                            preview = chunk["content"][:400]
                            if len(chunk["content"]) > 400:
                                preview += "..."
                            st.caption(preview)

                        else:
                            # Show actual page image thumbnail
                            # This is the key UI feature — users can see the
                            # chart/diagram that CLIP retrieved
                            if chunk.get("image") is not None:
                                st.image(
                                    chunk["image"],
                                    caption=f"Page {chunk['page']} — {chunk['source']}",
                                    width='stretch'
                                )

                        if i < len(msg["sources"]):
                            st.divider()

    # ── CHAT INPUT ────────────────────────────────────────────────────
    if question := st.chat_input(
        f"Ask a question about {st.session_state.indexed_file}..."
    ):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching indexes..."):
                # ── RETRIEVAL ─────────────────────────────────────────
                # Search both FAISS indexes, normalize scores, merge
                retrieved = search_and_merge(
                    query        = question,
                    text_index   = st.session_state.text_index,
                    image_index  = st.session_state.image_index,
                    text_chunks  = st.session_state.text_chunks,
                    image_chunks = st.session_state.image_chunks,
                    bge_model    = bge_model,
                    clip_model   = clip_model,
                    clip_processor = clip_proc,
                    top_k_text   = top_k_text,
                    top_k_image  = top_k_image
                )

            with st.spinner("🤖 Generating answer with Groq..."):
                # ── GENERATION ────────────────────────────────────────
                # Build prompt with source labels, send to Groq 70B
                answer = generate_answer(question, retrieved)

            # Display answer
            st.markdown(answer)

            # Display sources expander
            with st.expander(f"📚 Sources ({len(retrieved)} retrieved)"):
                for i, chunk in enumerate(retrieved, 1):
                    score_pct = f"{chunk['normalized_score']*100:.0f}%"
                    st.markdown(
                        f"**[Source {i}]** `{chunk['source']}` — "
                        f"Page {chunk['page']} | "
                        f"Type: `{chunk['type']}` | "
                        f"Relevance: {score_pct}"
                    )

                    if chunk["type"] == "text":
                        st.caption(chunk["content"][:400] + ("..." if len(chunk["content"]) > 400 else ""))
                    else:
                        if chunk.get("image") is not None:
                            st.image(
                                chunk["image"],
                                caption=f"Page {chunk['page']} — {chunk['source']}",
                                width='stretch'
                            )

                    if i < len(retrieved):
                        st.divider()

        # Save to chat history (PIL images are stored in chunk dicts)
        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer,
            "sources": retrieved
        })