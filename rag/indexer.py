"""
rag/indexer.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  1. Builds two FAISS indexes — one for text (384-dim), one for images (512-dim)
  2. At query time, searches BOTH indexes independently
  3. Merges the results using score normalization so they can be ranked together

WHY TWO SEPARATE FAISS INDEXES:
  BGE produces 384-dimensional vectors.
  CLIP produces 512-dimensional vectors.
  A FAISS index is initialized for a fixed dimension — you cannot mix
  different-sized vectors in the same index. Two indexes is the only option.

HOW SCORE MERGING WORKS:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Problem: BGE scores and CLIP scores are not on the same scale.      │
  │                                                                      │
  │  BGE might return scores like:  [0.82, 0.79, 0.71]  (text chunks)  │
  │  CLIP might return scores like: [0.31, 0.28, 0.19]  (image pages)  │
  │                                                                      │
  │  If we merged these raw scores, all text chunks would always rank    │
  │  higher than all image chunks — not because they're more relevant,   │
  │  but just because BGE happens to produce higher numbers.             │
  │                                                                      │
  │  Solution: normalize each set of scores independently to [0, 1]:    │
  │    normalized = (score - min) / (max - min)                         │
  │                                                                      │
  │  Now both sets are comparable and the best result from either index  │
  │  can rank above a mediocre result from the other.                   │
  └──────────────────────────────────────────────────────────────────────┘

FAISS IndexFlatIP:
  "Flat"  = brute force exact search (no approximation)
  "IP"    = inner product (= cosine similarity when vectors are normalized)
  Good up to ~100k vectors before you'd need approximate methods (HNSW, IVF)
  For a demo PDF (< 1000 chunks) this is perfect.
─────────────────────────────────────────────────────────────────────────────
"""

import faiss
import numpy as np

# Fixed dimensions — must match the embedding models in embedder.py
BGE_DIM  = 384   # BAAI/bge-small-en-v1.5 output dimension
CLIP_DIM = 512   # openai/clip-vit-base-patch32 output dimension


# ── INDEX BUILDING ────────────────────────────────────────────────────────

def build_text_index(
    text_chunks: list[dict],
    bge_model
) -> tuple[faiss.Index, list[dict]]:
    """
    Embed all text chunks with BGE and store in a FAISS index.

    Args:
        text_chunks : list of chunk dicts from parser.py
        bge_model   : loaded BGE SentenceTransformer

    Returns:
        faiss_text_index : FAISS index with all text vectors
        text_chunks      : same list (returned for positional lookup)

    How the index works:
        FAISS assigns each vector a sequential integer ID starting at 0.
        When we search later and get back index 42, that maps to
        text_chunks[42] — so we can retrieve the original chunk text,
        page number, and filename for the citation.
    """
    from rag.embedder import embed_texts_bge

    texts = [chunk["content"] for chunk in text_chunks]

    # Embed all texts in one batch call
    vectors = embed_texts_bge(texts, bge_model)   # shape: (N, 384)

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Build FAISS index
    index = faiss.IndexFlatIP(BGE_DIM)
    index.add(vectors)

    return index, text_chunks


def build_image_index(
    image_chunks: list[dict],
    clip_model,
    clip_processor
) -> tuple[faiss.Index, list[dict]]:
    """
    Embed all page images with CLIP and store in a FAISS index.

    Args:
        image_chunks   : list of image chunk dicts from parser.py
        clip_model     : loaded CLIPModel
        clip_processor : loaded CLIPProcessor

    Returns:
        faiss_image_index : FAISS index with all image vectors
        image_chunks      : same list (for positional lookup)
    """
    from rag.embedder import embed_images_clip

    images = [chunk["image"] for chunk in image_chunks]

    # Embed all images — this is the slowest step (~1-2s per page on CPU)
    vectors = embed_images_clip(images, clip_model, clip_processor)   # shape: (N, 512)

    # ensure always 2D even if single image
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Build FAISS index
    index = faiss.IndexFlatIP(CLIP_DIM)
    index.add(vectors)

    return index, image_chunks


# ── SEARCH + MERGE ────────────────────────────────────────────────────────

def search_and_merge(
    query: str,
    text_index: faiss.Index,
    image_index: faiss.Index,
    text_chunks: list[dict],
    image_chunks: list[dict],
    bge_model,
    clip_model,
    clip_processor,
    top_k_text: int = 3,
    top_k_image: int = 2
) -> list[dict]:
    """
    Search both FAISS indexes independently, normalize scores, merge and rank.

    Args:
        query          : the user's question string
        text_index     : FAISS index of BGE text vectors
        image_index    : FAISS index of CLIP image vectors
        text_chunks    : original text chunk dicts (for metadata lookup)
        image_chunks   : original image chunk dicts (for metadata lookup)
        bge_model      : loaded BGE model (to embed the query)
        clip_model     : loaded CLIP model (to embed the query)
        clip_processor : loaded CLIP processor
        top_k_text     : how many text chunks to retrieve (default 3)
        top_k_image    : how many image pages to retrieve (default 2)

    Returns:
        list of up to (top_k_text + top_k_image) chunk dicts,
        sorted by normalized score descending,
        each with an added "score" and "normalized_score" field.

    ─────────────────────────────────────────────────────────────────
    STEP-BY-STEP WALKTHROUGH:

    1. Embed the query with BGE  → 384-dim query vector
       Embed the query with CLIP → 512-dim query vector

    2. Search text_index  with BGE  vector  → top 3 (score, index) pairs
       Search image_index with CLIP vector  → top 2 (score, index) pairs

    3. Look up each index to get the original chunk dict
       Attach the raw similarity score to each chunk

    4. Normalize text scores  to [0,1] within their group
       Normalize image scores to [0,1] within their group

    5. Combine all 5 results into one list
       Sort by normalized_score descending

    6. Return the sorted list — now text and image results compete fairly
    ─────────────────────────────────────────────────────────────────
    """
    from rag.embedder import embed_query_bge, embed_query_clip

    results = []

    # ── STEP 1: Embed query with both models ──────────────────────────
    query_vec_bge  = embed_query_bge(query, bge_model)    # shape: (384,)
    query_vec_clip = embed_query_clip(query, clip_model, clip_processor)  # shape: (512,)

    # FAISS expects shape (1, dim) for single-query search
    query_bge_2d  = query_vec_bge.reshape(1, -1)
    query_clip_2d = query_vec_clip.reshape(1, -1)

    # ── STEP 2: Search both indexes ───────────────────────────────────
    # search() returns:
    #   scores  : shape (1, k) — similarity scores, higher = more relevant
    #   indices : shape (1, k) — positions in the original chunk list

    text_scores,  text_indices  = text_index.search(query_bge_2d,  top_k_text)
    image_scores, image_indices = image_index.search(query_clip_2d, top_k_image)

    # Flatten from shape (1, k) to (k,)
    text_scores   = text_scores[0]
    text_indices  = text_indices[0]
    image_scores  = image_scores[0]
    image_indices = image_indices[0]

    # ── STEP 3: Collect raw results ───────────────────────────────────
    text_results = []
    for score, idx in zip(text_scores, text_indices):
        if idx == -1:   # FAISS returns -1 if fewer results than k
            continue
        chunk = text_chunks[idx].copy()
        chunk["raw_score"] = float(score)
        text_results.append(chunk)

    image_results = []
    for score, idx in zip(image_scores, image_indices):
        if idx == -1:
            continue
        chunk = image_chunks[idx].copy()
        chunk["raw_score"] = float(score)
        image_results.append(chunk)

    # ── STEP 4: Normalize scores to [0, 1] within each group ─────────
    #
    # Formula: normalized = (score - min) / (max - min)
    #
    # Example:
    #   BGE raw scores:  [0.82, 0.79, 0.71]
    #   min=0.71, max=0.82
    #   normalized: [(0.82-0.71)/(0.82-0.71), (0.79-0.71)/(0.82-0.71), (0.71-0.71)/(0.82-0.71)]
    #             = [1.0, 0.73, 0.0]
    #
    #   CLIP raw scores: [0.31, 0.19]
    #   min=0.19, max=0.31
    #   normalized: [1.0, 0.0]
    #
    #   Now both groups have their best result at 1.0 and worst at 0.0.
    #   They compete fairly when merged.

    def normalize_scores(chunk_list: list[dict]) -> list[dict]:
        if not chunk_list:
            return chunk_list
        scores = [c["raw_score"] for c in chunk_list]
        min_s, max_s = min(scores), max(scores)
        score_range = max_s - min_s
        for c in chunk_list:
            if score_range == 0:
                c["normalized_score"] = 1.0   # all scores identical
            else:
                c["normalized_score"] = (c["raw_score"] - min_s) / score_range
        return chunk_list

    text_results  = normalize_scores(text_results)
    image_results = normalize_scores(image_results)

    # ── STEP 5 & 6: Merge and sort ────────────────────────────────────
    all_results = text_results + image_results
    all_results.sort(key=lambda x: x["normalized_score"], reverse=True)

    return all_results