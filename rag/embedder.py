"""
rag/embedder.py
"""

import numpy as np
import torch
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


@st.cache_resource(show_spinner="Loading BGE text embedding model...")
def load_bge() -> SentenceTransformer:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model


@st.cache_resource(show_spinner="Loading CLIP image embedding model...")
def load_clip() -> tuple:
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def embed_texts_bge(texts: list, bge_model: SentenceTransformer) -> np.ndarray:
    vectors = bge_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    return np.atleast_2d(vectors.astype("float32"))   # always (N, 384)


def embed_query_bge(query: str, bge_model: SentenceTransformer) -> np.ndarray:
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vector = bge_model.encode([prefixed], normalize_embeddings=True)
    return vector[0].astype("float32")   # (384,)


def embed_images_clip(
    images: list,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor
) -> np.ndarray:
    vectors = []
    for img in images:
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            # get_image_features returns a plain tensor (batch_size, 512)
            # use clip_model.vision_model + visual_projection to be explicit
            pixel_values = inputs["pixel_values"]
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
            pooled = vision_outputs.pooler_output           # (1, hidden_size)
            projected = clip_model.visual_projection(pooled)  # (1, 512)

        vec = projected[0].detach().numpy()    # (512,)
        vec = vec / np.linalg.norm(vec)        # L2 normalize
        vectors.append(vec)                    # list of (512,) arrays

    return np.vstack(vectors).astype("float32")   # always (N, 512)


def embed_query_clip(
    query: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor
) -> np.ndarray:
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        # same explicit approach for text
        text_outputs = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        pooled = text_outputs.pooler_output              # (1, hidden_size)
        projected = clip_model.text_projection(pooled)   # (1, 512)

    vec = projected[0].detach().numpy()   # (512,)
    vec = vec / np.linalg.norm(vec)
    return vec.astype("float32")