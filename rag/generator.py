"""
rag/generator.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Takes the merged retrieved chunks and the user's question, builds a
  structured prompt, sends it to Groq (llama-3.3-70b), and returns
  the answer with inline citations.

THIS IS WHERE THE LLM LEARNS WHICH IS TEXT AND WHICH IS IMAGE:

  Each retrieved chunk carries a "type" field ("text" or "image")
  from the parser. The prompt builder uses this to format each
  chunk differently:

  TEXT chunk → formatted as readable passage the LLM can quote from:
    ┌──────────────────────────────────────────────────────────────┐
    │ [SOURCE 1 | TYPE: text | FILE: report.pdf | PAGE: 3]        │
    │ Revenue grew 23% year-over-year driven by APAC expansion...  │
    └──────────────────────────────────────────────────────────────┘

  IMAGE chunk → formatted as a visual marker the LLM cannot read but
  can acknowledge and cite:
    ┌──────────────────────────────────────────────────────────────┐
    │ [SOURCE 2 | TYPE: image | FILE: report.pdf | PAGE: 5]       │
    │ This source is a visual/diagram/chart. You cannot see the    │
    │ image content directly. Acknowledge it exists and cite it if │
    │ relevant to the question.                                    │
    └──────────────────────────────────────────────────────────────┘

  The system prompt then instructs the LLM:
    - Answer from text sources directly
    - For image sources, acknowledge they exist and tell the user
      to check the visual in the Sources section
    - Always cite with [Source N] notation

WHY THIS APPROACH:
  Groq 70B is a text-only model — it literally cannot process image
  pixels. But by telling it "Source 2 is a visual on page 5", it can
  still produce useful output like:
    "The revenue trend is discussed in the text [Source 1].
     Additionally, there appears to be a relevant chart on page 5
     [Source 2] — see the Sources section below for the visual."

  This is honest, useful, and demonstrates the retrieval working.
─────────────────────────────────────────────────────────────────────────────
"""

from groq import Groq
import streamlit as st


def get_groq_client() -> Groq:
    """Initialize Groq client with API key from Streamlit secrets."""
    return Groq(api_key=st.secrets["groq"]["api_key"])


# ── PROMPT BUILDER ────────────────────────────────────────────────────────

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build the full prompt from retrieved chunks.

    Each chunk is labeled with its source number, type, filename, and page.
    This labeling is what enables the LLM to produce [Source N] citations.

    The type field ("text" or "image") determines how the chunk is presented:
      - Text chunks: show the actual content the LLM can read and quote
      - Image chunks: show a placeholder so the LLM knows a visual exists
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_label = (
            f"[SOURCE {i} | "
            f"TYPE: {chunk['type']} | "
            f"FILE: {chunk['source']} | "
            f"PAGE: {chunk['page']}]"
        )

        if chunk["type"] == "text":
            # LLM can read and quote from this directly
            context_parts.append(
                f"{source_label}\n"
                f"{chunk['content']}"
            )
        else:
            # Image — LLM cannot see pixels, but knows a visual exists here
            context_parts.append(
                f"{source_label}\n"
                f"This source is a visual page (chart, diagram, figure, or table). "
                f"You cannot see the image content directly. "
                f"If this page appears relevant to the question, acknowledge that "
                f"a visual exists on this page and cite it with [Source {i}], "
                f"and tell the user to refer to the Sources section to view it."
            )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise document Q&A assistant. Answer the user's question using ONLY the sources provided below.

INSTRUCTIONS:
1. Read all sources carefully before answering.
2. For TEXT sources: quote or paraphrase the content and cite with [Source N].
3. For IMAGE sources: you cannot see the visual — if relevant, say a chart/figure exists on that page and cite with [Source N] so the user can check it.
4. Every factual claim in your answer MUST have a [Source N] citation.
5. If multiple sources support a claim, cite all of them: [Source 1][Source 3].
6. If the answer is not in the provided sources, say: "I couldn't find this information in the uploaded document."
7. Do not use any knowledge outside the provided sources.
8. Keep the answer clear and structured. Use bullet points if listing multiple facts.

SOURCES:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


# ── GENERATION ────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Send the prompt to Groq llama-3.3-70b and return the answer.

    Model choice — llama-3.3-70b-versatile:
      - Best free model on Groq for synthesis and instruction following
      - Strong at following citation formatting instructions
      - Free tier: ~1000 req/day, 6000 tokens/min
      - For a demo this is more than sufficient

    Temperature = 0.1:
      Low temperature makes the model more faithful to the source material
      and less likely to hallucinate facts not in the retrieved chunks.
      For RAG, you want deterministic, grounded answers — not creativity.
    """
    client = get_groq_client()
    prompt = build_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document Q&A assistant. "
                    "You answer questions strictly from provided document sources. "
                    "You always cite sources using [Source N] notation. "
                    "You never make up information not present in the sources."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content