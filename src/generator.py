"""
LLM Generator
==============
Generates answers from retrieved context using:
  - Groq API   (if GROQ_API_KEY is set) — default, free
  - OpenAI API (if OPENAI_API_KEY is set)
  - Local HuggingFace flan-t5-base (fallback, offline)

Study modes: qa | summarize | exam_questions | viva | formulas | explain
"""

from __future__ import annotations

import os
from typing import List, Tuple

from src.ingestion import Document


SYSTEM_PROMPT = (
    "You are an expert AI study assistant for college students. "
    "You answer questions STRICTLY based on the provided study material context. "
    "If the answer is not in the context, say 'I could not find this in the uploaded notes.' "
    "Be concise, accurate, and student-friendly. "
    "Use bullet points, numbered lists, and clear headings where appropriate."
)

STUDY_MODE_INSTRUCTIONS = {
    "qa": "Answer the student's question based only on the provided context.",
    "summarize": (
        "Summarize the provided context in a structured way. "
        "Use headings, key points, and a brief conclusion. "
        "Make it useful for exam revision."
    ),
    "exam_questions": (
        "Generate 10 important exam questions based on the provided context. "
        "Include a mix of short-answer and long-answer questions. "
        "Format: numbered list."
    ),
    "viva": (
        "Generate 10 likely viva voce questions based on the provided context. "
        "These should test deep understanding. "
        "Format: numbered list."
    ),
    "formulas": (
        "Extract and list all mathematical formulas, equations, and key definitions "
        "from the provided context. Format clearly with labels."
    ),
    "explain": (
        "Explain the topic in simple terms as if teaching a beginner. "
        "Use analogies, examples, and step-by-step explanations."
    ),
}


def _build_prompt(
    query: str,
    context_docs: List[Tuple[Document, float]],
    mode: str = "qa",
) -> Tuple[str, str]:
    """Build system and user prompts."""
    context_parts = []
    for i, (doc, score) in enumerate(context_docs, 1):
        context_parts.append(
            f"[Source {i}: {doc.source}, chunk {doc.chunk_id}]\n{doc.text}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    mode_instruction = STUDY_MODE_INSTRUCTIONS.get(
        mode, STUDY_MODE_INSTRUCTIONS["qa"]
    )

    user_prompt = (
        f"STUDY MATERIAL CONTEXT:\n\n{context_str}\n\n"
        f"TASK: {mode_instruction}\n\n"
        f"QUERY: {query}"
    )
    return SYSTEM_PROMPT, user_prompt


def generate_answer(
    query: str,
    context_docs: List[Tuple[Document, float]],
    mode: str = "qa",
    model_size: str = "small",
) -> str:
    """
    Generate an answer. Routes to Groq -> OpenAI -> Local HF in that order.
    """
    system_prompt, user_prompt = _build_prompt(query, context_docs, mode)

    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if groq_key:
        return _generate_groq(system_prompt, user_prompt, model_size, groq_key)
    elif openai_key:
        return _generate_openai(system_prompt, user_prompt, model_size, openai_key)
    else:
        return _generate_local(user_prompt)


def _generate_groq(
    system_prompt: str,
    user_prompt: str,
    model_size: str,
    api_key: str,
) -> str:
    """Use Groq API (OpenAI-compatible, free tier)."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        groq_model = (
            "llama-3.3-70b-versatile"
            if model_size == "large"
            else "llama-3.1-8b-instant"
        )
        print(f"[Generator] Using Groq -> {groq_model}")
        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Generator] Groq error: {e}")
        return _generate_local(user_prompt)


def _generate_openai(
    system_prompt: str,
    user_prompt: str,
    model_size: str,
    api_key: str,
) -> str:
    """Use OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini" if model_size == "small" else "gpt-4o"
        print(f"[Generator] Using OpenAI -> {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Generator] OpenAI error: {e}")
        return _generate_local(user_prompt)


def _generate_local(user_prompt: str) -> str:
    """Fallback: use local flan-t5-base model."""
    try:
        from transformers import pipeline
        print("[Generator] Using local flan-t5-base (offline fallback)")
        gen = pipeline("text2text-generation", model="google/flan-t5-base")
        truncated = user_prompt[:1500]
        result = gen(truncated, max_new_tokens=300)[0]["generated_text"]
        return result
    except Exception as e:
        return f"[Generator Error] Could not generate a response: {e}"
