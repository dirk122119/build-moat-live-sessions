import json
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer

CANNOT_CONFIRM = "I cannot confirm from the knowledge base."
MIN_RETRIEVAL_SCORE = float(os.getenv("KB_MIN_RETRIEVAL_SCORE", "0.1"))

# TODO: Write the system prompt for the knowledge base Q&A assistant.
#
# Design decision: Hallucination defense for raw Markdown context.
#
# Hints:
# 1. Only answer using the provided CONTEXT.
# 2. Cite only exact source IDs shown in [Source: ...].
#    Each source ID uses filename#heading format.
# 3. Define fallback behavior when the context lacks the answer.
# 4. Explicitly prohibit guessing or outside knowledge.

SYSTEM_PROMPT = """
You are a knowledge base Q&A assistant.

Answer the user's question using only the provided CONTEXT.
Do not use outside knowledge, assumptions, or guesses.

Every factual claim in your answer must be supported by the CONTEXT.
When you use information from a source, cite the exact source ID shown in
[Source: ...]. Source IDs use the format filename.md#heading.

If the CONTEXT does not contain enough information to answer the question,
reply exactly: "I cannot confirm from the knowledge base."
"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_sections: list) -> str:
    # TODO: Build the prompt from top-ranked Markdown sections.
    #
    # Design decision: Put raw Markdown sections into CONTEXT with citations.
    #
    # Hints:
    # 1. Include [Source: filename#heading] before each section.
    # 2. Include heading_path so the model sees the document structure.
    # 3. Include only top sections passed into this function.
    # 4. Place CONTEXT before QUESTION.
    context_blocks = []

    for section, score in ranked_sections:
        context_blocks.append(
            "\n".join(
                [
                    f"[Source: {section.id}]",
                    f"Heading: {' > '.join(section.heading_path)}",
                    f"Retrieval score: {score:.3f}",
                    "",
                    section.content,
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def strong_enough(ranked_sections: list) -> bool:
    # TODO: Tune this threshold with real queries before production use.
    #
    # BM25 scores are higher when query terms strongly match the section.
    return bool(ranked_sections) and ranked_sections[0][1] >= MIN_RETRIEVAL_SCORE


def build_sources(ranked_sections: list) -> list[dict]:
    return [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]


def sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def query(question: str) -> dict:
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)
    if not strong_enough(ranked_sections):
        return {
            "answer": CANNOT_CONFIRM,
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    return {
        "answer": response.content,
        "sources": build_sources(ranked_sections),
    }


def stream_query(question: str):
    # TODO: Stream grounded answers over SSE while preserving /chat behavior.
    if not indexer.sections:
        yield sse_event("sources", [])
        yield sse_event("token", "The knowledge base has not been indexed yet. Call POST /index first.")
        yield sse_event("done", True)
        return

    ranked_sections = indexer.search(question, k=3)
    if not strong_enough(ranked_sections):
        yield sse_event("sources", [])
        yield sse_event("token", CANNOT_CONFIRM)
        yield sse_event("done", True)
        return

    yield sse_event("sources", build_sources(ranked_sections))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ]
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield sse_event("token", chunk.content)

    yield sse_event("done", True)
