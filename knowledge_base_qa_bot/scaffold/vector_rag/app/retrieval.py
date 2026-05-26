import json
import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer

CANNOT_CONFIRM = "I cannot confirm from the knowledge base."
MAX_RETRIEVAL_DISTANCE = float(os.getenv("KB_MAX_RETRIEVAL_DISTANCE", "1.4"))


# TODO: Write the system prompt for the knowledge base Q&A assistant.
#
# Design decision: Hallucination defense for retrieved chunks.
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


def build_prompt(query: str, ranked_chunks: list) -> str:
    # TODO: Build the prompt from retrieved vector chunks.
    #
    # Design decision: Give the LLM enough context without flooding it.
    #
    # Hints:
    # 1. Include [Source: filename#heading] before each chunk.
    # 2. Include retrieval distance or score only for debugging.
    # 3. Use top-k chunks passed into this function.
    # 4. Place CONTEXT before QUESTION.
    context_blocks = []

    for doc, score in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        heading = doc.metadata.get("heading", "unknown")
        context_blocks.append(
            "\n".join(
                [
                    f"[Source: {source}]",
                    f"Heading: {heading}",
                    f"Retrieval distance: {float(score):.3f}",
                    "",
                    doc.page_content,
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no context)"
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def strong_enough(ranked_chunks: list) -> bool:
    # TODO: Tune this threshold with real embeddings before production use.
    #
    # FAISS similarity_search_with_score returns distance, so lower is better.
    return bool(ranked_chunks) and float(ranked_chunks[0][1]) <= MAX_RETRIEVAL_DISTANCE


def build_sources(ranked_chunks: list) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]


def sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not strong_enough(ranked_chunks):
        return {
            "answer": CANNOT_CONFIRM,
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    return {
        "answer": response.content,
        "sources": build_sources(ranked_chunks),
    }


def stream_query(question: str):
    # TODO: Stream grounded answers over SSE while preserving /chat behavior.
    if indexer.vectorstore is None:
        yield sse_event("sources", [])
        yield sse_event("token", "The knowledge base has not been indexed yet. Call POST /index first.")
        yield sse_event("done", True)
        return

    ranked_chunks = indexer.search(question, k=3)
    if not strong_enough(ranked_chunks):
        yield sse_event("sources", [])
        yield sse_event("token", CANNOT_CONFIRM)
        yield sse_event("done", True)
        return

    yield sse_event("sources", build_sources(ranked_chunks))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ]
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield sse_event("token", chunk.content)

    yield sse_event("done", True)
