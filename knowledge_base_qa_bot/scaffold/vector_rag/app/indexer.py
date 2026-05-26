import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# TODO: Configure chunking parameters for traditional RAG.
#
# Design decision: Balance semantic recall against context noise.
#
# Hints:
# 1. chunk_size around 500 chars is a reasonable prototype default.
# 2. chunk_overlap helps avoid cutting facts at boundaries.
# 3. separators should prefer Markdown structure before individual words.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    # TODO: Load Markdown into source-citable Document records.
    #
    # Design decision: Preserve filename#heading metadata before chunking.
    #
    # Hints:
    # 1. Use HEADING_RE to split by Markdown headings.
    # 2. Put heading_path and content into page_content.
    # 3. Store source metadata like "refund_policy.md#refund-timeline".
    documents: list[Document] = []
    heading_stack: list[str] = []
    current_heading: str | None = None
    current_content: list[str] = []
    seen_ids: dict[str, int] = {}

    def flush_section() -> None:
        if current_heading is None:
            return

        content = "\n".join(current_content).strip()
        if not content:
            return

        base_id = f"{path.name}#{slugify(current_heading)}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        source = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
        heading_path = heading_stack.copy()

        documents.append(
            Document(
                page_content="\n".join(
                    [
                        f"Heading: {' > '.join(heading_path)}",
                        "",
                        content,
                    ]
                ),
                metadata={
                    "source": source,
                    "file": path.name,
                    "heading": " > ".join(heading_path),
                    "heading_path": heading_path,
                },
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_section()

            level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(current_heading)
            current_content = []
            continue

        if current_heading is not None:
            current_content.append(line)

    flush_section()
    return documents


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    # TODO: Build a FAISS vector index from docs/*.md.
    #
    # Hints:
    # 1. Load all Markdown files from docs_dir.
    # 2. Convert each heading section to a Document.
    # 3. Split documents into chunks with splitter.split_documents().
    # 4. Create FAISS.from_documents(chunks, get_embeddings()).
    # 5. Save the FAISS index to .kb/faiss_index/.
    # 6. Return (files_indexed, chunks_indexed).
    markdown_files = sorted(docs_dir.glob("*.md")) if docs_dir.exists() else []
    documents: list[Document] = []

    for path in markdown_files:
        documents.extend(load_markdown_sections(path))

    chunks = splitter.split_documents(documents) if documents else []
    vectorstore = FAISS.from_documents(chunks, get_embeddings()) if chunks else None
    files_indexed = len(markdown_files)
    sections_indexed = len(chunks)
    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    # TODO: Persist the FAISS index so restart does not require re-embedding.
    #
    # Hints:
    # 1. Return early if vectorstore is None.
    # 2. Clear stale persisted files with shutil.rmtree(...) if the new index is empty.
    # 3. Use vectorstore.save_local(str(index_dir)).
    # 4. Write metadata.json with embedding_model, files_indexed, and sections_indexed.
    # 5. json.dumps(..., indent=2) makes the metadata easy to inspect.
    if vectorstore is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    # TODO: Load .kb/faiss_index/ on server startup if it exists.
    #
    # Hints:
    # 1. Check for index.faiss and index.pkl.
    # 2. Read metadata.json and verify embedding_model still matches.
    # 3. Use FAISS.load_local(..., allow_dangerous_deserialization=True).
    # 4. Only use dangerous deserialization for indexes created by this local app.
    index_file = index_dir / "index.faiss"
    pickle_file = index_dir / "index.pkl"
    metadata_file = index_dir / "metadata.json"

    if not (index_file.exists() and pickle_file.exists() and metadata_file.exists()):
        vectorstore = None
        files_indexed = 0
        sections_indexed = 0
        return files_indexed, sections_indexed

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        vectorstore = None
        files_indexed = 0
        sections_indexed = 0
        return files_indexed, sections_indexed

    vectorstore = FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    files_indexed = int(metadata.get("files_indexed", 0))
    sections_indexed = int(metadata.get("sections_indexed", 0))
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
