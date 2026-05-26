import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    # TODO: Parse one Markdown file into section-level records.
    #
    # Design decision: The retrieval unit is a heading section, not a whole file.
    #
    # Hints:
    # 1. Use HEADING_RE to detect Markdown headings.
    # 2. Track heading_path so citations include parent context.
    # 3. Each Section id should look like "refund_policy.md#refund-timeline".
    # 4. Tokens should include both headings and content.
    parsed_sections: list[Section] = []
    heading_stack: list[str] = []
    current_heading: str | None = None
    current_content: list[str] = []
    seen_ids: Counter[str] = Counter()

    def flush_section() -> None:
        if current_heading is None:
            return

        content = "\n".join(current_content).strip()
        if not content:
            return

        base_id = f"{path.name}#{slugify(current_heading)}"
        seen_ids[base_id] += 1
        section_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"
        section_heading_path = heading_stack.copy()
        token_text = " ".join([*section_heading_path, content])

        parsed_sections.append(
            Section(
                id=section_id,
                file=path.name,
                heading=current_heading,
                heading_path=section_heading_path,
                content=content,
                tokens=tokenize(token_text),
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
    return parsed_sections


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    # TODO: Persist the section index to .kb/index.json so it is inspectable.
    #
    # Hints:
    # 1. Create index_path.parent if it does not exist.
    # 2. Write {"sections": [...], "stats": {...}} as pretty JSON.
    # 3. Use section.to_dict() for each Section.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [section.to_dict() for section in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
            "vocabulary_size": len(doc_freq),
        },
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rebuild_stats() -> None:
    # TODO: Rebuild doc_freq, avg_doc_len, and files_indexed from sections.
    #
    # Hints:
    # 1. files_indexed can be derived from the unique section.file values.
    # 2. doc_freq counts how many sections contain each token.
    # 3. avg_doc_len is the average token count across sections.
    global doc_freq, avg_doc_len, files_indexed

    files_indexed = len({section.file for section in sections})
    doc_freq = Counter()

    for section in sections:
        doc_freq.update(set(section.tokens))

    if sections:
        avg_doc_len = sum(len(section.tokens) for section in sections) / len(sections)
    else:
        avg_doc_len = 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    # TODO: Load .kb/index.json into the in-memory sections list.
    #
    # Hints:
    # 1. If index_path does not exist, return (0, 0).
    # 2. Read payload["sections"] and convert each item back to Section.
    # 3. Call rebuild_stats() after assigning sections.
    # 4. Return (files_indexed, sections_indexed).
    global sections

    if not index_path.exists():
        return 0, 0

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sections = []
    for item in payload.get("sections", []):
        section = Section(
            id=item["id"],
            file=item["file"],
            heading=item["heading"],
            heading_path=item["heading_path"],
            content=item["content"],
            tokens=item["tokens"],
        )
        sections.append(section)
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections

    # TODO: Build an in-memory section index from docs/*.md.
    #
    # Hints:
    # 1. Read all Markdown files from docs_dir.
    # 2. Call parse_markdown() for each file.
    # 3. Call rebuild_stats() to compute BM25 metadata.
    # 4. Persist .kb/index.json with write_index_json().
    # 5. Call write_index_json() so students can inspect the generated index.
    # 6. Return (files_indexed, sections_indexed).
    for path in sorted(docs_dir.glob("*.md")):
        sections.extend(parse_markdown(path))
    rebuild_stats()
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    # TODO: Score one section for the query using BM25.
    #
    # Hints:
    # 1. Count term frequency in the section.
    # 2. Use doc_freq to give rare terms higher weight.
    # 3. Normalize by section length using avg_doc_len.
    # 4. Add a small boost when query terms appear in heading_path.
    if not query_tokens or not section.tokens or not sections or avg_doc_len == 0:
        return 0.0

    token_counts = Counter(section.tokens)
    section_len = len(section.tokens)
    total_sections = len(sections)
    score = 0.0

    for token in query_tokens:
        term_frequency = token_counts[token]
        if term_frequency == 0:
            continue

        document_frequency = doc_freq[token]
        inverse_document_frequency = math.log(
            1 + (total_sections - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        length_penalty = 1 - b + b * (section_len / avg_doc_len)
        score += inverse_document_frequency * (
            term_frequency * (k1 + 1)
        ) / (term_frequency + k1 * length_penalty)

    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    if any(token in heading_tokens for token in query_tokens):
        score *= 1.15

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
