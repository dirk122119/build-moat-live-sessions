from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

from .indexer import build_index
from .retrieval import query, stream_query
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()

CHAT_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Knowledge Base Q&A</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; }
      textarea { width: 100%; height: 80px; }
      button { margin-top: 8px; }
      pre { white-space: pre-wrap; background: #f6f6f6; padding: 12px; }
    </style>
  </head>
  <body>
    <h1>Knowledge Base Q&A</h1>
    <textarea id="query" placeholder="Ask a question..."></textarea>
    <br />
    <button id="ask">Ask</button>
    <h2>Sources</h2>
    <pre id="sources"></pre>
    <h2>Answer</h2>
    <pre id="answer"></pre>
    <script>
      const queryEl = document.querySelector("#query");
      const sourcesEl = document.querySelector("#sources");
      const answerEl = document.querySelector("#answer");

      document.querySelector("#ask").addEventListener("click", async () => {
        sourcesEl.textContent = "";
        answerEl.textContent = "";

        const response = await fetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: queryEl.value }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\\n\\n");
          buffer = events.pop();

          for (const rawEvent of events) {
            const event = rawEvent.split("\\n").find((line) => line.startsWith("event: "))?.slice(7);
            const dataLine = rawEvent.split("\\n").find((line) => line.startsWith("data: "));
            if (!event || !dataLine) continue;

            const data = JSON.parse(dataLine.slice(6));
            if (event === "sources") sourcesEl.textContent = JSON.stringify(data, null, 2);
            if (event === "token") answerEl.textContent += data;
          }
        }
      });
    </script>
  </body>
</html>
"""


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/", response_class=HTMLResponse)
def chat_page():
    return CHAT_PAGE


@router.post("/index", response_model=IndexResponse)
def index_docs():
    files_count, sections_count = build_index()
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return query(req.query)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_query(req.query),
        media_type="text/event-stream",
    )
