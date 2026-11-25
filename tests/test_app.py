import os

os.environ["EMBED_MODEL"] = ""          # BM25 only: fast, no model download in CI
os.environ["RATE_LIMIT_PER_MIN"] = "3"

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.rag import load_chunks, rrf  # noqa: E402


class FakeLLM:
    providers = []

    def __init__(self):
        self.last_messages = None

    def chat(self, messages, **_):
        self.last_messages = messages
        return "stub answer"


def test_chunks_have_titles():
    chunks = load_chunks(main.KNOWLEDGE_DIR)
    assert len(chunks) > 10
    assert any("Turing" in c.title for c in chunks)


def test_rrf_prefers_docs_ranked_high_in_both():
    fused = rrf([[1, 2, 3], [1, 3, 2]])
    assert fused[0][0] == 1


def test_retrieval_finds_right_section():
    top = main.retriever.search("What did he do at Turing?", top_k=1)[0]
    assert "Turing" in top.title
    top = main.retriever.search("dental x-ray segmentation", top_k=1)[0]
    assert "Dental" in top.title


def test_chat_endpoint_and_rate_limit(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(main, "llm", fake)
    main._hits.clear()
    client = TestClient(main.app)

    r = client.post("/api/chat", json={"message": "Where does he teach?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "stub answer"
    assert body["sources"]
    assert "GIK" in fake.last_messages[0]["content"]  # context injected into system prompt

    for _ in range(2):
        client.post("/api/chat", json={"message": "hi"})
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 429


def test_rejects_oversized_message():
    client = TestClient(main.app)
    assert client.post("/api/chat", json={"message": "x" * 501}).status_code == 422
