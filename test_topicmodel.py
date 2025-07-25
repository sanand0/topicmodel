# usage: uvx --with pytest-asyncio,httpx,pandas,numpy,scikit-learn,tiktoken,tqdm pytest

import json
import pytest


class FakeResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeClient:
    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def post(self, url, *, headers=None, json=None):
        if not self._responses:
            raise AssertionError("Unexpected call")
        return FakeResponse(self._responses.pop(0))


@pytest.mark.asyncio
async def test_similarity(monkeypatch, tmp_path):
    docs = '[{"t":"a"},{"t":"b"}]'
    topics = '[{"t":"x"},{"t":"y"}]'
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("LLM_CACHE", str(tmp_path / "cache.db"))
    import importlib
    import topicmodel

    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain(["--docs", docs, "--topics", topics, "--output", str(out)])
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "x", "x": 1.0, "y": 0.0},
        {"doc": "b", "best_match": "y", "x": 0.0, "y": 1.0},
    ]
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient([]))
    await topicmodel.amain(["--docs", docs, "--topics", topics, "--output", str(out)])


@pytest.mark.asyncio
async def test_cluster(monkeypatch, tmp_path):
    docs = '[{"t":"a"},{"t":"b"},{"t":"c"},{"t":"d"}]'
    responses = [
        {
            "data": [
                {"embedding": [1, 0]},
                {"embedding": [1, 0]},
                {"embedding": [0, 1]},
                {"embedding": [0, 1]},
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": '{"topics": [{"id": 1, "topic": "T1"}, {"id": 2, "topic": "T2"}]}'
                    }
                }
            ]
        },
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("LLM_CACHE", str(tmp_path / "cache.db"))
    import importlib
    import topicmodel

    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain(["--docs", docs, "--output", str(out), "--ntopics", "2"])
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "T1", "T1": 1.0, "T2": 0.0},
        {"doc": "b", "best_match": "T1", "T1": 1.0, "T2": 0.0},
        {"doc": "c", "best_match": "T2", "T1": 0.0, "T2": 1.0},
        {"doc": "d", "best_match": "T2", "T1": 0.0, "T2": 1.0},
    ]


@pytest.mark.asyncio
async def test_txt(monkeypatch, tmp_path):
    docs_file = tmp_path / "docs.TXT"
    docs_file.write_text("a\nb\n")
    topics_file = tmp_path / "topics.csv"
    topics_file.write_text("t\nx\ny\n")
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("LLM_CACHE", str(tmp_path / "cache.db"))
    import importlib
    import topicmodel

    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain(
        [
            "--docs",
            str(docs_file),
            "--topics",
            str(topics_file),
            "--output",
            str(out),
        ]
    )
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "x", "x": 1.0, "y": 0.0},
        {"doc": "b", "best_match": "y", "x": 0.0, "y": 1.0},
    ]
