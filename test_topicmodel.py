import csv
import importlib
import json
import pytest
import topicmodel


class FakeResponse:
    """Fake httpx response (optionally non-200) to avoid network calls"""

    def __init__(self, data, *, status_code: int = 200, text: str | None = None):
        self._data = data
        self.status_code = status_code
        # text shown in failed-request log
        self.text = text or json.dumps(data)

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
        resp = self._responses.pop(0)
        return resp if isinstance(resp, FakeResponse) else FakeResponse(resp)


@pytest.mark.asyncio
async def test_similarity(monkeypatch, tmp_path):
    """Test matching documents to topics."""
    docs = '[{"t":"a"},{"t":"b"}]'
    topics = '[{"t":"x"},{"t":"y"}]'
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain([docs, "--topics", topics, "--output", str(out)])
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "x", "best_score": 1.0, "x": 1.0, "y": 0.0},
        {"doc": "b", "best_match": "y", "best_score": 1.0, "x": 0.0, "y": 1.0},
    ]
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient([]))
    await topicmodel.amain([docs, "--topics", topics, "--output", str(out)])


@pytest.mark.asyncio
async def test_cluster(monkeypatch, tmp_path):
    """Test topic discovery."""
    docs = '[{"t":"a"},{"t":"b"},{"t":"c"},{"t":"d"}]'
    message = {"content": '{"topics": [{"id": 1, "topic": "T1"}, {"id": 2, "topic": "T2"}]}'}
    responses = [
        {
            "data": [
                {"embedding": [1, 0]},
                {"embedding": [1, 0]},
                {"embedding": [0, 1]},
                {"embedding": [0, 1]},
            ]
        },
        {"choices": [{"message": message}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain([docs, "--output", str(out), "--ntopics", "2"])
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "T1", "best_score": 1.0, "T1": 1.0, "T2": 0.0},
        {"doc": "b", "best_match": "T1", "best_score": 1.0, "T1": 1.0, "T2": 0.0},
        {"doc": "c", "best_match": "T2", "best_score": 1.0, "T1": 0.0, "T2": 1.0},
        {"doc": "d", "best_match": "T2", "best_score": 1.0, "T1": 0.0, "T2": 1.0},
    ]


@pytest.mark.asyncio
async def test_txt(monkeypatch, tmp_path):
    """Test reading mixed-case TXT and CSV files."""
    docs_file = tmp_path / "docs.TXT"
    docs_file.write_text("a\nb\n")
    topics_file = tmp_path / "topics.csv"
    topics_file.write_text("t\nx\ny\n")
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))
    out = tmp_path / "out.json"
    await topicmodel.amain([str(docs_file), "--topics", str(topics_file), "--output", str(out)])
    result = json.loads(out.read_text())
    assert result == [
        {"doc": "a", "best_match": "x", "best_score": 1.0, "x": 1.0, "y": 0.0},
        {"doc": "b", "best_match": "y", "best_score": 1.0, "x": 0.0, "y": 1.0},
    ]


@pytest.mark.asyncio
async def test_csv_output(monkeypatch, tmp_path):
    """--output *.csv should create a well-formed CSV."""
    docs = '[{"t":"a"},{"t":"b"}]'
    topics = '[{"t":"x"},{"t":"y"}]'
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))

    out_csv = tmp_path / "out.csv"
    await topicmodel.amain([docs, "--topics", topics, "--output", str(out_csv)])

    with out_csv.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["t", "best_match", "best_score", "x", "y"]
        rows = list(reader)
        assert rows == [
            ["a", "x", "1.00000", "1.00000", "0.00000"],
            ["b", "y", "1.00000", "0.00000", "1.00000"],
        ]


def test_help_defaults(capsys):
    """--help output should include default values (via ArgumentDefaultsHelpFormatter)."""
    with pytest.raises(SystemExit):
        topicmodel.parse(["--help"])
    out = capsys.readouterr().out
    # One default that must appear in help text
    assert "text-embedding-3-small" in out


@pytest.mark.asyncio
async def test_embedding_error_logging(monkeypatch, capsys, tmp_path):
    """On non-200 response, the full request/response should be printed."""
    docs = '[{"t":"a"}]'
    topics = '[{"t":"x"}]'
    # First API call fails (status 400) → should trigger log print
    responses = [
        FakeResponse({"data": [{"embedding": [1, 0]}]}, status_code=400, text="Bad request"),
        FakeResponse({"data": [{"embedding": [1, 0]}]}),  # succeeding second call
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))

    # Run once; we don’t care about output file contents here
    await topicmodel.amain([docs, "--topics", topics, "--output", str(tmp_path / "o.json")])

    captured = capsys.readouterr().out
    assert "REQUEST" in captured and "RESPONSE" in captured and "Bad request" in captured


@pytest.mark.asyncio
async def test_chat_error_logging(monkeypatch, capsys, tmp_path):
    """Verify that chat() prints full diagnostic info on non-200."""
    docs = '[{"t":"a"},{"t":"b"}]'
    responses = [
        {"data": [{"embedding": [1, 0]}, {"embedding": [0, 1]}]},
        FakeResponse(
            {"choices": [{"message": {"content": '{"topics":[{"id":1,"topic":"T"}]}'}}]},
            status_code=400,
            text="Bad request",
        ),
        {"data": [{"embedding": [1, 0]}]},
    ]
    monkeypatch.setenv("TOPICMODEL_CACHE", str(tmp_path / "cache.db"))
    importlib.reload(topicmodel)
    monkeypatch.setattr(topicmodel.httpx, "AsyncClient", lambda *a, **k: FakeClient(responses))

    await topicmodel.amain([docs, "--output", str(tmp_path / "o.json"), "--ntopics", "1"])

    log = capsys.readouterr().out
    assert "chat/completions" in log and "REQUEST" in log and "Bad request" in log
