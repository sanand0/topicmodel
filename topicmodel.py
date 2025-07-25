# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pandas", "numpy", "scikit-learn", "tiktoken", "tqdm"]
# ///
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import tiktoken
from sklearn.cluster import KMeans
from tqdm import tqdm

cache_path = Path(os.getenv("LLM_CACHE", Path.home() / ".cache" / "llmfoundry" / "embeddings.db"))


def cache_conn() -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, data BLOB)")
    return conn


def load_data(text: str, fmt: str | None = None) -> tuple[pd.DataFrame, str]:
    if os.path.exists(text):
        ext = os.path.splitext(text)[1].lower()
        fmt = {".csv": "csv", ".txt": "txt"}.get(ext, "json")
        text = open(text).read()
    if fmt == "csv":
        df = pd.read_csv(io.StringIO(text))
    elif fmt == "txt":
        df = pd.DataFrame({"text": text.splitlines()})
    else:
        df = pd.DataFrame(json.loads(text))
    return df, df.columns[0]


async def embed(texts: list[str], model: str) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    conn = cache_conn()
    keys = [hashlib.sha256(f"{model}\n{t}".encode()).hexdigest() for t in texts]
    ph = ",".join("?" * len(keys))
    cached = {
        k: np.frombuffer(b, np.float32)
        for k, b in conn.execute(f"SELECT key, data FROM cache WHERE key IN ({ph})", keys)
    }
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cached]
    if missing:
        api_key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        enc = tiktoken.encoding_for_model(model)
        counts = [len(enc.encode(t)) for _, t in missing]
        chunks, chunk, count = [], [], 0
        for text, tok in zip([t for _, t in missing], counts):
            if count + tok > 8192 and chunk:
                chunks.append(chunk)
                chunk, count = [], 0
            chunk.append(text)
            count += tok
        if chunk:
            chunks.append(chunk)
        result = []
        async with httpx.AsyncClient(timeout=300) as client:
            for chunk in tqdm(chunks, desc="embed", unit="doc"):
                res = await client.post(
                    f"{base}/embeddings",
                    headers=headers,
                    json={"model": model, "input": chunk},
                )
                if res.status_code != 200:
                    print(res.text)
                res.raise_for_status()
                result.extend([d["embedding"] for d in res.json()["data"]])
        for (key, _), emb in zip(missing, result):
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)",
                (key, np.array(emb, np.float32).tobytes()),
            )
            cached[key] = np.array(emb, np.float32)
        conn.commit()
    conn.close()
    return np.stack([cached[k] for k in keys])


async def chat(model: str, system: str, user: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "topics",
                "schema": {
                    "type": "object",
                    "properties": {
                        "topics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "topic": {"type": "string"},
                                },
                                "required": ["id", "topic"],
                            },
                        }
                    },
                    "required": ["topics"],
                },
            },
        },
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        res = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
        if res.status_code != 200:
            print(res.text)
        res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


async def similarity(args: argparse.Namespace, fmt: str, out: io.TextIOBase) -> None:
    docs_df, doc_key = load_data(args.docs)
    topics_df, topic_key = load_data(args.topics)
    docs = docs_df[doc_key].astype(str).tolist()
    topics = topics_df[topic_key].astype(str).tolist()
    doc_emb = await embed(docs, args.model)
    topic_emb = await embed(topics, args.model)
    sim = doc_emb @ topic_emb.T
    best = [topics[i] for i in sim.argmax(1)]
    if fmt == "json":
        rows = []
        for doc, match, row in zip(docs, best, sim):
            data = {"doc": doc, "best_match": match}
            data.update({t: float(v) for t, v in zip(topics, row)})
            rows.append(data)
        json.dump(rows, out, indent=2)
        out.write("\n")
        return
    header = [doc_key, "best_match", *topics]
    rows = [[d, m, *[f"{v:.5f}" for v in r]] for d, m, r in zip(docs, best, sim)]
    if fmt == "csv":
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)
        return
    out.write("\t".join(header) + "\n")
    for row in rows:
        out.write("\t".join(map(str, row)) + "\n")


async def cluster(args: argparse.Namespace, fmt: str, out: io.TextIOBase) -> None:
    df, key = load_data(args.docs)
    docs = df[key].astype(str).tolist()
    emb = await embed(docs, args.model)
    km = KMeans(args.ntopics, n_init="auto", random_state=0).fit(emb)
    samples = [
        {
            "id": i,
            "docs": [
                t[: args.truncate]
                for t in np.array(docs)[km.labels_ == i][: args.nsamples].tolist()
            ],
        }
        for i in range(args.ntopics)
    ]
    system = f'{args.prompt}\nReturn JSON {{"topics": [{{"id": integer, "topic": string}}]}}'
    names = json.loads(await chat(args.name_model, system, json.dumps(samples))).get("topics", [])
    names = sorted(names, key=lambda d: d.get("id", 0))[: args.ntopics]
    topics = [n.get("topic", f"Topic {i + 1}") for i, n in enumerate(names)]
    for i, name in enumerate(topics, 1):
        print(f"{i}: {name}")
    args.topics = json.dumps([{key: t} for t in topics])
    args.docs = json.dumps(df.to_dict(orient="records"))
    await similarity(args, fmt, out)


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True)
    p.add_argument("--topics")
    p.add_argument("--output")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--ntopics", type=int, default=20)
    p.add_argument("--name_model", default="gpt-4.1-nano")
    p.add_argument("--nsamples", type=int, default=5)
    p.add_argument("--truncate", type=int, default=200)
    p.add_argument(
        "--prompt",
        default=(
            "Here are clusters of documents. Suggest 2-4 word topic names for each cluster. "
            "Capture the spirit of each cluster. Differentiate from other clusters."
        ),
    )
    return p.parse_args(argv)


async def amain(argv: list[str]) -> None:
    args = parse(argv)
    ext_map = {".csv": "csv", ".json": "json", ".txt": "txt"}
    fmt = "txt"
    out = sys.stdout
    if args.output:
        ext = Path(args.output).suffix.lower()
        fmt = ext_map.get(ext)
        if not fmt:
            raise SystemExit("output must end with .csv, .json or .txt")
        out = open(args.output, "w")
    try:
        if args.topics:
            await similarity(args, fmt, out)
        else:
            await cluster(args, fmt, out)
    finally:
        if out is not sys.stdout:
            out.close()


def main(argv: list[str] | None = None) -> int:
    import asyncio

    asyncio.run(amain(argv or sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
