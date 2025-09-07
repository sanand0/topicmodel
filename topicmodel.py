# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pandas", "numpy", "scikit-learn", "tiktoken", "tqdm"]
# ///
"""Cluster documents or match them to topics using OpenAI embeddings."""

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
from typing import List
import umap
from affine import Affine
from shapely.geometry import shape as shp_shape, MultiPoint, Polygon
from rasterio.features import shapes

# cache embeddings in a shared SQLite database
cache_path = Path(
    os.getenv("TOPICMODEL_CACHE", Path.home() / ".cache" / "topicmodel" / "embeddings.db")
)


def cache_conn() -> sqlite3.Connection:
    """Return SQLite connection to cache."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_path)
    conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, data BLOB)")
    return conn


def load_data(text: str, fmt: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load .txt, .csv, .json file or JSON string and return DataFrame + first column"""
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
    """Return embeddings for texts, caching results."""
    if not texts:
        return np.empty((0, 0))
    conn = cache_conn()
    keys = [hashlib.sha256(f"{model}\n{t}".encode()).hexdigest() for t in texts]
    cached: dict[str, np.ndarray] = {}
    # Check cache in batches to stay under SQLite limit of 32K wildcard keys
    for i in range(0, len(keys), 32000):
        batch = keys[i : i + 32000]
        ph = ",".join("?" * len(batch))
        cached.update(
            {
                k: np.frombuffer(b, np.float32)
                for k, b in conn.execute(
                    f"SELECT key, data FROM cache WHERE key IN ({ph})",
                    batch,
                )
            }
        )
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
                payload = {"model": model, "input": chunk}
                res = await client.post(f"{base}/embeddings", headers=headers, json=payload)
                if res.status_code != 200:
                    print(f"URL: {base}/embeddings\nREQUEST\n{payload}\n\nRESPONSE\n{res.text}")
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
    """Query the chat model to name clusters, return JSON response."""
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
    }
    async with httpx.AsyncClient(timeout=300) as client:
        res = await client.post(f"{base}/chat/completions", headers=headers, json=payload)
        if res.status_code != 200:
            print(f"URL: {base}/chat/completions\nREQUEST\n{payload}\n\nRESPONSE\n{res.text}")
        res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


async def similarity(args: argparse.Namespace, fmt: str, out: io.TextIOBase) -> None:
    """Compute document-topic similarity scores. Write similarity matrix to output."""
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
            data = {"doc": doc, "best_match": match, "best_score": float(row.max())}
            data.update({t: float(v) for t, v in zip(topics, row)})
            rows.append(data)
        json.dump(rows, out, indent=2)
        out.write("\n")
        return
    header = [doc_key, "best_match", "best_score", *topics]
    rows = [[d, m, f"{r.max():.5f}", *[f"{v:.5f}" for v in r]] for d, m, r in zip(docs, best, sim)]
    if fmt == "csv":
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)
        return
    out.write("\t".join(header) + "\n")
    for row in rows:
        out.write("\t".join(map(str, row)) + "\n")

def visualize_embeddings(emb: np.ndarray, labels: np.ndarray, topics: List[str], filename: str) -> None:
    if emb.shape[0] <= 1:
        print("Not enough data points to visualize")
        return

    # Dimensionality reduction
    emb_2d = umap.UMAP(n_components=2, n_neighbors=min(15, emb.shape[0] - 1)).fit_transform(emb)
    label_list = sorted(np.unique(labels))
    k = len(label_list)

    # Map compact index -> topic name
    if len(topics) == k:
        idx_to_topic = dict(enumerate(topics))
    else:
        idx_to_topic = {i: (topics[int(lab)] if int(lab) < len(topics) else f"Topic {int(lab)}") for i, lab in enumerate(label_list)}

    # Centroids and padded bounds
    centroids = np.vstack([
        emb_2d[labels == lab].mean(axis=0) if np.any(labels == lab) else [0, 0]
        for lab in label_list
    ])
    (min_x, min_y), (max_x, max_y) = emb_2d.min(0), emb_2d.max(0)
    rngx, rngy = max_x - min_x, max_y - min_y
    pad = 0.02 * max(rngx if rngx > 0 else 1, rngy if rngy > 0 else 1)
    min_x, min_y, max_x, max_y = min_x - pad, min_y - pad, max_x + pad, max_y + pad
    rngx, rngy = max_x - min_x, max_y - min_y

    # Dynamic grid
    base = int(min(600, max(200, emb.shape[0] * 2)))
    W = max(60, base if rngx >= rngy else int(round(base * rngx / rngy)))
    H = max(60, int(round(base * rngy / rngx)) if rngx >= rngy else base)
    dx, dy = rngx / W, rngy / H

    # Assign each grid cell center to nearest centroid
    xs = min_x + dx * (np.arange(W) + 0.5)
    ys = min_y + dy * (np.arange(H) + 0.5)
    Xc, Yc = np.meshgrid(xs, ys)
    grid_pts = np.c_[Xc.ravel(), Yc.ravel()]
    idx_assign = ((grid_pts[:, None, :] - centroids[None, :, :])**2).sum(2).argmin(1).reshape(H, W)

    # Polygonize labeled grid into gapless polygons
    transform = Affine.translation(min_x, min_y) * Affine.scale(dx, dy)
    best_poly_by_idx = {i: None for i in range(k)}
    for geom, val in shapes(idx_assign.astype(np.int32), transform=transform):
        i = int(val)  # compact cluster index (0..k-1)
        g = shp_shape(geom)  # Polygon or MultiPolygon
        # Pick largest polygon for each cluster
        poly = max((g.geoms if g.geom_type == "MultiPolygon" else [g]), key=lambda p: p.area, default=None)
        if poly and (best_poly_by_idx[i] is None or poly.area > best_poly_by_idx[i].area):
            best_poly_by_idx[i] = poly

    # Fallbacks for clusters without grid polygons
    features = []
    for i in range(k):
        poly = best_poly_by_idx[i]
        lab = label_list[i]
        pts = emb_2d[labels == lab]
        if poly is None or poly.is_empty or poly.area == 0:
            if len(pts) == 0:
                cx, cy = centroids[i]
                eps = max(rngx, rngy) * 0.01
                poly = Polygon([(cx-eps, cy-eps), (cx+eps, cy-eps), (cx+eps, cy+eps), (cx-eps, cy+eps)])
            elif len(pts) == 1:
                cx, cy = pts[0]
                eps = max(rngx, rngy) * 0.005
                poly = Polygon([(cx-eps, cy-eps), (cx+eps, cy-eps), (cx+eps, cy+eps), (cx-eps, cy+eps)])
            else:
                hull = MultiPoint(pts).convex_hull
                if hull.geom_type != "Polygon" or hull.area == 0:
                    c = hull.centroid if not hull.is_empty else None
                    cx, cy = (c.x, c.y) if c else centroids[i]
                    eps = max(rngx, rngy) * 0.005
                    poly = Polygon([(cx-eps, cy-eps), (cx+eps, cy-eps), (cx+eps, cy+eps), (cx-eps, cy+eps)])
                else:
                    poly = hull
        cen = poly.centroid if poly.area > 0 else None
        centroid_out = [float(cen.x), float(cen.y)] if cen else [float(centroids[i, 0]), float(centroids[i, 1])]
        coords = [list(map(float, c)) for c in np.asarray(poly.exterior.coords)]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "topic": idx_to_topic.get(i, topics[i] if i < len(topics) else f"Topic {i}"),
                "cluster_id": int(lab),
                "centroid": centroid_out
            }
        })

    with open(filename, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh, indent=2)


async def cluster(args: argparse.Namespace, fmt: str, out: io.TextIOBase) -> None:
    """Cluster documents to discover topics and name them via chat()."""
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
    prompt = args.prompt
    if args.hierarchy:
        prompt += f"""\nName topics as hierarchies separated by ` / `, e.g. parent / child / ... . {args.hierarchy}"""

    system = f'{prompt}\nReturn JSON {{"topics": [{{"id": integer, "topic": string}}]}}'
    names = json.loads(await chat(args.name_model, system, json.dumps(samples))).get("topics", [])
    names = sorted(names, key=lambda d: d.get("id", 0))[: args.ntopics]
    topics = [n.get("topic", f"Topic {i + 1}") for i, n in enumerate(names)]
    for i, name in enumerate(topics, 1):
        print(f"{i}: {name}")
    args.topics = json.dumps([{key: t} for t in topics])
    args.docs = json.dumps(df.to_dict(orient="records"))
    await similarity(args, fmt, out)

    if args.plot:
        visualize_embeddings(emb, km.labels_, topics, args.plot)

def parse(argv: list[str]) -> argparse.Namespace:
    """Return parsed command line arguments."""
    p = argparse.ArgumentParser(
        description="Automatically discover topics from documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("docs", help="File with docs: .txt, .csv, .json")
    p.add_argument("--topics", help="File with topics: .txt, .csv, .json")
    p.add_argument("--output", help="Output file: .txt, .csv, .json")
    p.add_argument("--model", default="text-embedding-3-small", help="Embedding model")
    p.add_argument("--name_model", default="gpt-5-mini", help="Topic naming model")
    p.add_argument("--ntopics", type=int, default=20, help="Approx # of topics to generate")
    p.add_argument("--nsamples", type=int, default=5, help="# docs to send for naming")
    p.add_argument("--truncate", type=int, default=200, help="Send first N chars of each doc")
    p.add_argument("--hierarchy", nargs="?", const="2 level depth", help="Instruction to create hierarchical topic names")
    p.add_argument("--plot", type=str, help="Visualize document clusters")
    p.add_argument(
        "--prompt",
        help="Prompt used to name topics",
        default=(
            "Here are clusters of documents. Suggest 2-4 word topic names for each cluster. "
            "Capture the spirit of each cluster. Differentiate from other clusters."
        ),
    )
    return p.parse_args(argv)


async def amain(argv: list[str]) -> None:
    """Run the tool with parsed arguments."""
    args = parse(argv)
    ext_map = {".csv": "csv", ".json": "json", ".txt": "txt"}
    fmt = "txt"
    out = sys.stdout
    if args.output:
        ext = Path(args.output).suffix.lower()
        fmt = ext_map.get(ext)
        if not fmt:
            raise SystemExit(f"Output {args.output} must end with .csv, .json or .txt")
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
    """Entry point for the `topicmodel` script."""
    import asyncio

    asyncio.run(amain(argv or sys.argv[1:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
