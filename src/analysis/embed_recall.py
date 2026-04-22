#!/usr/bin/env python3
"""
Embedding-based semantic recall over ALL bili comments (no regex prefilter).

1. Load every comment (L1 + L2) from data/raw/bili_video_*.json
2. Embed with BAAI/bge-small-zh-v1.5 (Chinese sentence transformer, 512-d)
3. Define ~10 anchor queries describing personal pity reports
4. Cosine sim per comment vs each anchor → take max
5. Rank, take top K (default 5000) → write data/processed/embed_candidates.jsonl

Usage: python3 embed_recall.py [--top 5000] [--model BAAI/bge-small-zh-v1.5]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ANCHORS = [
    "我刷了多少只污染才出异色精灵",
    "我抓了多少次噩梦才出异色",
    "我用了多少个球出货异色",
    "我吃了 80 只保底才出异色",
    "终于在第几次保底出了异色",
    "几次污染才出异色，太肝了",
    "出货异色的污染数量记录",
    "刷了几百只才出一只异色，非酋",
    "异色保底我已经累计多少次了，还没出",
    "我家这只异色花了多少噩梦",
    "今天用果实刷了 N 只污染出了异色",
    "1.8% 概率出异色，我打了多少只",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_all_comments():
    """Return list of {bvid, video_title, rpid, parent_rpid, uname, like, text}."""
    rows = []
    for path in sorted(RAW.glob("bili_video_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        v = d["video_meta"]
        for c in d["comments"]:
            for src in [(None, c)] + [(c["rpid"], sr) for sr in c.get("sub_replies", [])]:
                parent, item = src
                text = (item.get("content") or "").strip()
                if not text:
                    continue
                rows.append({
                    "bvid": v["bvid"],
                    "video_title": v["title"],
                    "rpid": item["rpid"],
                    "parent_rpid": parent,
                    "uname": item.get("uname"),
                    "like": item.get("like", 0),
                    "text": text[:600],
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5000)
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    log(f"Loading comments...")
    comments = load_all_comments()
    log(f"Loaded {len(comments)} comments (L1+L2)")

    log(f"Loading model {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 256

    # Embed anchors
    log(f"Embedding {len(ANCHORS)} anchors")
    anchor_emb = model.encode(ANCHORS, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=False)
    anchor_emb = np.asarray(anchor_emb, dtype=np.float32)

    # Embed comments
    log(f"Embedding {len(comments)} comments (batch={args.batch})...")
    t0 = time.time()
    texts = [c["text"] for c in comments]
    emb = model.encode(texts, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    log(f"Embedding done in {time.time()-t0:.1f}s; shape={emb.shape}")

    # Score: cosine sim with each anchor → take max
    log("Computing similarity scores...")
    scores = emb @ anchor_emb.T    # (N_comments, N_anchors)
    max_sim = scores.max(axis=1)
    best_anchor = scores.argmax(axis=1)

    # Sort
    order = np.argsort(-max_sim)
    keep = order[: args.top]

    # Write candidate set
    out_path = OUT / "embed_candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for rank, idx in enumerate(keep):
            row = dict(comments[idx])
            row["embed_score"] = float(max_sim[idx])
            row["best_anchor_idx"] = int(best_anchor[idx])
            row["best_anchor_text"] = ANCHORS[best_anchor[idx]]
            row["rank"] = rank + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"Wrote {out_path} (top {args.top} of {len(comments)})")

    # Save scores for later analysis
    np.savez_compressed(OUT / "embed_scores.npz",
                        scores=scores.astype(np.float16),
                        rpids=np.array([c["rpid"] for c in comments], dtype=np.int64))
    log(f"Wrote embed_scores.npz")

    # Quick stats
    log(f"Score distribution:")
    log(f"  min={max_sim.min():.3f} median={np.median(max_sim):.3f} max={max_sim.max():.3f}")
    log(f"  top {args.top}: min={max_sim[keep].min():.3f} median={np.median(max_sim[keep]):.3f}")
    log(f"  threshold for top: {max_sim[keep[-1]]:.3f}")

    # Show top 10 samples
    print()
    log("=== Top 10 samples by embed score ===")
    for i, idx in enumerate(keep[:10]):
        c = comments[idx]
        print(f"  [{i+1}] score={max_sim[idx]:.3f} anchor={best_anchor[idx]} like={c['like']}")
        print(f"      {c['text'][:160]}")


if __name__ == "__main__":
    main()
