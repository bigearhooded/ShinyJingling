#!/usr/bin/env python3
"""Build a candidates file: all bili comments with at least one digit + a pity-relevant keyword."""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

DIGIT = re.compile(r"\d")
KW = re.compile(r"污染|噩梦|异色|保底|刷|抓|肝|球|出货|梦魇")

candidates = []
for path in sorted(RAW.glob("bili_video_*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    v = d["video_meta"]
    for c in d["comments"]:
        for src in [(None, c)] + [(c["rpid"], sr) for sr in c.get("sub_replies", [])]:
            parent, item = src
            text = item.get("content", "") or ""
            if not DIGIT.search(text) or not KW.search(text):
                continue
            candidates.append({
                "bvid": v["bvid"],
                "video_title": v["title"],
                "rpid": item["rpid"],
                "parent_rpid": parent,
                "uname": item.get("uname"),
                "like": item.get("like", 0),
                "text": text[:500],  # cap individual text length
            })

out_path = OUT / "llm_candidates.jsonl"
with out_path.open("w", encoding="utf-8") as fh:
    for c in candidates:
        fh.write(json.dumps(c, ensure_ascii=False) + "\n")
print(f"Wrote {out_path} with {len(candidates)} candidates")
total_chars = sum(len(c["text"]) for c in candidates)
print(f"Total text chars: {total_chars:,} (~{total_chars//4:,} tokens)")
