#!/usr/bin/env python3
"""Merge regex prefilter (llm_candidates) + embedding top-K → unified candidate set."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "data" / "processed"

regex_rows = {json.loads(l)["rpid"]: json.loads(l) for l in (OUT/"llm_candidates.jsonl").open(encoding="utf-8")}
embed_rows = {json.loads(l)["rpid"]: json.loads(l) for l in (OUT/"embed_candidates.jsonl").open(encoding="utf-8")}

merged = {}
for rpid, r in regex_rows.items():
    merged[rpid] = {**r, "source": "regex"}
for rpid, r in embed_rows.items():
    if rpid in merged:
        merged[rpid]["source"] = "both"
        merged[rpid]["embed_score"] = r["embed_score"]
        merged[rpid]["best_anchor_text"] = r["best_anchor_text"]
    else:
        merged[rpid] = {**r, "source": "embed"}

out = OUT / "merged_candidates.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for r in merged.values():
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
src = Counter(r["source"] for r in merged.values())
print(f"Total merged: {len(merged)}")
print(f"By source: {dict(src)}")
total_chars = sum(len(r["text"]) for r in merged.values())
print(f"Total text chars: {total_chars:,} (~{total_chars//4:,} tokens)")
