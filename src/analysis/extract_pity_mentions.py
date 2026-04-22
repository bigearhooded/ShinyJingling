#!/usr/bin/env python3
"""
Analyze the existing XHS data (5 search files + 1 details_deep JSONL) to extract
"第 X 次出异色" pity-count mentions.

Output:
- data/processed/pity_mentions.jsonl
- data/processed/pity_mentions_summary.md
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Patterns capture (1) "第 X 次/只" and (2) "X 只/次 ... 出/爆/保底 ... 异色"
# and (3) "打/抓/肝 X 只 ... 才/就 ... 出/爆"
PAT = re.compile(
    r"第\s*(\d{1,3})\s*[只次发条]"
    r"|(\d{1,3})\s*(?:只|次|发|条).{0,25}(?:出|打出|爆|中|保底).{0,15}(?:异色|蛋|精灵|噩梦|梦魇|污染)"
    r"|(?:打|抓|肝|刷)\s*了?\s*(\d{1,3})\s*(?:只|次|发|条).{0,15}(?:才|就|终于|没).{0,15}(?:出|爆|到)",
    re.UNICODE,
)
PITY_KEYWORDS = ("异色", "保底", "噩梦", "梦魇", "污染", "蛋")


def excerpt(text: str, span, radius: int = 40) -> str:
    s = max(0, span[0] - radius)
    e = min(len(text), span[1] + radius)
    return text[s:e].replace("\n", " ").strip()


def is_pity_context(text: str, span, radius: int = 50) -> bool:
    s = max(0, span[0] - radius)
    e = min(len(text), span[1] + radius)
    window = text[s:e]
    return any(k in window for k in PITY_KEYWORDS)


def harvest_strings(node, parent_keys=()):
    """Yield (key_path, str) for every string in a nested JSON structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from harvest_strings(v, parent_keys + (str(k),))
    elif isinstance(node, list):
        for item in node:
            yield from harvest_strings(item, parent_keys)
    elif isinstance(node, str) and node.strip():
        yield ("/".join(parent_keys), node)


def find_mentions(source_id: str, source_type: str, text: str, extra: dict) -> list[dict]:
    out = []
    for m in PAT.finditer(text):
        n = next((int(g) for g in m.groups() if g), None)
        if n is None or not (1 <= n <= 999):
            continue
        if not is_pity_context(text, m.span()):
            continue
        out.append({
            "source_id": source_id,
            "source_type": source_type,
            "n": n,
            "excerpt": excerpt(text, m.span()),
            **extra,
        })
    return out


def main():
    mentions = []

    # 1. Search results (titles only, mostly catalog/showcase)
    for path in sorted(RAW.glob("search_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        for feed in (d.get("results", {}).get("feeds") or []):
            note = feed.get("noteCard") or {}
            title = note.get("displayTitle") or ""
            if not title:
                continue
            mentions += find_mentions(
                source_id=feed.get("id", ""),
                source_type="xhs_search_title",
                text=title,
                extra={"keyword_file": path.name, "title": title, "key_path": "displayTitle"},
            )

    # 2. Deep details (body + comments)
    detail_files = sorted(RAW.glob("details_deep_*.jsonl"))
    n_succ = 0
    n_err = 0
    for path in detail_files:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            raw = row.get("raw_response", {})
            if str(raw.get("result", {}).get("isError")) == "True" or "isError" in str(raw.get("result", {})) and raw["result"].get("isError"):
                n_err += 1
                continue
            n_succ += 1
            fid = row["feed_id"]
            title = row["title"]
            for kpath, s in harvest_strings(raw):
                if not s.strip():
                    continue
                mentions += find_mentions(
                    source_id=fid,
                    source_type="xhs_detail_field",
                    text=s,
                    extra={"title": title, "key_path": kpath, "snippet_chars": len(s)},
                )

    # 3. Dedup
    seen = set()
    deduped = []
    for m in mentions:
        key = (m["source_id"], m["n"], m["excerpt"][:40])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    # 4. Write outputs
    out_jsonl = OUT / "pity_mentions.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for m in deduped:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Summary
    by_type = Counter(m["source_type"] for m in deduped)
    by_n = Counter(m["n"] for m in deduped)
    sm = []
    sm.append("# Pity Mention Smoke Test Summary")
    sm.append("")
    sm.append(f"- XHS detail files: {len(detail_files)}")
    sm.append(f"- Successful detail rows (mined): {n_succ}")
    sm.append(f"- Error detail rows (skipped): {n_err}")
    sm.append(f"- Total mentions extracted: {len(deduped)}")
    sm.append("")
    sm.append("## By source type")
    for t, n in by_type.most_common():
        sm.append(f"- `{t}`: {n}")
    sm.append("")
    sm.append("## Histogram of N (pity-count value)")
    if by_n:
        for n in sorted(by_n):
            bar = "#" * min(by_n[n], 60)
            sm.append(f"`n={n:>3}` {bar} ({by_n[n]})")
        all_ns = sorted(by_n.elements())
        mean = sum(all_ns) / len(all_ns)
        median = sorted(all_ns)[len(all_ns)//2]
        sm.append(f"\n**N stats**: count={len(all_ns)}, mean={mean:.1f}, median={median}, min={min(all_ns)}, max={max(all_ns)}")
    else:
        sm.append("(no mentions matched)")
    sm.append("")
    sm.append("## Sample extracts (top 10 by relevance)")
    for m in sorted(deduped, key=lambda x: (abs(x["n"] - 80), -len(x["excerpt"])))[:10]:
        sm.append(f"- **n={m['n']}** [{m['source_type']}] `{m['source_id']}`")
        sm.append(f"  > {m['excerpt']}")
    (OUT / "pity_mentions_summary.md").write_text("\n".join(sm) + "\n", encoding="utf-8")
    print(f"Wrote {out_jsonl} ({len(deduped)} mentions)")
    print(f"Wrote {OUT / 'pity_mentions_summary.md'}")
    print()
    print("\n".join(sm))


if __name__ == "__main__":
    main()
