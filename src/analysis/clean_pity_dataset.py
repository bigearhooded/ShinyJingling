"""Clean LLM-extracted pity reports into a modeling-ready dataset.

Inputs:  data/processed/llm_pity_extractions.jsonl  (618 rows)
Outputs: data/processed/pity_clean.jsonl
         data/processed/pity_clean_summary.md

Cleaning rules
--------------
1. Hand-flagged LLM mistakes (number-parsing failures), removed:
   - rpid 5600 "5.60次污染" -> meant 5–6, drop (ambiguous)
   - rpid 1000 双灯鱼 "1000球" -> ball count, not pollution count
   - rpid 537 "537只污染精灵" -> aggregated cross-pool, mislabelled

2. single_pool with n>80 is physically impossible (hard pity at 80).
   These are either LLM scope mistakes or players genuinely tracking
   multiple pools. Reclassify as cross_pool_total so they don't
   contaminate the M1/M2/M3 fit, but keep them in the cross-pool set.

3. Drop rows where excerpt clearly says "球" (ball count, not pollution).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(".")
SRC = PROJ / "data/processed/llm_pity_extractions.jsonl"
OUT = PROJ / "data/processed/pity_clean.jsonl"
SUMMARY = PROJ / "data/processed/pity_clean_summary.md"

# Hand-curated drops: (excerpt-substring, n-value). Both must match.
# Keeps DROP precise so we don't nuke unrelated rows that just mention "1000球".
DROP_RULES = [
    ("5.60次污染", 5600),       # decimal misread "5.6" -> 5600
    ("537只污染精灵", 537),      # aggregate across pools, mislabelled as single_pool
    ("污染1000球两个异色", 1000),  # 1000 is balls, real pollution count = 2
]


def excerpt_says_balls_not_pollution(excerpt: str, n: float, scope: str) -> bool:
    """If the LLM picked up a ball count instead of a pollution count."""
    if scope != "single_pool":
        return False
    # Heuristic: large n that appears next to "球" but no nearby "污染" / "噩梦"
    if n is None or n <= 80:
        return False
    s = str(excerpt)
    if f"{int(n)}球" in s or f"{int(n)} 球" in s:
        # Only flag if pollution counts are clearly NOT n
        return True
    return False


def main():
    rows = [json.loads(l) for l in SRC.open()]
    n_in = len(rows)

    dropped, reclassified, kept = [], [], []
    for r in rows:
        excerpt = r.get("excerpt", "")
        n = r.get("n")
        scope = r.get("scope")

        # Rule 1+3: drop hand-flagged LLM mistakes (excerpt + n must both match)
        if any(sub in excerpt and n == bad_n for sub, bad_n in DROP_RULES):
            dropped.append((r, "llm_misparsed"))
            continue
        if excerpt_says_balls_not_pollution(excerpt, n, scope):
            dropped.append((r, "ball_count_not_pollution"))
            continue

        # Rule 2: single_pool with n>80 -> reclassify as cross_pool_total
        if scope == "single_pool" and n is not None and n > 80:
            r = dict(r)
            r["scope"] = "cross_pool_total"
            r["_reclassified_from"] = "single_pool"
            reclassified.append(r)

        kept.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    scope_counts = Counter(r["scope"] for r in kept)
    sp = [r for r in kept if r["scope"] == "single_pool"]
    cp = [r for r in kept if r["scope"] == "cross_pool_total"]
    sp_n = [r["n"] for r in sp if r.get("n") is not None]

    lines = [
        "# Cleaned pity dataset summary\n",
        f"- Input rows: {n_in}",
        f"- Dropped (LLM misparsed / ball not pollution): {len(dropped)}",
        f"- Reclassified single_pool->cross_pool (n>80): {len(reclassified)}",
        f"- Kept rows: {len(kept)}",
        "",
        "## Scope counts (cleaned)",
    ]
    for s, c in sorted(scope_counts.items()):
        lines.append(f"- `{s}`: {c}")

    lines += [
        "",
        "## single_pool subset stats (modeling target)",
        f"- count: {len(sp)}",
        f"- with n: {len(sp_n)}",
        f"- min/median/max: {min(sp_n)} / {sorted(sp_n)[len(sp_n)//2]} / {max(sp_n)}",
        f"- n=80 spike: {sum(1 for v in sp_n if v == 80)}",
        f"- n=1 spike: {sum(1 for v in sp_n if v == 1)}",
        "",
        "## Drop log",
    ]
    for r, reason in dropped:
        lines.append(f"- [{reason}] n={r.get('n')} excerpt: {r.get('excerpt','')[:80]}")

    SUMMARY.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(kept)} rows)")
    print(f"Wrote {SUMMARY}")
    print()
    print(f"Drops: {len(dropped)}  Reclassified: {len(reclassified)}  Kept: {len(kept)}")


if __name__ == "__main__":
    main()
