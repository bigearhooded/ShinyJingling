#!/usr/bin/env python3
"""
Re-classify the 131 raw bili pity-mention extracts into:
  - personal_pity   : 真实"我第 X 次/只 出异色" 个人保底报告 (the gold)
  - mechanic_qna    : 关于"80只保底"机制的问答 / 规则陈述
  - ball_count      : "抓了 X 只" 但语义是球数/捕捉总数, 不是污染数
  - genshin_noise   : 原神/其他游戏 120 抽干扰
  - other           : 不能确定的杂项

Output:
  data/processed/pity_mentions_classified.jsonl
  data/processed/pity_mentions_classified_summary.md
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Markers
PERSONAL = re.compile(r"我|俺|咱|本人|本宝|哥们|姐们|楼主|up主自己")  # personal subject
PERSONAL_VERB = re.compile(r"(刷|抓|肝|刚才|今天|昨天|刚刚|花了|用了|抽了|破了|歪了|爆了|出货|出了|终于|才出|出的|抓到)")
MECHANIC = re.compile(
    r"保底.{0,8}(是|为|分|怎么|多少|算|计|池|机制|规则)"
    r"|(是|应该|至少|起码|算|官方说|官方|公告).{0,10}保底"
    r"|必.{0,3}出.{0,5}异色"
    r"|是不是.{0,15}保底"
    r"|怎么算.{0,5}的"
    r"|是同一种|每个赛季|每个种族|每个家族|累计.{0,10}保底|有没有.{0,5}保底"
    r"|累计|官方白纸"
)
QUESTION = re.compile(r"[?？]|多少|怎么|是不是|那如果|假如|意思是|是吗|呢$|有没有")
GENSHIN = re.compile(r"原神|up角色|卡池|限定六星|海角|绫华|杰哥|伊冯|刻晴|纳西妲")
BALL_COUNT_HINT = re.compile(r"球|拾光|捕光|高级|棱镜|球了|球出|球抓")
POLLUTION_HINT = re.compile(r"污染|噩梦|梦魇")
SHINY_HINT = re.compile(r"异色|炫彩|棱镜异")
NUM_PATTERN = re.compile(r"\d+")


def is_personal_subject(text: str, span: tuple) -> bool:
    s = max(0, span[0] - 30)
    e = min(len(text), span[1] + 20)
    win = text[s:e]
    return bool(PERSONAL.search(win) and PERSONAL_VERB.search(win))


def is_mechanic_qna(text: str) -> bool:
    return bool(MECHANIC.search(text)) or bool(QUESTION.search(text) and "保底" in text)


def is_genshin(text: str) -> bool:
    return bool(GENSHIN.search(text))


def has_ball_lexeme_near(text: str, span: tuple, radius: int = 30) -> bool:
    s = max(0, span[0] - radius)
    e = min(len(text), span[1] + radius)
    return bool(BALL_COUNT_HINT.search(text[s:e]))


def has_pollution_lexeme_near(text: str, span: tuple, radius: int = 30) -> bool:
    s = max(0, span[0] - radius)
    e = min(len(text), span[1] + radius)
    return bool(POLLUTION_HINT.search(text[s:e]))


def classify(row: dict) -> tuple[str, dict]:
    text = row.get("full_text") or row.get("text") or row.get("excerpt", "")
    n = row["n"]
    # Need to find span in full_text. If full_text not present, fall back to whole excerpt
    span_text = text
    # use the same regex used for extraction to find the matched n
    # Robust: search for "第 N" or "N 只/次"
    span = None
    for m in re.finditer(rf"\b{n}\b|第\s*{n}\s*[只次发条]", span_text):
        span = m.span()
        break
    if span is None:
        span = (0, len(span_text))

    flags = {
        "personal": is_personal_subject(span_text, span),
        "mechanic": is_mechanic_qna(span_text),
        "genshin": is_genshin(span_text),
        "near_ball": has_ball_lexeme_near(span_text, span),
        "near_pollution": has_pollution_lexeme_near(span_text, span),
    }

    is_question = bool(QUESTION.search(span_text))
    flags["question"] = is_question

    if flags["genshin"]:
        return "genshin_noise", flags
    # Question about pity rule — even if "personal subject" present
    if is_question and ("保底" in span_text or "异色" in span_text and "?" in span_text or "？" in span_text):
        if flags["mechanic"] or any(w in span_text for w in ("是不是", "怎么算", "怎么计", "意思是", "如果")):
            return "mechanic_qna", flags
    if flags["mechanic"] and not flags["personal"]:
        return "mechanic_qna", flags
    if flags["personal"] and flags["near_pollution"]:
        return "personal_pity", flags
    if flags["personal"] and flags["near_ball"] and not flags["near_pollution"]:
        return "ball_count", flags
    if flags["personal"]:
        return "personal_pity", flags
    if flags["mechanic"]:
        return "mechanic_qna", flags
    return "other", flags


def main():
    candidates = sorted(RAW.glob("bili_mentions_v2_*.jsonl")) or sorted(RAW.glob("bili_mentions_*.jsonl"))
    in_file = candidates[-1]
    rows = [json.loads(l) for l in in_file.open(encoding="utf-8")]

    by_class: dict[str, list[dict]] = defaultdict(list)
    enriched = []
    for r in rows:
        cls, flags = classify(r)
        r2 = {**r, "class": cls, "flags": flags}
        by_class[cls].append(r2)
        enriched.append(r2)

    out_file = OUT / "pity_mentions_classified.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for r in enriched:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    sm = []
    sm.append("# Pity Mentions Classified")
    sm.append(f"\nTotal: {len(rows)} mentions from {in_file.name}\n")
    sm.append("## Class breakdown")
    for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
        sm.append(f"- **{cls}**: {len(items)}")
    sm.append("\n## Personal pity reports (the gold)\n")
    personal = [r for r in enriched if r["class"] == "personal_pity"]
    personal.sort(key=lambda r: r["n"])
    sm.append(f"Count: {len(personal)}")
    if personal:
        ns = [r["n"] for r in personal]
        sm.append(f"N stats: mean={sum(ns)/len(ns):.1f}, median={sorted(ns)[len(ns)//2]}, min={min(ns)}, max={max(ns)}")
        sm.append("\n### Histogram (personal_pity only)")
        h = Counter(ns)
        for n in sorted(h):
            sm.append(f"`n={n:>4}` {'#'*min(h[n],50)} ({h[n]})")
        sm.append("\n### Sample personal reports (sorted by N)")
        for r in personal:
            sm.append(f"- **n={r['n']:>4}** like={r.get('like',0)}: {r['text'][:140]}")
    sm.append("\n## Mechanic Q&A samples (top 8)")
    for r in by_class["mechanic_qna"][:8]:
        sm.append(f"- n={r['n']}: {r['text'][:140]}")
    sm.append("\n## Ball-count samples (top 8)")
    for r in by_class["ball_count"][:8]:
        sm.append(f"- n={r['n']}: {r['text'][:140]}")
    sm.append("\n## Genshin noise samples")
    for r in by_class["genshin_noise"][:5]:
        sm.append(f"- n={r['n']}: {r['text'][:140]}")
    sm_path = OUT / "pity_mentions_classified_summary.md"
    sm_path.write_text("\n".join(sm) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")
    print(f"Wrote {sm_path}")
    print()
    print("\n".join(sm[:80]))


if __name__ == "__main__":
    main()
