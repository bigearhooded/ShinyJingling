#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")
IN_PATH = ROOT / "data" / "processed" / "merged_candidates.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "llm_pity_extractions.jsonl"
SUMMARY_PATH = ROOT / "data" / "processed" / "llm_extraction_summary.md"

VALID_SCOPES = {"single_pool", "cross_pool_total", "qualitative_no_n", "unknown"}
VALID_BALL_OR_POLLUTION = {"pollution", "ball", "unknown"}
VALID_CONFIDENCE = {"high", "med", "low"}
PREVIOUS_RUN_COUNT = 128

PERSONAL_RE = re.compile(r"我|俺|咱|本人|自己|朋友|哥们|室友|同学")
QUESTION_RE = re.compile(r"[?？]|怎么|怎么算|是怎么算|啥意思|请问|问一下|求问|大佬.*看|有没有懂")
GENSHIN_RE = re.compile(r"原神|up角色|限定六星|卡池|吃井|抽到|十连|小保底")
PITY_RE = re.compile(r"异色|保底|出货|没出|不出|刷不出来|出了|才出|污染|噩梦")
SUCCESS_OR_PROGRESS_RE = re.compile(r"异色|出货|没出|不出|刷不出来|还没|没有|保底")
FUZZY_RE = re.compile(r"好几百|几百|上千|几十")
POLLUTION_WORD_RE = re.compile(r"污染|噩梦")
BALL_WORD_RE = re.compile(r"球|捕光|补光|高级|抓了|捉了|捕获|精灵")
MIXED_RE = re.compile(r"混|歪|随机|总共|累计|一共|加起来|前后|换|不同|每只|四只|八只|池")
SHOWOFF_RE = re.compile(r"第一只异色|第1只异色|第一个异色|第一次抓.*异色|开服.*异色")
META_NOISE_RE = re.compile(
    r"赛季作业|倍率|变×|第[一二三四五六七八九十0-9]+章|任务由|改动|官方小报|"
    r"概率为|概率是|机制|规则|理论|攻略|教程|视频|up主|主播|测试|样本|统计|"
    r"任务|爱分享|慈悲为怀|特性|特长|性格"
)

FAMILY_WORDS = [
    "大耳帽兜", "大耳兜帽", "帽兜", "大耳兔", "治愈兔", "兔子", "兔",
    "恶魔狼", "狼王", "狼", "雪影娃娃", "雪影", "燃薪虫", "柴渣虫",
    "火马", "火尾马", "火红尾", "奇丽草", "奇丽叶", "奇丽花",
    "粉星仔", "粉粉星", "雪熊", "月牙雪熊", "格兰种子", "格兰",
    "贝瑟", "绒绒", "酷拉", "拉特", "空空颅", "蚊子", "嗜光蚊子",
    "双灯鱼", "呼呼猪", "猪", "极光千兽", "千兽", "犀角鸟",
    "红绒兔", "红丹鬃", "柴渣", "方方", "酷拉", "火花",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt(text: str, needle: str | None = None) -> str:
    text = clean_text(text)
    if needle and needle in text and len(text) > 80:
        idx = text.index(needle)
        start = max(0, idx - 32)
        return text[start : start + 80]
    return text[:80]


CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_cn_int(token: str) -> int | None:
    token = token.strip()
    token = token.replace("多", "").replace("来", "").replace("+", "")
    token = token.replace("几", "").replace("好", "")
    if not token:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        value = float(token)
        if value < 10 and "." in token:
            return int(value * 1000)
        return int(value)
    if token in CN_DIGITS:
        return CN_DIGITS[token]
    if token == "十":
        return 10
    if "千" in token:
        left, _, right = token.partition("千")
        base = CN_DIGITS.get(left, 1 if left == "" else None)
        if base is None:
            return None
        return base * 1000 + (parse_cn_int(right) or 0)
    if "百" in token:
        left, _, right = token.partition("百")
        base = CN_DIGITS.get(left, 1 if left == "" else None)
        if base is None:
            return None
        return base * 100 + (parse_cn_int(right) or 0)
    if "十" in token:
        left, _, right = token.partition("十")
        tens = CN_DIGITS.get(left, 1 if left == "" else None)
        ones = parse_cn_int(right) if right else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return None


def normalize_n(raw: str, text: str) -> tuple[int | None, str]:
    raw = raw.strip()
    if re.search(r"几百|好几百|上千|几十", raw):
        return None, "low"
    if "一百多" in raw:
        return 100, "low"
    confidence = "low" if re.search(r"多|来|\+|左右|大概|大约|将近|快|差不多", raw + text) else "high"
    return parse_cn_int(raw), confidence


def family_hint(text: str) -> str | None:
    positions = [(text.find(word), word) for word in FAMILY_WORDS if word in text]
    positions = [(pos, word) for pos, word in positions if pos >= 0]
    if not positions:
        return None
    return min(positions)[1]


def is_noise(text: str) -> bool:
    if GENSHIN_RE.search(text):
        return True
    if META_NOISE_RE.search(text) and not re.search(r"我.*(刷了|抓了|捉了|出了|没出|不出|才出|还没)", text):
        return True
    if re.search(r"保底是|保底.*怎么算|算.*保底|平均.*出.*污染|多少.*出.*污染", text):
        return True
    if re.search(r"赛季.*污染[0-9一二两三四五六七八九十百千万]+|污染[0-9一二两三四五六七八九十百千万]+.*任务", text):
        return True
    if QUESTION_RE.search(text) and not re.search(r"我.*(没出|出了|才出|还没|刷了|抓了|捉了)", text):
        return True
    if QUESTION_RE.search(text) and not re.search(r"我.*(才出.*异色|异色.*才出|没出.*异色|不出.*异色|出了.*异色|异色.*出了|还没.*异色)", text):
        return True
    if re.search(r"ID[:：]|加我|来抓|牵手|互刷|欢迎|可抓|蹭金币|蹭洛克贝", text):
        return True
    if re.search(r"卖|价格|氪佬|改性格|PVP|pvp", text) and not re.search(r"我.*(异色.*没出|没出.*异色|才出.*异色|异色.*才出)", text):
        return True
    if re.search(r"概率|机制|规则|理论|怎么算", text) and not PERSONAL_RE.search(text):
        return True
    return False


def personal_enough(text: str) -> bool:
    if PERSONAL_RE.search(text):
        return True
    return bool(re.search(r"刷了|抓了|捉了|出了|才出|没出|不出|还没", text))


def classify_scope(text: str, n: int | None, ball_or_pollution: str) -> str:
    if n is None:
        return "qualitative_no_n"
    if ball_or_pollution == "ball":
        return "cross_pool_total"
    if MIXED_RE.search(text):
        return "cross_pool_total"
    if family_hint(text) or POLLUTION_WORD_RE.search(text):
        return "single_pool"
    return "unknown"


def local_report_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 24) : min(len(text), end + 32)]
    if not re.search(r"我|俺|朋友|本人|刷|抓|捉", window):
        return False
    if re.search(r"异色|出货", window):
        return True
    return bool(re.search(r"没出|不出|才出|出了|还没|保底", window) and re.search(r"异色|出货", text))


def pollution_report_context(text: str, start: int, end: int) -> bool:
    if not local_report_context(text, start, end):
        return False
    window = text[max(0, start - 18) : min(len(text), end + 28)]
    if QUESTION_RE.search(window) and not re.search(r"我.*(刷|抓|捉|出|没)", window):
        return False
    return True


def ball_report_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 20) : min(len(text), end + 36)]
    return bool(
        re.search(r"我|俺|朋友|本人|自己", window)
        and re.search(r"异色|出货", text)
        and re.search(r"才出|没出|不出|出了|还没|出货", window)
        and not re.search(r"爱分享|慈悲|特长|花|金币|洛克贝|粉尘|等级|级|宝可梦", window)
    )


def add_row(
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    source_row: dict[str, Any],
    n: int | None,
    scope: str,
    ball_or_pollution: str,
    confidence: str,
    text: str,
    needle: str | None = None,
) -> None:
    if scope not in VALID_SCOPES or ball_or_pollution not in VALID_BALL_OR_POLLUTION or confidence not in VALID_CONFIDENCE:
        raise ValueError((scope, ball_or_pollution, confidence))
    out = {
        "rpid": int(source_row["rpid"]),
        "n": n,
        "scope": scope,
        "ball_or_pollution": ball_or_pollution,
        "family_hint": family_hint(text),
        "confidence": confidence,
        "excerpt": excerpt(text, needle),
        "source": source_row.get("source", "unknown"),
    }
    key = (out["rpid"], out["n"], out["scope"], out["ball_or_pollution"], out["family_hint"], out["excerpt"])
    compact_key = (out["rpid"], out["n"], out["scope"], out["ball_or_pollution"], out["family_hint"])
    if compact_key in seen:
        return
    if key not in seen:
        rows.append(out)
        seen.add(key)
        seen.add(compact_key)


NUM = r"\d+(?:\.\d+)?|[一二两三四五六七八九十百千万]+"
APPROX = r"(?:多|来|\+|左右|大概|大约|将近|快|差不多)?"
POLLUTION_PATTERNS = [
    re.compile(rf"(?P<num>{NUM}){APPROX}(?:只|个|次|头)?(?:污染|噩梦)"),
    re.compile(rf"第(?P<num>{NUM}){APPROX}(?:只|个|次)?(?:污染|噩梦)"),
    re.compile(rf"(?:污染|噩梦)(?:精灵)?(?P<num>{NUM}){APPROX}(?:只|个|次|头)?"),
]
BALL_PATTERNS = [
    re.compile(rf"(?P<num>{NUM}){APPROX}(?:个|颗|只)?(?:球|捕光|补光|高级)"),
    re.compile(rf"(?:抓了|捉了|捕获)(?P<num>{NUM}){APPROX}(?:只|个|头)?(?:精灵|宠物|[一-龥]{{1,6}})?(?:才出|没出|不出|出了|出异色)"),
]


def heuristic_extract(source_row: dict[str, Any], rows: list[dict[str, Any]], seen: set[tuple[Any, ...]]) -> None:
    text = clean_text(source_row.get("text", ""))
    if not text or is_noise(text):
        return
    if not (PITY_RE.search(text) and SUCCESS_OR_PROGRESS_RE.search(text) and personal_enough(text)):
        return
    if SHOWOFF_RE.search(text) and not POLLUTION_WORD_RE.search(text):
        return

    matched = False
    for pattern in POLLUTION_PATTERNS:
        for match in pattern.finditer(text):
            if not pollution_report_context(text, match.start(), match.end()):
                continue
            raw = match.group("num")
            n, conf = normalize_n(raw, text)
            if n is None or n <= 0:
                continue
            if n > 500 and "污染" not in match.group(0) and "噩梦" not in match.group(0):
                continue
            scope = classify_scope(text, n, "pollution")
            add_row(rows, seen, source_row, n, scope, "pollution", conf, text, match.group(0))
            matched = True

    if matched:
        return

    for pattern in BALL_PATTERNS:
        for match in pattern.finditer(text):
            if not ball_report_context(text, match.start(), match.end()):
                continue
            raw = match.group("num")
            n, conf = normalize_n(raw, text)
            if n is None or n <= 0:
                continue
            if n < 500:
                continue
            add_row(rows, seen, source_row, n, "cross_pool_total", "ball", "low" if conf == "high" else conf, text, match.group(0))
            matched = True

    if matched:
        return

    fuzzy_match = FUZZY_RE.search(text)
    if fuzzy_match and local_report_context(text, fuzzy_match.start(), fuzzy_match.end()):
        ball_or_pollution = "pollution" if POLLUTION_WORD_RE.search(text) else "ball" if BALL_WORD_RE.search(text) else "unknown"
        add_row(rows, seen, source_row, None, "qualitative_no_n", ball_or_pollution, "low", text, fuzzy_match.group(0))
        return

    if source_row.get("source") == "embed" and PERSONAL_RE.search(text) and re.search(r"吃保底|刷不出来|没刷出来|还没.*异色|刷了.*没出|刷了.*才出", text):
        ball_or_pollution = "pollution" if POLLUTION_WORD_RE.search(text) else "unknown"
        add_row(rows, seen, source_row, None, "qualitative_no_n", ball_or_pollution, "low", text, None)


def hist_lines(values: list[int]) -> list[str]:
    if not values:
        return ["- (none)"]
    counts = Counter(values)
    return [f"- `{n}`: {counts[n]}" for n in sorted(counts)]


def source_lines(counter: Counter[str]) -> list[str]:
    keys = ["regex", "embed", "both", "unknown"]
    lines = [f"- `{key}`: {counter.get(key, 0)}" for key in keys if counter.get(key, 0)]
    return lines or ["- (none)"]


def main() -> None:
    candidates = load_jsonl(IN_PATH)
    by_rpid = {int(row["rpid"]): row for row in candidates}
    previous = load_jsonl(OUT_PATH)
    seed_previous = len(previous) == PREVIOUS_RUN_COUNT
    previous_rpids = {int(row["rpid"]) for row in previous} if seed_previous else set()

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for row in previous if seed_previous else []:
        source_row = by_rpid.get(int(row["rpid"]))
        if source_row is None:
            continue
        out = {
            "rpid": int(row["rpid"]),
            "n": row.get("n"),
            "scope": row.get("scope", "unknown"),
            "ball_or_pollution": row.get("ball_or_pollution", "unknown"),
            "family_hint": row.get("family_hint"),
            "confidence": row.get("confidence", "low"),
            "excerpt": clean_text(row.get("excerpt", ""))[:80],
            "source": source_row.get("source", "unknown"),
        }
        key = (out["rpid"], out["n"], out["scope"], out["ball_or_pollution"], out["family_hint"], out["excerpt"])
        rows.append(out)
        seen.add(key)

    for candidate in candidates:
        heuristic_extract(candidate, rows, seen)

    rows.sort(key=lambda row: (row["rpid"], -1 if row["n"] is None else row["n"], row["scope"], row["excerpt"]))
    write_jsonl(OUT_PATH, rows)

    input_sources = Counter(row.get("source", "unknown") for row in candidates)
    kept_sources = Counter(row["source"] for row in rows)
    new_rows = [row for row in rows if row["rpid"] not in previous_rpids]
    new_sources = Counter(row["source"] for row in new_rows)
    rejected_sources = input_sources.copy()
    for row in rows:
        rejected_sources[row["source"]] -= 1

    by_scope = Counter(row["scope"] for row in rows)
    single_ns = [row["n"] for row in rows if row["scope"] == "single_pool" and isinstance(row["n"], int)]
    cross_ns = [row["n"] for row in rows if row["scope"] == "cross_pool_total" and isinstance(row["n"], int)]
    qualitative_count = by_scope.get("qualitative_no_n", 0)

    top_single = []
    for row in rows:
        if row["scope"] != "single_pool":
            continue
        source_row = by_rpid.get(row["rpid"], {})
        top_single.append((int(source_row.get("like", 0) or 0), row))
    top_single.sort(key=lambda item: (-item[0], item[1]["rpid"], -1 if item[1]["n"] is None else item[1]["n"]))

    previous_count = PREVIOUS_RUN_COUNT
    new_rpids = {row["rpid"] for row in new_rows}
    embed_new = [row for row in new_rows if row["source"] == "embed"]

    lines = [
        "# LLM Pity Extraction Summary",
        "",
        "## Input and source breakdown",
        f"- Total candidates input: {len(candidates)}",
        "- Input source counts:",
        *source_lines(input_sources),
        "- Kept source counts:",
        *source_lines(kept_sources),
        "- New kept source counts vs previous run:",
        *source_lines(new_sources),
        "- Rejected/skipped source counts:",
        *source_lines(+rejected_sources),
        "",
        "## Kept reports by scope",
        f"- Total kept: {len(rows)}",
        f"- `single_pool`: {by_scope.get('single_pool', 0)}",
        f"- `cross_pool_total`: {by_scope.get('cross_pool_total', 0)}",
        f"- `qualitative_no_n`: {qualitative_count}",
        f"- `unknown`: {by_scope.get('unknown', 0)}",
        "",
        "## Histogram: single_pool n",
        *hist_lines(single_ns),
        "",
        "## Histogram: cross_pool_total n",
        *hist_lines(cross_ns),
        "",
        "## qualitative_no_n",
        f"- Count: {qualitative_count}",
        "",
        "## Top 10 single_pool excerpts by like",
    ]

    for like, row in top_single[:10]:
        lines.append(
            f"- like={like} n={row['n']} family={row['family_hint']} source={row['source']}: {row['excerpt']}"
        )
    if not top_single:
        lines.append("- (none)")

    lines += [
        "",
        "## Comparison vs previous run",
        f"- Previous run reports: {previous_count}",
        f"- Current reports: {len(rows)}",
        f"- Net change: {len(rows) - previous_count:+d}",
    ]
    if seed_previous:
        lines += [
            f"- New report rows: {len(new_rows)} across {len(new_rpids)} rpids",
            f"- New embed-only report rows: {len(embed_new)}",
        ]
    else:
        lines += [
            "- Exact previous-row overlap could not be recomputed because the 128-row baseline file was already overwritten before this final pass.",
            f"- Current embed-only kept rows: {kept_sources.get('embed', 0)}",
        ]
    lines += [
    ]
    embed_examples = embed_new if seed_previous else [row for row in rows if row["source"] == "embed"]
    if embed_examples:
        lines.append("- Example embedding-only catches:")
        for row in embed_examples[:10]:
            lines.append(f"  - rpid={row['rpid']} n={row['n']} scope={row['scope']}: {row['excerpt']}")

    lines += [
        "",
        "## Notes",
        "- Previous run baseline is 128 reports. If that exact file is present before execution, those adjudicated rows are retained and annotated with merged candidate `source`; otherwise the current file is rebuilt from the merged candidates with the stricter recall rules.",
        "- New recall is conservative: concrete `污染/噩梦` counts are preferred; ball/catch totals are kept as `cross_pool_total`; vague counts such as `几百`, `好几百`, `几十`, and `上千` are `qualitative_no_n` with `n=null`.",
        "- Mechanism Q&A, recruitment/ID posts, Genshin-style gacha language, and first-shiny showoff without pollution context were skipped.",
    ]

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
