#!/usr/bin/env python3
"""
B站 数据采集 for 26_luoke 项目.

1. 搜索若干关键词, 收集视频列表 (bvid, 标题, 播放量, 评论数, 点赞数, 作者)
2. 按评论数降序挑 top N 视频
3. 拉每个视频的全部评论 + 二级回复
4. 增量写盘 (data/raw/bili_comments_<bvid>.json)
5. 跑正则抽取 "第 X 次出异色" 类数据
6. 输出 data/raw/bili_summary.md

Usage: python3 bili_fetch.py [--top 30] [--max_pages 50]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from bilibili_api import search, comment, video, sync, Credential
from bilibili_api.comment import CommentResourceType, OrderType

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def load_credential() -> Credential | None:
    env_path = ROOT / "bin" / "bili_cookie.env"
    if not env_path.exists():
        return None
    kv = {}
    for ln in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()
    return Credential(
        sessdata=kv.get("SESSDATA", ""),
        bili_jct=kv.get("BILI_JCT", ""),
        dedeuserid=kv.get("DEDEUSERID", ""),
        buvid3=kv.get("BUVID3", ""),
    )


CRED = load_credential()

KEYWORDS = [
    # round 1 (already collected, kept for completeness)
    "洛克王国 异色 保底",
    "洛克王国 异色 80次",
    "洛克王国世界 异色",
    "洛克王国 噩梦 异色",
    "洛克王国 异色 概率",
    # round 2 - 口语化, targeting comment-rich personal-experience posts
    "洛克王国 异色 出货",
    "洛克王国 异色 非酋",
    "洛克王国 异色 多少次",
    "洛克王国 异色 心得",
    "洛克王国 噩梦 80",
]

# 同时匹配 梦魇 / 噩梦 / 污染 + "第 X 次出"
PAT = re.compile(
    r"第\s*(\d{1,4})\s*[只次发条]"
    r"|(\d{1,4})\s*(?:只|次|发|条).{0,25}(?:出|打出|爆|中|保底).{0,15}(?:异色|蛋|精灵|噩梦|污染|梦魇)"
    r"|打\s*了\s*(\d{1,4})\s*(?:只|次|发|条).{0,30}(?:才|就|终于).{0,15}(?:出|爆)"
    r"|(?:刷|抓|肝)\s*了?\s*(\d{1,4})\s*(?:只|次|发|条).{0,15}(?:才|就|终于)",
    re.UNICODE,
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def excerpt(text: str, span, radius: int = 36) -> str:
    s = max(0, span[0] - radius)
    e = min(len(text), span[1] + radius)
    return text[s:e].replace("\n", " ").strip()


def search_videos(keyword: str, pages: int = 2) -> list[dict]:
    out = []
    for p in range(1, pages + 1):
        try:
            res = sync(search.search_by_type(
                keyword,
                search.SearchObjectType.VIDEO,
                order_type=search.OrderVideo.TOTALRANK,
                page=p,
            ))
        except Exception as exc:
            log(f"  search page {p} failed: {exc}")
            break
        results = res.get("result") or []
        if not results:
            break
        for v in results:
            out.append({
                "bvid": v.get("bvid"),
                "aid": v.get("aid"),
                "title": re.sub(r"<[^>]+>", "", v.get("title", "")),
                "author": v.get("author"),
                "play": v.get("play"),
                "review": v.get("review"),  # 弹幕数
                "video_review": v.get("video_review"),  # 评论数
                "favorites": v.get("favorites"),
                "like": v.get("like"),
                "pubdate": v.get("pubdate"),
            })
        time.sleep(2)
    return out


async def fetch_comments(aid: int, max_pages: int = 50, sub_per_root: int = 50) -> list[dict]:
    """Fetch L1 comments + L2 replies for a video using lazy (cursor) API."""
    out = []
    cursor = ""
    page = 0
    while page < max_pages:
        page += 1
        try:
            res = await comment.get_comments_lazy(
                oid=aid,
                type_=CommentResourceType.VIDEO,
                offset=cursor,
                order=OrderType.TIME,
                credential=CRED,
            )
        except Exception as exc:
            log(f"    L1 page {page} failed: {exc}")
            break
        replies = res.get("replies") or []
        if not replies:
            break
        cursor_obj = res.get("cursor") or {}
        # next pagination offset for lazy api
        next_pag = res.get("cursor", {}).get("pagination_reply", {}).get("next_offset") or ""
        if not next_pag:
            next_pag = res.get("cursor", {}).get("session_id") or ""
        for r in replies:
            row = {
                "rpid": r.get("rpid"),
                "uid": r.get("member", {}).get("mid"),
                "uname": r.get("member", {}).get("uname"),
                "ctime": r.get("ctime"),
                "like": r.get("like"),
                "content": r.get("content", {}).get("message", ""),
                "rcount": r.get("rcount", 0),
                "sub_replies": [],
            }
            if row["rcount"] > 0:
                try:
                    cobj = comment.Comment(
                        oid=aid,
                        type_=CommentResourceType.VIDEO,
                        rpid=r["rpid"],
                        credential=CRED,
                    )
                    sub_page = 1
                    while sub_page <= 5:
                        sub = await cobj.get_sub_comments(page_index=sub_page)
                        srs = sub.get("replies") or []
                        if not srs:
                            break
                        for sr in srs:
                            row["sub_replies"].append({
                                "rpid": sr.get("rpid"),
                                "uid": sr.get("member", {}).get("mid"),
                                "uname": sr.get("member", {}).get("uname"),
                                "ctime": sr.get("ctime"),
                                "like": sr.get("like"),
                                "content": sr.get("content", {}).get("message", ""),
                            })
                        if len(row["sub_replies"]) >= sub_per_root:
                            break
                        sub_page += 1
                        await asyncio.sleep(0.4)
                except Exception as exc:
                    log(f"    L2 for rpid={r['rpid']} failed: {exc}")
            out.append(row)
        await asyncio.sleep(1.2)
        if cursor_obj.get("is_end"):
            break
        cursor = next_pag
        if not cursor:
            break
        if len(out) >= 1500:
            log(f"    capped at {len(out)} comments")
            break
    return out


def extract_mentions(video_meta: dict, comments: list[dict]) -> list[dict]:
    mentions = []
    for c in comments:
        for src in [(None, c)] + [(c["rpid"], sr) for sr in c.get("sub_replies", [])]:
            parent_rpid, item = src
            text = item["content"]
            for m in PAT.finditer(text):
                n = next((int(g) for g in m.groups() if g), None)
                if n is None or not (1 <= n <= 999):
                    continue
                mentions.append({
                    "bvid": video_meta["bvid"],
                    "video_title": video_meta["title"],
                    "rpid": item["rpid"],
                    "parent_rpid": parent_rpid,
                    "uid": item["uid"],
                    "uname": item["uname"],
                    "like": item["like"],
                    "n": n,
                    "text": excerpt(text, m.span()),
                    "full_text": text,
                })
    return mentions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="how many top videos to fetch comments for")
    ap.add_argument("--pages", type=int, default=2, help="search pages per keyword")
    ap.add_argument("--max_pages", type=int, default=50, help="max comment pages per video")
    args = ap.parse_args()

    ts = int(time.time())

    # Step 1: searches
    all_videos = []
    seen = set()
    for kw in KEYWORDS:
        log(f"Search: {kw}")
        vids = search_videos(kw, pages=args.pages)
        log(f"  got {len(vids)} videos")
        for v in vids:
            if v["bvid"] and v["bvid"] not in seen:
                seen.add(v["bvid"])
                v["source_keyword"] = kw
                all_videos.append(v)

    search_path = RAW / f"bili_search_{ts}.json"
    search_path.write_text(json.dumps(all_videos, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {search_path} ({len(all_videos)} unique videos)")

    # Step 2: rank
    ranked = sorted(all_videos, key=lambda x: (x.get("video_review") or 0), reverse=True)
    selected = [v for v in ranked if (v.get("video_review") or 0) >= 30][:args.top]
    log(f"Selected top {len(selected)} videos with >=30 comments")
    for v in selected[:10]:
        log(f"  bv={v['bvid']}  views={v['play']:>6}  comments={v.get('video_review','?'):>5}  | {v['title'][:50]}")

    # Step 2.5: dedup against already-fetched videos
    existing_bvids = {p.name.split("_")[2] for p in RAW.glob("bili_video_*.json")}
    if existing_bvids:
        before = len(selected)
        selected = [v for v in selected if v["bvid"] not in existing_bvids]
        log(f"Skipped {before - len(selected)} already-fetched bvids; {len(selected)} remain")

    # Step 3: fetch comments per video, incremental
    all_comments_path = RAW / f"bili_comments_{ts}.jsonl"
    all_mentions_path = RAW / f"bili_mentions_{ts}.jsonl"
    total_comments = 0
    total_mentions = 0
    fout_c = all_comments_path.open("w", encoding="utf-8")
    fout_m = all_mentions_path.open("w", encoding="utf-8")

    for idx, v in enumerate(selected, 1):
        log(f"[{idx}/{len(selected)}] fetch comments aid={v['aid']} bv={v['bvid']}")
        try:
            comments = sync(fetch_comments(v["aid"], max_pages=args.max_pages))
        except Exception as exc:
            log(f"    failed: {exc}")
            continue
        n_l1 = len(comments)
        n_l2 = sum(len(c.get("sub_replies", [])) for c in comments)
        total_comments += n_l1 + n_l2
        log(f"    pulled L1={n_l1} L2={n_l2}")

        # write per-video JSON for full fidelity
        per_path = RAW / f"bili_video_{v['bvid']}_{ts}.json"
        per_path.write_text(json.dumps({
            "video_meta": v,
            "comments": comments,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False), encoding="utf-8")

        # accumulate JSONL (lighter)
        for c in comments:
            fout_c.write(json.dumps({"bvid": v["bvid"], **c}, ensure_ascii=False) + "\n")

        mentions = extract_mentions(v, comments)
        total_mentions += len(mentions)
        for m in mentions:
            fout_m.write(json.dumps(m, ensure_ascii=False) + "\n")
        log(f"    wrote {per_path.name}, mentions={len(mentions)}, running total mentions={total_mentions}")

    fout_c.close()
    fout_m.close()

    # Step 4: summary
    log(f"Done. comments_total={total_comments} mentions_total={total_mentions}")

    # mentions histogram
    ns = []
    if all_mentions_path.exists():
        for line in all_mentions_path.open(encoding="utf-8"):
            try:
                ns.append(json.loads(line)["n"])
            except Exception:
                pass

    sm = []
    sm.append(f"# B站 SMOKE TEST ({datetime.now().isoformat()})")
    sm.append("")
    sm.append(f"- Searched keywords: {len(KEYWORDS)}")
    sm.append(f"- Unique videos discovered: {len(all_videos)}")
    sm.append(f"- Videos with >=30 comments: {len(selected)}")
    sm.append(f"- Total comments fetched (L1+L2): {total_comments}")
    sm.append(f"- Pity-count mentions extracted: {total_mentions}")
    sm.append("")
    sm.append("## N value histogram")
    if ns:
        hist = Counter(ns)
        for n in sorted(hist):
            sm.append(f"`n={n:>3}` | {'#' * min(hist[n], 60)} ({hist[n]})")
        sm.append(f"\nMean: {sum(ns)/len(ns):.1f}, Median: {sorted(ns)[len(ns)//2]}, Max: {max(ns)}")
    else:
        sm.append("(no mentions matched)")
    (RAW / "bili_summary.md").write_text("\n".join(sm) + "\n", encoding="utf-8")
    log(f"Wrote bili_summary.md")


if __name__ == "__main__":
    main()
