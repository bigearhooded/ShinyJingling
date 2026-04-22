#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None
    import urllib.error
    import urllib.request


ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw"
SCRIPT_PATH = ROOT / "src" / "scrapers" / "xhs_deep_fetch.py"
MCP_URL = "http://localhost:18060/mcp"
PAT = re.compile(
    r"第\s*(\d{1,3})\s*[只次条发]|(\d{1,3})\s*(?:只|次|条|发).{0,20}(?:出|打出|爆|中|保底).{0,20}异色",
    re.UNICODE,
)
COMMENT_TEXT_KEYS = (
    "content",
    "text",
    "comment",
    "commentContent",
    "commentText",
    "message",
    "desc",
    "description",
)
COMMENT_ID_KEYS = ("comment_id", "commentId", "id")
USER_ID_KEYS = ("user_id", "userId")


@dataclass
class FeedRecord:
    keyword: str
    feed_id: str
    xsec_token: str
    title: str
    author: str
    comment_count: int | None
    liked_count: int | None
    source_file: str


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def parse_count(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip().lower()
    if not text:
        return None
    text = text.replace(",", "")
    mult = 1
    if text.endswith("万"):
        mult = 10000
        text = text[:-1]
    elif text.endswith("k"):
        mult = 1000
        text = text[:-1]
    try:
        return int(float(text) * mult)
    except ValueError:
        digits = re.sub(r"[^\d.]", "", text)
        if not digits:
            return None
        try:
            return int(float(digits) * mult)
        except ValueError:
            return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_search_feeds() -> list[FeedRecord]:
    records: list[FeedRecord] = []
    for path in sorted(RAW_DIR.glob("search_*.json")):
        payload = read_json(path)
        keyword = str(payload.get("keyword", ""))
        feeds = (((payload.get("results") or {}).get("feeds")) or [])
        for feed in feeds:
            if not isinstance(feed, dict) or feed.get("modelType") != "note":
                continue
            note = feed.get("noteCard") or {}
            user = note.get("user") or {}
            interact = note.get("interactInfo") or {}
            records.append(
                FeedRecord(
                    keyword=keyword,
                    feed_id=str(feed.get("id", "")),
                    xsec_token=str(feed.get("xsecToken", "")),
                    title=str(note.get("displayTitle") or ""),
                    author=str(user.get("nickname") or user.get("nickName") or ""),
                    comment_count=parse_count(interact.get("commentCount")),
                    liked_count=parse_count(interact.get("likedCount")),
                    source_file=path.name,
                )
            )
    return records


def sse_or_json_loads(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)

    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        raise ValueError("Unable to parse MCP response as JSON or SSE")

    decoded: list[Any] = []
    for item in data_lines:
        if item == "[DONE]":
            continue
        decoded.append(json.loads(item))
    if len(decoded) == 1:
        return decoded[0]
    return decoded


class MCPClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.session_id: str | None = None
        self.rpc_id = 1

    def _next_id(self) -> int:
        self.rpc_id += 1
        return self.rpc_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _post(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], str]:
        if requests is not None:
            resp = requests.post(self.url, headers=self._headers(), json=payload, timeout=1800)
            return resp.status_code, dict(resp.headers), resp.text

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, dict(resp.headers), body
        except urllib.error.HTTPError as exc:  # pragma: no cover
            body = exc.read().decode("utf-8", errors="replace")
            return exc.code, dict(exc.headers), body

    def initialize(self) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "codex", "version": "0.1"},
            },
        }
        status, headers, text = self._post(payload)
        if status >= 400:
            raise RuntimeError(f"initialize failed with HTTP {status}: {text[:500]}")
        self.session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError("initialize succeeded without Mcp-Session-Id header")
        return sse_or_json_loads(text)

    def notify_initialized(self) -> None:
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        status, _, text = self._post(payload)
        if status >= 400:
            raise RuntimeError(f"notifications/initialized failed with HTTP {status}: {text[:500]}")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        status, _, text = self._post(payload)
        if status >= 400:
            raise RuntimeError(f"tools/call {name} failed with HTTP {status}: {text[:500]}")
        return sse_or_json_loads(text)


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)


def find_first_string(value: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            if key in value and isinstance(value[key], (str, int)):
                return str(value[key])
        for nested in value.values():
            found = find_first_string(nested, keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_first_string(item, keys)
            if found:
                return found
    return None


def likely_comment_node(node: dict[str, Any]) -> bool:
    has_text = any(isinstance(node.get(k), str) and str(node.get(k)).strip() for k in COMMENT_TEXT_KEYS)
    has_user = any(k in node for k in USER_ID_KEYS) or isinstance(node.get("user"), dict)
    has_id = any(k in node for k in COMMENT_ID_KEYS)
    return has_text and (has_user or has_id)


def extract_comment_records(value: Any) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def rec(node: Any) -> None:
        if isinstance(node, dict):
            if likely_comment_node(node):
                text = None
                for key in COMMENT_TEXT_KEYS:
                    if isinstance(node.get(key), str) and node.get(key).strip():
                        text = node[key].strip()
                        break
                if text:
                    comment_id = find_first_string(node, COMMENT_ID_KEYS) or ""
                    user_id = find_first_string(node, USER_ID_KEYS) or ""
                    key = (comment_id, text)
                    if key not in seen:
                        seen.add(key)
                        comments.append(
                            {
                                "comment_id": comment_id,
                                "user_id": user_id,
                                "text": text,
                                "node": node,
                            }
                        )
            for child in node.values():
                rec(child)
        elif isinstance(node, list):
            for child in node:
                rec(child)

    rec(value)
    return comments


def extract_body_text(raw_response: Any) -> str:
    candidates: list[str] = []

    def rec(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                next_path = f"{path}.{key}" if path else key
                lowered = key.lower()
                if isinstance(val, str) and any(
                    token in lowered for token in ("desc", "content", "text", "title")
                ):
                    candidates.append(val)
                rec(val, next_path)
        elif isinstance(node, list):
            for item in node:
                rec(item, path)

    rec(raw_response)
    non_empty = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    non_empty.sort(key=len, reverse=True)
    return non_empty[0] if non_empty else ""


def excerpt(text: str, span: tuple[int, int], radius: int = 36) -> str:
    start = max(0, span[0] - radius)
    end = min(len(text), span[1] + radius)
    return text[start:end].replace("\n", " ").strip()


def extract_mentions(
    selected_posts: list[FeedRecord], deep_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mentions: list[dict[str, Any]] = []
    comments_by_feed: dict[str, int] = {}
    by_feed = {row["feed_id"]: row for row in deep_rows}

    for post in selected_posts:
        row = by_feed.get(post.feed_id)
        if not row:
            comments_by_feed[post.feed_id] = 0
            continue
        raw = row["raw_response"]
        body = extract_body_text(raw)
        comments = extract_comment_records(raw)
        comments_by_feed[post.feed_id] = len(comments)

        for match in PAT.finditer(body):
            n = int(match.group(1) or match.group(2))
            mentions.append(
                {
                    "feed_id": post.feed_id,
                    "keyword": post.keyword,
                    "source": "body",
                    "commenter_id": None,
                    "text_excerpt": excerpt(body, match.span()),
                    "n": n,
                    "is_pity_mention": True,
                }
            )

        for comment in comments:
            text = comment["text"]
            for match in PAT.finditer(text):
                n = int(match.group(1) or match.group(2))
                mentions.append(
                    {
                        "feed_id": post.feed_id,
                        "keyword": post.keyword,
                        "source": "comment",
                        "commenter_id": comment["user_id"] or None,
                        "text_excerpt": excerpt(text, match.span()),
                        "n": n,
                        "is_pity_mention": True,
                    }
                )

    return mentions, comments_by_feed


def build_smoke_test(
    records: list[FeedRecord],
    selected_posts: list[FeedRecord],
    comments_by_feed: dict[str, int],
    mentions: list[dict[str, Any]],
) -> str:
    selected_ids = {post.feed_id for post in selected_posts}
    post_counts = Counter(post.keyword for post in selected_posts)
    comment_counts = Counter()
    for post in selected_posts:
        comment_counts[post.keyword] += comments_by_feed.get(post.feed_id, 0)
    hit_counts = Counter(m["keyword"] for m in mentions)

    lines: list[str] = []
    lines.append("# SMOKE TEST")
    lines.append("")
    lines.append("| keyword | # posts collected | # comments pulled total | # pity-count hits |")
    lines.append("|---|---:|---:|---:|")
    all_keywords = sorted({r.keyword for r in records})
    for keyword in all_keywords:
        lines.append(
            f"| {keyword} | {post_counts.get(keyword, 0)} | {comment_counts.get(keyword, 0)} | {hit_counts.get(keyword, 0)} |"
        )

    lines.append("")
    lines.append("## Histogram")
    values = sorted(m["n"] for m in mentions)
    if values:
        hist = Counter(values)
        for n in sorted(hist):
            lines.append(f"`n={n:>3}` | {'#' * hist[n]} ({hist[n]})")
    else:
        lines.append("No pity-count mentions matched the regex.")

    lines.append("")
    lines.append("## Informative Comments")
    best = sorted(
        (m for m in mentions if m["source"] == "comment"),
        key=lambda m: (abs((m["n"] or 0) - 80), len(m["text_excerpt"])),
    )[:3]
    if best:
        for idx, item in enumerate(best, start=1):
            lines.append(
                f"{idx}. `{item['keyword']}` / `{item['feed_id']}` / n={item['n']}: {item['text_excerpt']}"
            )
    else:
        lines.append("No informative comment examples available.")

    lines.append("")
    lines.append("## Recommended Next Actions")
    lines.append("1. Run additional search keywords centered on comments, such as `第几次出异色`, `多少次出异色`, and `80次没出`.")
    lines.append("2. Add search-result pagination if the MCP search endpoint can expose more than the first ~20 results.")
    lines.append("3. Tighten comment extraction once the exact `get_feed_detail` response schema is stable, especially for nested reply containers.")
    lines.append("4. If login remains stable, consider a second pass on posts just below the 20-comment threshold to widen the sample.")

    lines.append("")
    lines.append("## Notes")
    lines.append(f"Selected deep-fetch posts: {len(selected_ids)}")
    missing_rank_data = sum(1 for r in records if r.comment_count is None)
    lines.append(f"Search results missing comment-count metadata: {missing_rank_data} / {len(records)}")
    return "\n".join(lines) + "\n"


def deep_fetch(client: MCPClient, post: FeedRecord) -> Any:
    args = {
        "feed_id": post.feed_id,
        "xsec_token": post.xsec_token,
        "load_all_comments": True,
    }
    return client.call_tool("get_feed_detail", args)


def main() -> int:
    if ROOT.resolve() != Path.cwd().resolve():
        log(f"Running from {Path.cwd()} but writing strictly under {ROOT}")
    timestamp = int(time.time())
    records = load_search_feeds()
    ranked = sorted(
        [r for r in records if r.comment_count is not None and r.comment_count >= 20],
        key=lambda r: (r.comment_count or 0, r.liked_count or 0),
        reverse=True,
    )
    selected_posts = ranked[:20]

    if not selected_posts:
        log("No posts met the >=20 comment threshold. Writing empty artifacts.")

    client = MCPClient(MCP_URL)
    log("Initializing MCP session")
    client.initialize()
    client.notify_initialized()

    log("Checking login status")
    login_result = client.call_tool("check_login_status", {})
    login_text = json.dumps(login_result, ensure_ascii=False)
    if "未登录" in login_text or '"loggedIn": false' in login_text or '"isLoggedIn": false' in login_text:
        log("Login check indicates not logged in; stopping.")
        return 2

    detail_path = RAW_DIR / f"details_deep_{timestamp}.jsonl"
    log(f"Streaming details to {detail_path}")
    deep_rows: list[dict[str, Any]] = []
    for idx, post in enumerate(selected_posts, start=1):
        log(f"Deep-fetch {idx}/{len(selected_posts)} {post.feed_id} ({post.keyword})")
        try:
            raw = deep_fetch(client, post)
        except Exception as exc:
            log(f"Transport error on {post.feed_id}: {exc}. Sleeping 30s before retry.")
            time.sleep(30)
            try:
                raw = deep_fetch(client, post)
            except Exception as exc2:
                log(f"Second failure on {post.feed_id}: {exc2}. Skipping.")
                continue

        row = {
            "keyword": post.keyword,
            "feed_id": post.feed_id,
            "title": post.title,
            "author": post.author,
            "comment_count_reported": post.comment_count,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": raw,
        }
        deep_rows.append(row)
        with detail_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"  appended row {idx} ({len(json.dumps(row, ensure_ascii=False))} bytes)")
        if idx < len(selected_posts):
            log("Sleeping 5s before next MCP call")
            time.sleep(5)

    log(f"Finished detail loop; total rows = {len(deep_rows)}")

    mentions, comments_by_feed = extract_mentions(selected_posts, deep_rows)
    mention_path = RAW_DIR / f"pity_mentions_{timestamp}.jsonl"
    jsonl_write(mention_path, mentions)
    log(f"Wrote {mention_path}")

    smoke = build_smoke_test(records, selected_posts, comments_by_feed, mentions)
    smoke_path = RAW_DIR / "SMOKE_TEST.md"
    smoke_path.write_text(smoke, encoding="utf-8")
    log(f"Wrote {smoke_path}")

    log(
        f"Done. Ranked={len(ranked)} selected={len(selected_posts)} fetched={len(deep_rows)} mentions={len(mentions)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
