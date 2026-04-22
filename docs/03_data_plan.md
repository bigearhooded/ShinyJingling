# 03 · 数据采集方案

## 1. 数据源与覆盖度估计

| 平台 | 关键词 | 预期相关帖 | 质量 | 爬取难度 |
|---|---|---|---|---|
| 小红书 | "洛克王国世界 异色" / "异色保底" / "第几次出异色" | 2000+ | 高 (图文并茂) | 需登录, 有 MCP |
| B 站 | 同上, 视频评论区 | 5000+ 评论 | 中 (短评) | 公开 API |
| TapTap | 论坛 app/188212 版块 | 500+ | 高 (长帖) | Playwright |
| 贴吧 | 洛克王国吧 + 世界吧 | 1000+ | 中 | 老 API 易用 |
| 知乎 | 专栏 + 问答 | 200+ | 高 | 公开 |
| 抖音 | 同关键词 | — | 低 (视频较难抽取) | 暂缓 |

## 2. 抓取策略

### 2.1 小红书 (主力)
**工具**: `xpzouying/xiaohongshu-mcp` — Linux binary, HTTP endpoint at `http://localhost:18060/mcp`.

**关键词列表** (滚动执行):
```
洛克王国世界 异色
异色保底 洛克王国
洛克王国 80次保底
洛克王国 梦魇污染
洛克王国异色出货
洛克王国 第几次 异色
{16 种赛季限定异色精灵名} × {"异色", "出货"}
```

**抽取字段** (note_id 为 key):
- `title`, `content`, `image_urls[]`
- `like_count`, `comment_count`, `favorite_count`, `share_count`
- `author_id`, `author_name`, `publish_time`
- `top_comments[]` (若可获得, 评论中常见"第 X 次")

### 2.2 B 站评论
**工具**: `bilibili-api-python` 或直接 `https://api.bilibili.com/x/v2/reply/main`.

目标视频: 搜索 "洛克王国世界 异色 保底", 取前 50 个高播放视频, 抽所有一级评论 + 热门二级评论.

### 2.3 TapTap
论坛 URL: `https://www.taptap.cn/app/188212/topic?type=feed`
Playwright 渲染, 按时间倒序扫 5000 条帖子, 过滤含异色关键词.

## 3. 结构化抽取 pipeline

```
raw text
   │
   ▼
正则粗筛 (含 "第 X 次|用了 X 只|打了 X 只|X 只出")
   │
   ▼
LLM 结构化抽取 (Haiku / GPT-4o-mini, 单条 ~2k tokens)
   │
   ▼
Schema: {
  source: [xhs|bili|taptap|tieba|zhihu],
  source_id: str,
  author_id: str,
  family: str | null,        # "方方" | "小霸王" | ...
  breaks_to_shiny: int,      # 击破次数
  triggered_pity: bool,      # 是否是第 80 次保底
  captured: bool,            # 是否用球捕获
  ball_type: str | null,     # "补光球" | "炫彩球" | ...
  has_screenshot: bool,
  confidence: float,         # LLM 自评置信度
  raw_excerpt: str,          # 原文片段
  publish_time: str,
}
   │
   ▼
PostgreSQL / Parquet
```

### LLM 抽取 prompt 模板
```
从以下帖子抽取异色保底相关数据. 若不相关, 输出 null.

帖子正文:
"""
{text}
"""

输出 JSON:
{
  "family": "...",
  "breaks_to_shiny": int,
  "triggered_pity": bool,
  ...
}
```

## 4. 去重与质控

- **跨平台去重**: 同一用户名在多平台发相同内容用笔记配图 pHash 检出.
- **可信度加权**: 有截图 > 有视频 > 纯文字. 似然函数里加权重 $w_i$.
- **人工金标**: 随机抽 200 条做人工标注, 估计 LLM 抽取准确率.
- **异常剔除**: `breaks_to_shiny > 80` 或 `< 1` 直接丢弃 (除非是 "累计多池" 自述).

## 5. 偏差修正模型

观测到的 $(X_i, \text{posted}_i)$ 中 $\text{posted}_i = 1$ (只看到发帖者).
真实似然 = $P(X \mid p) \cdot w(X)$ 归一化.
假设 $w(x) = \sigma(\beta_0 + \beta_1 \mathbb{1}[x<30] + \beta_2 \mathbb{1}[x\ge 78])$, 用 MCMC 联合估计 $(p, \beta)$.

## 6. 预算与节流

- 小红书 MCP: ~1 req/s 建议上限, 2000 笔记 ≈ 35 min.
- B 站 API: ~5 req/s (带抖动 + 代理), 5000 条 ≈ 20 min.
- TapTap: ~0.5 req/s (PW 渲染慢), 500 帖 ≈ 17 min.
- LLM 抽取成本: Haiku @ $0.80/M input, 2M tokens ≈ $1.6. 总预算 < $5.

## 7. Codex 委派流程

1. Codex 通过 MCP 分批调用 `search_feeds` / `get_feed_detail`.
2. 结果存 `data/raw/xhs_{keyword}_{timestamp}.jsonl`.
3. Codex 跑 `src/scrapers/extract.py` 调 LLM 做结构化抽取.
4. 输出 `data/processed/shiny_reports.parquet`.

每批 100 条 smoke test → 若质量 OK 则放开全量.
