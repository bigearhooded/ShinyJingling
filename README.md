# Shiny Jingling: 《洛克王国：世界》异色精灵保底机制研究

> 你被官方骗了 3 倍：异色精灵真实保底机制的家族分层贝叶斯识别与最优捕捉策略

腾讯《洛克王国：世界》（4.16 版本）异色精灵获取服从一种带硬保底的截断几何过程：每次击破"噩梦污染体"以官方公布的"综合概率约 0.02"出货，未出则进入下一次，第 80 次必出。本仓库是对该机制的系统贝叶斯识别 + 最优捕捉策略推导。

## 核心发现

- **官方公示的"综合概率 0.02"被强烈拒绝**：在双向自报偏差校正后，三模型综合出货率后验集中在 0.06--0.09，是官方的 **3--4.5 倍**（拒绝官方概率的后验质量大于 0.999）
- **数据偏好硬保底**：M1 LOO 严格主导（Δelpd > 13），软保底证据缺失，民间"70--75 冲刺大法"在 M1 无记忆性下统计无效
- **家族异质性显著**：分层 Beta-Binomial 给出家族级 $\hat p_{0,f} \in [0.08, 0.17]$，最易出（贝瑟）与最难出（恶魔狼）相差 2 倍
- **最优策略 = 属性球抓污染 + 普通球摔野怪 + 单颗捕光球抓异色**：单轮耗时 ~0.5h 始终是绑定约束，三档玩家产出：轻度 0.85 / 中度 5.1 / 重度 17 异色/天
- **棱镜球氪金不值**：升格期权需炫彩 collection premium > 640 倍才回本

## 目录结构

```
shiny-jingling/
├── README.md                       本文件
├── paper/
│   ├── main.tex                    论文 LaTeX 源 (XeLaTeX + xeCJK)
│   ├── main.pdf                    编译后 PDF (10 页)
│   ├── figures/                    论文用图 (24 张 PNG)
│   ├── LOGO1.png, LOGO2.png        S.H.*.T Journal LOGO
├── src/
│   ├── analysis/                   13 个 Python 模块 (清洗 / 模型 / 仿真 / 绘图)
│   │   ├── clean_pity_dataset.py
│   │   ├── inspect_clean_dataset.py
│   │   ├── fit_pity_models.py        全局 M1/M2/M3 PyMC 识别
│   │   ├── fit_family_hierarchical.py 家族分层 Beta-Binomial
│   │   ├── simulate_strategies.py    经济仿真 (toy)
│   │   ├── simulate_strategies_v2.py 4.16 真实球价仿真
│   │   ├── multi_pool_optimization.py 多池贪心
│   │   ├── time_budget_lp.py         时间预算 LP
│   │   ├── _plotting.py              CJK 字体加载共享 helper
│   │   └── (其他: 数据抽取/合并/嵌入召回/LLM 提取)
│   └── scrapers/
│       ├── bili_fetch.py             B 站评论爬虫
│       └── xhs_deep_fetch.py         XHS MCP 包装器
├── data/
│   └── processed/
│       ├── pity_clean.jsonl          清洗后数据集 (616 条)
│       ├── llm_pity_extractions.jsonl LLM 抽取原始 (618 条)
│       ├── pity_clean_summary.md
│       ├── inspection_report.md
│       ├── strategy_table.md
│       ├── strategy_v2_summary.md
│       ├── multi_pool_summary.md
│       ├── time_budget_summary.md
│       └── pity_fits/
│           ├── comparison.csv         LOO 对比
│           ├── posterior_summary.md
│           ├── M1_summary.csv / M2_summary.csv / M3_summary.csv
│           ├── family_summary.csv     家族级后验
│           └── family_summary.md
├── docs/
│   ├── 01_research_design.md         V2 研究设计
│   ├── 02_game_mechanics.md          游戏机制说明
│   ├── 03_data_plan.md               数据采集方案
│   ├── 04_economic_model.md          经济模型推导
│   ├── 05_smoke_test_findings.md     冒烟测试报告
│   ├── 06_bayesian_findings.md       贝叶斯诊断
│   ├── 07_final_report.md            阶段总报告
│   └── 08_economic_params.md         4.16 经济参数
└── figures/                          24 张图 (与 paper/figures 重复)
```

## 复现

### 1. 环境

需要 Python 3.10+ 与 PyMC 5.x：

```bash
pip install pymc>=5.20 arviz>=0.18 numpy pandas matplotlib scipy scikit-learn pytensor
```

CJK 字体（绘图必需）：把 `NotoSansCJKsc-Regular.otf` 和 `NotoSansCJKsc-Bold.otf` 放到 `./fonts/` 或修改 `src/analysis/_plotting.py` 中 `_CJK_DIR` 路径。

LaTeX 编译需 XeLaTeX + xeCJK（思源宋体/思源黑体）：

```bash
xelatex paper/main.tex && xelatex paper/main.tex
```

### 2. 全流程

```bash
# 数据清洗
python src/analysis/clean_pity_dataset.py

# 数据 EDA
python src/analysis/inspect_clean_dataset.py

# 全局 Bayesian 识别 (M1/M2/M3 + LOO)
python src/analysis/fit_pity_models.py

# 家族分层识别 (核心新增)
python src/analysis/fit_family_hierarchical.py

# 经济仿真 (toy + V2 真实球价)
python src/analysis/simulate_strategies.py
python src/analysis/simulate_strategies_v2.py

# 多池 + 时间预算 LP
python src/analysis/multi_pool_optimization.py
python src/analysis/time_budget_lp.py
```

### 3. 重新爬虫

需要 B 站 cookie。本仓库**不包含** cookie 文件（脱敏排除）。如需重新采集：

```bash
# 在 bin/bili_cookie.env 中放置 SESSDATA 等字段
python src/scrapers/bili_fetch.py
```

## 数据集说明

`data/processed/pity_clean.jsonl` 是清洗后的玩家自报数据，每条包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `n` | int | 报告的污染击破次数（1--80） |
| `scope` | str | `single_pool` / `cross_pool_total` / `qualitative_no_n` |
| `family_hint` | str/null | 玩家提到的精灵家族（如"恶魔狼"、"贝瑟"） |
| `ball_or_pollution` | str | 是球数 vs 污染数 |
| `confidence` | str | LLM 抽取置信度 high/low |
| `excerpt` | str | 原文片段 |
| `rpid` | int | B 站评论 rpid（公开数据） |
| `source` | str | regex / embed / both |

`single_pool` 子集（266 条）是建模主体。

## 引用

如果你用了本仓库的数据/方法，请引用：

```bibtex
@misc{bigearhooded2026shiny,
  title  = {你被官方骗了 3 倍：《洛克王国：世界》异色精灵真实保底机制的家族分层贝叶斯识别与最优捕捉策略},
  author = {大耳帽兜（Big-eared Hooded Bunny）},
  year   = {2026},
  note   = {S.H.*.T Journal preprint},
  url    = {https://github.com/bigearhooded/ShinyJingling}
}
```

## 致谢

本研究由**安妮**（商店街安妮商店店长）赞助。安妮女士在 4.16 版本主动将高级球价格由 15,000 洛克贝下调至 12,000 洛克贝以应对远行商人竞争，使本研究"高级球抓污染"路线的经济仿真得以基于真实交易价格——这一价格调整是表 IV 与第 VIII 节策略推论的基础常数。

感谢 B 站玩家社区对异色保底机制的开放讨论；感谢魔法学院全域魔法传送阵系统的稳定运行使多池并行实验中家族池切换零延迟；S.H.*.T Journal 反串学术风格启发了本文的标题与组织。

## License

MIT License. 见 `LICENSE`。
