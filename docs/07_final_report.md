# 07 · 异色狩猎研究 — 阶段性总报告

> **数据**: 自建 B 站评论爬虫 + LLM 抽取, N = 616 cleaned reports (266 single_pool / 280 cross_pool / 70 qualitative)
> **方法**: PyMC 5.25 NUTS 贝叶斯估计 + 选择偏差修正 + 仿真
> **范围**: 《洛克王国: 世界》 4.16 版本异色精灵保底机制

---

## 一句话总结

> 在校正了"晒福报 + 保底投诉"双向 posting bias 后, **数据显示真实综合出货率约 0.06–0.09, 是官方公布"综合概率 0.02"的 3–4.5 倍** — 即官方保守低报. 同时机制偏好"硬保底" (单点 80 必出), 软保底证据缺失. 经济上**全抓策略 (S-F) 在所有 $p_0$ 下显著最优**, 即使最差非酋玩家每循环净利润也 ≈ +19000 洛克贝-equiv. 多池均分 vs 集中: **N=16 池均分比单池集中高 8 倍效率**.

---

## 二. 数据管道与质量

```
LLM 抽取 (8644 条) → embed recall (5000) → 合并去重 (11658) → LLM 二次提取 (618) → 手工清洗 (616)
                                                                                       │
                                                                                       ▼
                                                          single_pool 266 (建模主体)
                                                          cross_pool   280
                                                          qualitative   70
```

清洗规则 (见 `src/analysis/clean_pity_dataset.py`):
1. 删除 LLM 数字解析错误 (例: "5.60次污染" 被解析为 5600; "1000球" 被当成污染数)
2. single_pool n>80 重分类到 cross_pool (硬保底物理上不可能)
3. 保留 70 qualitative reports 作为可能的 informative prior

---

## 三. 贝叶斯模型识别 (Stage 2/3)

### 3.1 模型空间

| 模型 | $p_k$ 形式 | 自由参数 |
|---|---|---|
| **M1 硬保底** | $p_k = p_0$ for $k<80$, $p_{80}=1$ | $p_0$ |
| **M2 线性软保底** | $p_k = p_0$ for $k \le k^*$, 线性升至 $p_{80}=1$ | $p_0, k^*$ |
| **M3 凸软保底** | $p_k = p_0 + (1-p_0)(k/80)^\eta$ | $p_0, \eta$ |

每个模型加入 4 段 posting weight:
$$
\log w(k) = b_0 + b_\text{first}\cdot \mathbf 1[k=1] + b_\text{low}\cdot \mathbf 1[2 \le k \le 29] + b_\text{pity}\cdot \mathbf 1[k \ge 78]
$$

### 3.2 主结果

**LOO 排序**: M1 > M2 > M3 (LOO weight 1.0 / ~0 / ~0)

| 模型 | $\hat p_0$ | $1/\hat E[\tau]$ (= 综合出货率) | $P(p_0 < 0.0143)$ | $P(p_0 < 0.018)$ |
|---|---|---|---|---|
| M1 | **0.092** | **0.092** | 0.00% | 0.00% |
| M2 | **0.062** | **0.063** | 0.00% | 0.00% |
| M3 | **0.062** | **0.062** | 0.00% | 0.00% |

> 0.0143 = 官方"综合概率 0.02"在 M1 下对应的 per-trial $p_0$.
> 0.018 = 玩家社区反推的 per-trial 估计.

### 3.3 Posting bias 量级 (M2 后验)

| 段 | $\hat b$ | 相对中段倍数 |
|---|---|---|
| n = 1 (一发入魂) | 2.30 | **~10×** |
| 2 ≤ n ≤ 29 (早期) | 0.09 | ~1.1× |
| n ≥ 78 (保底投诉) | 2.30 | **~10×** |

> 中段 30–77 几乎被 selection 完全抹平. n=1 和 n≥78 各被超采样 ~10 倍.
> 这与 V2 §1.3 "凡尔赛 + 终于肝完" 双峰假说完全吻合.

### 3.4 Identifiability 警示

M1 和 M2/M3 的 $p_0$ 不同 (0.092 vs 0.062), 反映"高 $p_0$ + 弱 bias" 与 "低 $p_0$ + 强 bias" 不可分.
鲁棒结论: $\bar p \in [0.06, 0.09]$, 至少 **3× 官方综合概率**.

### 3.5 PPC 检查

`figures/09_ppc.png` — 三个模型的 posterior predictive replicates 都基本覆盖观测形状.
**无法拟合的 artifact**: n=10/20/30/40/50/60/70 的整十数 spike (玩家凑整心理), 这是文化 artifact 不是模型问题.

---

## 四. 经济模型 (Stage 4)

### 4.1 单循环利润 (per V2 §1.2)

$$
\mathbb E[\Pi(S)] = \sum_{k=1}^{80} \mathbb E[\pi_k | S] \cdot P(\tau \ge k)
$$

仿真参数: $v_0 = 50$, $c_\text{ball} = 5$, $c^s_\text{ball} = 50$, $q = 0.95$, $m_\text{poll}=5$, $m_\text{shiny}=10$.

| 场景 | $E[\tau]$ | $E[\Pi_{S-P}]$ | $E[\Pi_{S-F}]$ | $\Delta$ |
|---|---|---|---|---|
| $p_0=0.018$ M1 (player guess) | 42.6 | 425 | **10321** | +9896 |
| $p_0=0.062$ M2 (后验) | 16.0 | 425 | **4152** | +3727 |
| $p_0=0.092$ M1 (M1 后验) | 10.9 | 425 | **2951** | +2526 |

### 4.2 临界球成本

$$
c^* = q \cdot m_\text{poll} \cdot v_0 = 0.95 \times 5 \times 50 = \mathbf{237.5}
$$

实际作物成本 $\ll c^*$, **S-F 全抓在所有 $p_0$ 下严格最优**, 策略选择对 $p_0$ 不敏感.

### 4.3 非酋上限定理 (V2 §7)

最差情况 (k=80 才出, 全抓 80 次):
$$
\Pi_\text{min} = 80 \cdot (q \cdot m_\text{poll} \cdot v_0 - c_\text{ball}) + q \cdot m_\text{shiny} \cdot v_0 - c^s_\text{ball} \approx +19025
$$

**即使最非酋的玩家全抓策略下每循环仍稳赚 ~19000 洛克贝-equiv. 这是研究最有传播力的反直觉结论.**

### 4.4 多池并行 (V2 §5)

T=160 总预算下:
| N (并行池数) | E[total shinies] | 倍率 |
|---|---|---|
| 1 (集中) | 1.00 | 1.0× |
| 2 | 2.00 | 2.0× |
| 4 | 3.70 | 3.7× |
| 8 | 5.78 | 5.8× |
| 16 | 7.55 | **7.5×** |

> 多池均分**严格优于**集中. 16 池效率比单池高 7.5 倍.
> 实战意义: 季节性轮换 +能同时刷多家族的 setup 是高 ROI 配置.

### 4.5 70-75 冲刺大法之伪

`figures/13_pity_sprint_test.png`:
- M1 (硬保底, 无记忆): 70-79 步累计概率线性上升 (常数 $p_0$), 80 步必出
- M2 (k*=78): 78 前线性, 79-80 飙升至 1
- M3 (eta=19): 70-75 累计 ~50%, 后段平缓上升

> LOO 选 M1 ⇒ 数据**不支持**冲刺有任何特殊收益.
> 在 M1 下 $p_{70} \approx p_1 \approx p_0$, 冲刺只是心理安慰.

---

## 五. 核心可发表结论 (按传播力排序)

1. **官方"综合概率 0.02"被强烈拒绝**. 双向 posting bias 修正后, 后验综合出货率 0.06–0.09, 是官方公布的 **3–4.5 倍**. ❗ 反直觉: **官方保守低报对玩家是利好欺骗**.

2. **数据偏好硬保底, 软保底证据缺失**. M1 LOO 主导, M2 的 $\hat k^* = 78$ (软保底窗口仅 2 步), M3 的 $\hat \eta = 19.3$ (curve 极陡). 民间"冲刺大法"无效.

3. **全抓策略 (S-F) 对所有 $p_0$ 一致最优**. 临界球成本 $c^* \approx 237$ 远高于实际作物成本. 即使最差非酋玩家每循环仍稳赚 +19000 洛克贝-equiv.

4. **多池均分严格优于集中**. T=160 下 16 池均分比单池高 7.5x 期望异色数.

5. **Posting bias 是发布平台的本质属性**. n=1 (晒福报) 和 n≥78 (保底投诉) 各被超采样 ~10 倍, 中段 30-77 完全沉默. **这意味着所有"自报 gacha 数据"研究都必须做 selection bias 校正**.

---

## 六. 研究局限

1. **数据源单一**: 只爬了 B 站评论, 缺 XHS / 贴吧 / TapTap 多平台对照
2. **Posting weight 模型简化**: 4 段 indicator, 真实 $w(k)$ 可能更平滑
3. **无人工金标**: LLM 抽取准确率未交叉验证 (V2 §3.4 计划但未执行)
4. **Identifiability**: $p_0$ 估计依赖 posting weight 假设, 鲁棒区间 [0.06, 0.09] 但具体值不可分
5. **机制混淆**: 数据看似硬保底, 但也可能是软保底窗口太窄 + posting bias 抹平
6. **未做横向对比 (V2 §3.4)**: 与原神/FGO/Arknights 的"方差-均值谱" 对比未做

---

## 七. 待做 (按价值排序)

| 优先级 | 工作 | 依赖 |
|---|---|---|
| 高 | 写公众号长文 / B站视频 (传播 1-4 结论) | 现有结果 + 设计 |
| 中 | XHS / 贴吧补充数据交叉验证 | smoke test 已通的爬虫栈 |
| 中 | 与玩家约 20 条 ground-truth (录播验证 LLM 抽取) | 找愿意配合的玩家 |
| 低 | 期权定价 (棱镜球→炫彩异色) | 棱镜球市场价数据 (难) |
| 低 | 横向 gacha 对比 (原神 / FGO) | 各家公开概率数据 |

---

## 八. 文件索引

### 数据
- `data/processed/pity_clean.jsonl` — 清洗后建模数据 (616 行)
- `data/processed/llm_pity_extractions.jsonl` — LLM 抽取原始 (618 行)
- `data/processed/pity_clean_summary.md` — 清洗 audit
- `data/processed/inspection_report.md` — 数据 EDA 报告

### 模型
- `data/processed/pity_fits/M1.nc M2.nc M3.nc` — InferenceData
- `data/processed/pity_fits/comparison.csv` — LOO 表
- `data/processed/pity_fits/posterior_summary.md` — 模型综述
- `data/processed/strategy_table.md` — S-P / S-F / 非酋上限
- `data/processed/multi_pool_summary.md` — 多池策略

### 图 (`figures/`)
| 编号 | 文件 | 内容 |
|---|---|---|
| 01 | `01_single_pool_hist.png` | n 分布直方图 |
| 02 | `02_posting_bias.png` | 双峰 posting 区域 |
| 03 | `03_empirical_vs_geometric.png` | 经验 CDF vs Geometric(0.018/0.02) |
| 04 | `04_family_breakdown.png` | top-15 家族 |
| 05 | `05_posterior_p0.png` | $p_0$ 后验 vs 官方 |
| 06 | `06_posterior_w_function.png` | posting weight 后验 |
| 07 | `07_implied_pmf_overlay.png` | 加权 PMF vs 观测 |
| 08 | `08_model_comparison_loo.png` | LOO 排序 |
| 09 | `09_ppc.png` | PPC (60 replicates × 3) |
| 10 | `10_strategy_profit_3p0.png` | S-P vs S-F barchart |
| 11 | `11_critical_ball_cost.png` | break-even $c^*$ |
| 12 | `12_S_C_threshold_curve.png` | S-C 最优阈值 |
| 13 | `13_pity_sprint_test.png` | 70-75 冲刺验证 |
| 14 | `14_marginal_shiny_per_battle.png` | 边际 $p_k$ |
| 15 | `15_optimal_allocation_T100.png` | 异质池贪心分配 |
| 16 | `16_concentrate_vs_split.png` | 集中 vs 均分 |

### 代码 (`src/analysis/`)
- `clean_pity_dataset.py` — 数据清洗
- `inspect_clean_dataset.py` — EDA + plots 01-04
- `fit_pity_models.py` — PyMC M1/M2/M3 + plots 05-09
- `simulate_strategies.py` — 经济仿真 + plots 10-13
- `multi_pool_optimization.py` — 多池 + plots 14-16

### 文档 (`docs/`)
- `01_research_design.md` — V2 设计
- `02_game_mechanics.md` — 游戏机制
- `03_data_plan.md` — 数据采集方案
- `04_economic_model.md` — 经济模型推导
- `05_smoke_test_findings.md` — Smoke test 报告
- `06_bayesian_findings.md` — Stage 2/3 诊断
- `07_final_report.md` — 本文件
