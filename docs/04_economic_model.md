# 04 · 经济模型数学推导

## 1. 单次战斗利润

**状态**: 当前已击破 $k-1$ 次, 正在打第 $k$ 只污染体.

**决策**: $\alpha_k \in \{0, 1\}$, 是否在护盾破后下球.

**单次利润**:
$$
\pi_k = \alpha_k \cdot \mathbb{1}[\text{cap}] \cdot m_{\text{poll}} \cdot v_0
      - \alpha_k \cdot c_{\text{ball}}
      + \mathbb{1}[\text{shiny}_k] \cdot \bigl( \mathbb{1}[\text{cap}_s] \cdot m_{\text{shiny}} \cdot v_0 - c_{\text{ball}}^s \bigr)
$$

其中第三项为异色命中时的"彩票奖励" (一定会花一颗好球抓).

**期望**:
$$
\mathbb{E}[\pi_k] = \alpha_k (q_{\text{cap}} m_{\text{poll}} v_0 - c_{\text{ball}})
                 + p_k (q_{\text{cap}}^s m_{\text{shiny}} v_0 - c_{\text{ball}}^s)
$$

## 2. 循环期望利润

定义 $\tau = \min\{k : \text{shiny hits}\}$, $\tau \le 80$ 保证.

**循环利润**:
$$
\Pi(S) = \sum_{k=1}^{\tau} \pi_k
$$

**循环期望** (策略 $S$):
$$
\mathbb{E}[\Pi(S)] = \sum_{k=1}^{80} \mathbb{E}[\pi_k \mid S] \cdot P(\tau \ge k)
$$

其中 $P(\tau \ge k) = \prod_{j=1}^{k-1}(1 - p_j)$.

## 3. 纯几何下的闭式

假设 **M1 硬保底, $p_k \equiv p_0 = 0.018$**, $q_{\text{cap}} = 1$ (补光球对目标属性), $q_{\text{cap}}^s = 1$ (炫彩球), 定义 $q = 1 - p_0$.

$P(\tau \ge k) = q^{k-1}$ for $k \le 80$, $P(\tau \ge 81) = 0$.

$$
\sum_{k=1}^{80} q^{k-1} = \frac{1 - q^{80}}{1 - q} = \frac{1 - 0.982^{80}}{0.018} \approx \frac{1 - 0.2325}{0.018} \approx 42.64
$$

这就是 **期望战斗次数** $\mathbb{E}[\tau]$.

**策略 S-F** (全抓, $\alpha \equiv 1$):
$$
\mathbb{E}[\Pi_{SF}] = (m_{\text{poll}} v_0 - c_{\text{ball}}) \cdot \mathbb{E}[\tau] + p_{\text{合}} \cdot m_{\text{shiny}} v_0 - c^s_{\text{ball}}
$$

其中 $p_{\text{合}} = \mathbb{E}[\mathbb{1}[\text{shiny hits before 80}]] = 1$ (因硬保底一定出).

简化:
$$
\boxed{\mathbb{E}[\Pi_{SF}] = 42.64 \cdot (5 v_0 - c_{\text{ball}}) + 10 v_0 - c_{\text{ball}}^s}
$$

**策略 S-P** (纯保底, $\alpha \equiv 0$):
$$
\boxed{\mathbb{E}[\Pi_{SP}] = 10 v_0 - c_{\text{ball}}^s}
$$

**临界球成本** $c^*$: 使两者相等的 $c_{\text{ball}}$:
$$
c^* = 5 v_0
$$

即**球成本低于 $5 v_0$ 时, S-F 严格优于 S-P**; 反之亦然.

在 4.16 后合成无洛克贝消耗, 实际 $c_{\text{ball}}$ 只计作物成本 (作物种植 30 分钟出 1 颗 ≈ 几十洛克贝). 即使精灵基础值 $v_0 \sim 50$, $5 v_0 = 250 \gg c_{\text{ball}}$, **S-F 显著最优**.

## 4. M2/M3 下的半解析

在软保底下 $p_k$ 随 $k$ 增大, $P(\tau \ge k)$ 衰减更快. 数值积分 (或仿真) 即可.

关键观察: $\mathbb{E}[\tau]_{M2} < \mathbb{E}[\tau]_{M1}$, 但**策略选择** (S-F 还是 S-P) 与 $p_k$ 曲线**无关**, 只取决于 $5 v_0 \gtrless c_{\text{ball}}$ — 因此即使模型不确定, 策略结论稳健.

## 5. 多池并行

设玩家同时推进 $N$ 个家族池. 每日预算 $T$ 次战斗 (受作物/时间约束).

**问题**: 如何把 $T$ 分配到 $N$ 个池以最大化**每日期望异色数**?

### 5.1 无保底下 (纯几何)
每池独立, 分配 $T_n$ 到池 $n$: 期望异色数 = $\sum_n (1 - (1-p)^{T_n})$. 因为 $1-(1-p)^T$ 是凹的, **均分最优**.

### 5.2 有硬保底
池 $n$ 已有进度 $k_n$. 剩余保底距离 $80 - k_n$. 投入 $T_n$ 战斗到该池, 期望异色数:
$$
E_n(T_n) = \sum_{j=1}^{\min(T_n, 80-k_n)} p \cdot (1-p)^{j-1} + \mathbb{1}[T_n \ge 80-k_n] \cdot (1-p)^{80-k_n-1}
$$

- 池已接近 80 → 投入更少步就出保底 → **优先冲保底池**
- 新池 ($k_n = 0$) → 收益曲线陡峭上升但很远 → 不优先

这是一个**带资源约束的多臂赌博机**, 可用贪心+前瞻或 DP 求解.

### 5.3 民间 "70-75 冲刺大法" 的分析
若 M1 成立 (无记忆): 投入 $k=71$ 和 $k=1$ 的每步期望异色相同 = $p$. 冲刺无收益.
若 M2 成立: $k\ge k^*$ 时 $p_k > p$, 每步期望异色高 → 冲刺有收益.
**因此, 观察玩家是否通过冲刺显著提高出货率, 可以反向证伪/证实软保底.**

## 6. 期权视角

持有异色后可用**棱镜球** (稀缺) 转为**异色炫彩** (外观极稀 + 满资质).

令 $V_{\text{shiny}}$ = 异色价值 (回顾值 10 $v_0$ 或收藏效用), $V_{\text{prism}}$ = 炫彩异色价值, $c_p$ = 棱镜球价值 (市场估价).

**持有异色选择**:
- Ex: 直接回顾 → 锁定 $10 v_0$
- Hold: 等到有棱镜球 → 升格 → 可回顾得 10 v_0 (炫彩 10×), 但炫彩形态有**额外收藏溢价**

类似 American call option: 异色 = underlying, 棱镜球 = strike, 行权价值 = $V_{\text{prism}} - V_{\text{shiny}}$.

玩家理性行为: 仅当棱镜球获得成本 < 炫彩溢价时行权.

## 7. 非酋上限定理 (Claim)

**命题**: 在 4.16 版本参数下, 采取 S-F 策略, **每循环净利润的 0 分位数 (最差情况) 严格大于 0**.

**证明草稿**: 最差情况 = 80 次全打完才保底出货. 利润:
$$
\Pi_{\min} = 80 \cdot (5 v_0 - c_{\text{ball}}) + 10 v_0 - c_{\text{ball}}^s = 410 v_0 - 80 c_{\text{ball}} - c_{\text{ball}}^s
$$

只要 $c_{\text{ball}} < 5.125 v_0$, $\Pi_{\min} > 0$. 如前所述, 实际 $c_{\text{ball}} \ll 5 v_0$, 故**极度非酋玩家仍有正收益** — 这是本研究最有传播力的结论.

## 8. 待估参数

| 参数 | 估计方法 |
|---|---|
| $p_0, p_k$ | 贝叶斯 MCMC on scraped data |
| $v_0$ | 社区公开帖 / 主播实测 / 游戏内截图 |
| $c_{\text{ball}}$ | 作物成本折算 |
| $q_{\text{cap}}, q_{\text{cap}}^s$ | 官方公告 + 玩家实测 |
| $V_{\text{prism}}$ | 市场调研 + 玩家访谈 |
