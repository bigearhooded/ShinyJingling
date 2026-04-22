"""Strategy economics simulator (V2 §1.2 / §3.2 / §5).

Uses the Stage-2 posteriors as scenarios for p0:
    - official:   p0 = 0.018   (官方公布)
    - M2/M3 post: p0 = 0.062   (bias-corrected base case)
    - M1 post:    p0 = 0.092   (alt mode)

For each (p0, mechanism in {M1, M2, M3}) and each strategy
in {S-P, S-F, S-C(k_c)} computes:
    - expected battles to first shiny  E[tau]
    - expected per-cycle profit       E[Pi(S)]
    - worst-case profit floor (n=80)

Outputs:
    figures/10_strategy_profit_3p0.png       — bar chart S-P vs S-F across p0
    figures/11_critical_ball_cost.png        — break-even c_ball under S-F
    figures/12_S_C_threshold_curve.png       — best k_c for conditional strategy
    figures/13_pity_sprint_test.png          — "70–75 冲刺大法" expected gain
    data/processed/strategy_table.md         — narrative summary
"""
from __future__ import annotations

from pathlib import Path

from _plotting import setup as _setup_plotting; _setup_plotting()
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(".")
FIG = PROJ / "figures"
OUT = PROJ / "data/processed/strategy_table.md"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 110

K_MAX = 80

# ───────────────────────── per-trial p_k builders ───────────────────────────


def pk_M1(p0):
    pk = np.full(K_MAX, p0, dtype=float)
    pk[-1] = 1.0
    return pk


def pk_M2(p0, k_star=78):
    ks = np.arange(1, K_MAX + 1, dtype=float)
    gamma = (1.0 - p0) / max(K_MAX - k_star, 1e-6)
    pk = np.where(ks <= k_star, p0, p0 + gamma * (ks - k_star))
    pk = np.clip(pk, 1e-9, 1.0)
    pk[-1] = 1.0
    return pk


def pk_M3(p0, eta=19.3):
    ks = np.arange(1, K_MAX + 1, dtype=float)
    pk = p0 + (1.0 - p0) * (ks / K_MAX) ** eta
    pk = np.clip(pk, 1e-9, 1.0)
    pk[-1] = 1.0
    return pk


# ───────────────────────── core economic block ──────────────────────────────


def survive(pk):
    """P(tau >= k) for k=1..80. Length 80 vector."""
    out = np.empty(K_MAX, dtype=float)
    out[0] = 1.0
    out[1:] = np.cumprod(1.0 - pk[:-1])
    return out


def expected_battles(pk):
    """E[tau] = sum_{k=1}^{80} P(tau >= k)."""
    return float(np.sum(survive(pk)))


def expected_profit(pk, alpha, v0, c_ball, c_ball_s,
                    q_cap=0.95, q_cap_s=0.95,
                    m_poll=5.0, m_shiny=10.0):
    """E[Pi(S)] = sum_k E[pi_k | S] · P(tau >= k).

    alpha is a length-80 vector of {0,1} indicators per battle.
    """
    s = survive(pk)
    # per-battle expected payoff
    per_pollution = alpha * (q_cap * m_poll * v0 - c_ball)
    per_shiny = pk * (q_cap_s * m_shiny * v0 - c_ball_s)
    return float(np.sum((per_pollution + per_shiny) * s))


def worst_case_profit(pk, alpha, v0, c_ball, c_ball_s,
                      q_cap=0.95, q_cap_s=0.95,
                      m_poll=5.0, m_shiny=10.0):
    """Worst-case = pity hits at exactly 80. Catches all 80 pollution + 1 shiny.
    NOTE: this is *one* deterministic realization, not E[min]."""
    pollution_take = np.sum(alpha) * (q_cap * m_poll * v0 - c_ball)
    shiny_take = q_cap_s * m_shiny * v0 - c_ball_s
    return pollution_take + shiny_take


# ───────────────────────── strategy library ─────────────────────────────────


def alpha_SP():
    return np.zeros(K_MAX, dtype=float)


def alpha_SF():
    return np.ones(K_MAX, dtype=float)


def alpha_SC(k_c):
    a = np.zeros(K_MAX, dtype=float)
    a[k_c - 1:] = 1.0
    return a


# ───────────────────────── plot helpers ─────────────────────────────────────


def plot_strategy_profit(scenarios, params):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    xs = np.arange(len(scenarios))
    sp_vals, sf_vals = [], []
    for label, pk in scenarios:
        sp_vals.append(expected_profit(pk, alpha_SP(), **params))
        sf_vals.append(expected_profit(pk, alpha_SF(), **params))
    ax.bar(xs - width / 2, sp_vals, width, label="S-P (纯保底)", color="#a1d99b")
    ax.bar(xs + width / 2, sf_vals, width, label="S-F (全抓)", color="#fdae6b")
    for i, (sp, sf) in enumerate(zip(sp_vals, sf_vals)):
        ax.text(xs[i] - width / 2, sp, f"{sp:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xs[i] + width / 2, sf, f"{sf:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("E[per-cycle profit] (洛克贝-equiv)")
    ax.set_title(
        f"10 · S-P vs S-F across (p0, mechanism)  "
        f"v0={params['v0']}  c_ball={params['c_ball']}  c_ball_s={params['c_ball_s']}"
    )
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "10_strategy_profit_3p0.png", bbox_inches="tight")
    plt.close(fig)
    return list(zip([s[0] for s in scenarios], sp_vals, sf_vals))


def plot_critical_ball_cost(p0_grid=(0.018, 0.062, 0.092), v0=50.0):
    """E[Pi_SF] - E[Pi_SP] = (m_poll·q·v0 - c_ball)·E[tau]; root in c_ball."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c_ball = np.linspace(0, 8 * v0, 200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, p0 in zip(colors, p0_grid):
        pk = pk_M1(p0)
        et = expected_battles(pk)
        # delta = E[SF] - E[SP] in M1 closed form (per §3 doc):
        # SF - SP = (q*m_poll*v0 - c_ball) * E[tau]
        delta = (0.95 * 5.0 * v0 - c_ball) * et
        ax.plot(c_ball, delta, color=color, lw=2,
                label=f"p0={p0:.3f}  E[tau]={et:.1f}")
    c_star = 0.95 * 5.0 * v0
    ax.axvline(c_star, color="red", linestyle="--", lw=1,
                label=f"break-even c*={c_star:.1f} (= q·m_poll·v0)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("ball cost  c_ball  (洛克贝-equiv)")
    ax.set_ylabel("E[Pi_SF] - E[Pi_SP]  (per cycle)")
    ax.set_title("11 · break-even ball cost — S-F vs S-P (M1 closed form)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "11_critical_ball_cost.png", bbox_inches="tight")
    plt.close(fig)
    return c_star


def plot_SC_threshold(p0=0.062, v0=50.0, c_ball_grid=(2.5, 5.0, 12.5, 25.0, 50.0)):
    """For each c_ball, sweep k_c and pick the optimal threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    pk = pk_M2(p0)
    ks = np.arange(1, K_MAX + 1)
    for c in c_ball_grid:
        params = dict(v0=v0, c_ball=c, c_ball_s=v0,
                       q_cap=0.95, q_cap_s=0.95)
        vals = []
        for kc in ks:
            vals.append(expected_profit(pk, alpha_SC(kc), **params))
        ax.plot(ks, vals, lw=1.6, label=f"c_ball={c:.1f}")
        kc_star = int(ks[np.argmax(vals)])
        ax.scatter([kc_star], [vals[kc_star - 1]], s=40, zorder=3)
    ax.set_xlabel("k_c  (S-C threshold: catch when k >= k_c)")
    ax.set_ylabel("E[per-cycle profit]")
    ax.set_title(f"12 · S-C optimal threshold (p0={p0}, M2 mechanism, v0={v0})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "12_S_C_threshold_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_pity_sprint_test(p0=0.062, v0=50.0):
    """民间 '70-75 冲刺大法': conditional on having survived 69 trials
    without a shiny, what's the probability of shiny in the NEXT m trials?

    Under M1 (hard pity at 80) the curve is linear for m=1..10 then jumps
    to 1 at m=11 (trial 80 forced). Under M2 (k*=78) it accelerates from
    m=9 onward; under M3 (eta=19) the acceleration is gentler but starts
    earlier.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, pk in [("M1 (hard pity)", pk_M1(p0)),
                      ("M2 (k*=78)", pk_M2(p0, 78)),
                      ("M3 (eta=19)", pk_M3(p0, 19.3))]:
        cond_pks = pk[69:]            # p_k for k=70..80, length 11
        # m additional trials past k=69 → trial number 69+m completed.
        # P(shiny by trial 69+m | no shiny in 1..69) =
        #     1 - prod_{i=0..m-1}(1 - cond_pks[i])
        m_grid = np.arange(0, 12)  # 0..11
        ys = [0.0]
        for m in range(1, 12):
            ys.append(1.0 - float(np.prod(1.0 - cond_pks[:m])))
        trial_x = 69 + m_grid       # 69..80
        ax.plot(trial_x, ys, lw=2, marker="o", label=name)
    ax.axvline(80, color="red", linestyle="--", lw=1, label="hard pity (trial 80)")
    ax.set_xlabel("trial number completed (after surviving k=1..69)")
    ax.set_ylabel("P(shiny by trial number | survived 1..69)")
    ax.set_title(
        "13 · 70-75 sprint: cumulative shiny prob conditional on surviving 69 trials "
        f"(p0={p0})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "13_pity_sprint_test.png", bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── narrative table ──────────────────────────────────


def write_table(scenarios, sp_sf_results, c_star, params):
    lines = [
        "# 03 · 经济策略仿真总表\n",
        "## 仿真参数",
        f"- $v_0$ (基础回顾值) = {params['v0']}  洛克贝-equiv",
        f"- $c_\\text{{ball}}$ (普通球成本) = {params['c_ball']}",
        f"- $c_\\text{{ball}}^s$ (好球成本) = {params['c_ball_s']}",
        f"- $q_\\text{{cap}} = q_\\text{{cap}}^s$ = {params['q_cap']}",
        f"- $m_\\text{{poll}} = 5$, $m_\\text{{shiny}} = 10$",
        "",
        "## 期望循环利润 (S-P vs S-F)",
        "",
        "| 场景 (p0, 机制) | $E[\\tau]$ | $E[\\Pi_{S-P}]$ | $E[\\Pi_{S-F}]$ | 差额 (S-F − S-P) |",
        "|---|---|---|---|---|",
    ]
    for (label, pk), (_, sp, sf) in zip(scenarios, sp_sf_results):
        et = expected_battles(pk)
        lines.append(
            f"| {label} | {et:.1f} | {sp:.1f} | {sf:.1f} | "
            f"**{sf - sp:+.1f}** |"
        )

    lines += [
        "",
        f"## 临界球成本 $c^*$ = $q \\cdot m_\\text{{poll}} \\cdot v_0$ ≈ **{c_star:.1f}**",
        "",
        "在 4.16 版本下普通球作物成本远低于该值, **S-F 全抓策略对所有 $p_0$ 一致显著最优**.",
        "策略选择对 $p_0$ 不敏感 — 不论真实概率是 0.018/0.062/0.092, 球成本只要 < 几十洛克贝, 全抓都赢. ",
        "",
        "## 非酋上限 (V2 §7)",
    ]
    for label, pk in scenarios:
        wc = worst_case_profit(pk, alpha_SF(), **params)
        lines.append(f"- {label}: 最差情况 (k=80 才出) 净利润 = **{wc:+.1f}**")

    lines += [
        "",
        "实际玩家最差情况净利润仍**显著为正** — 这是研究最有传播力的发现.",
        "",
        "## 70–75 冲刺大法 (V2 §5.3)",
        "见 `figures/13_pity_sprint_test.png`. ",
        "- M1 (硬保底, 无记忆): 70 → 71 …→ 79 步累计中奖概率几乎线性 (常数 $p_0$), 80 步突变到 1.",
        "- M2 (k*=78): 78 步前几乎线性, 79–80 突然飙升.",
        "- M3 (eta=19): 78 步前几乎平坦, 79–80 飙升.",
        "",
        "**实战意义**: 民间冲刺大法在三种机制下都没有显著好处 (相比从 k=1 重新开始), ",
        "因为只要 $k < 78$, $p_k \\approx p_0$. 真正起作用的是 80 那一步必出. ",
        "**冲刺只是心理安慰**.",
    ]

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


# ───────────────────────── main ─────────────────────────────────────────────


def main():
    # baseline params per V2 §8
    params = dict(
        v0=50.0,
        c_ball=5.0,        # 普通球作物成本
        c_ball_s=50.0,     # 炫彩/补光球
        q_cap=0.95,
        q_cap_s=0.95,
    )

    # scenarios across (p0, mechanism)
    # 0.018 is a player-derived per-trial estimate (NOT official); the
    # official anchor is the "comprehensive" rate ≈ 0.02, which corresponds
    # to per-trial p0 ≈ 0.0143 under M1 (since 1/E[tau] = comprehensive rate).
    scenarios = [
        ("p0=0.018 M1 (player-derived)", pk_M1(0.018)),
        ("p0=0.018 M2 (player-derived+soft)", pk_M2(0.018, 78)),
        ("p0=0.062 M1 (M2/M3 posterior)", pk_M1(0.062)),
        ("p0=0.062 M2 (M2/M3 posterior, soft)", pk_M2(0.062, 78)),
        ("p0=0.062 M3 (M2/M3 posterior, convex)", pk_M3(0.062, 19.3)),
        ("p0=0.092 M1 (M1 posterior)", pk_M1(0.092)),
    ]

    sp_sf = plot_strategy_profit(scenarios, params)
    c_star = plot_critical_ball_cost()
    plot_SC_threshold(p0=0.062)
    plot_pity_sprint_test(p0=0.062)
    write_table(scenarios, sp_sf, c_star, params)

    print("\n=== summary ===")
    for (label, pk), (_, sp, sf) in zip(scenarios, sp_sf):
        et = expected_battles(pk)
        print(f"  {label:50s} E[tau]={et:5.1f}  SP={sp:7.1f}  SF={sf:7.1f}  Δ={sf-sp:+7.1f}")


if __name__ == "__main__":
    main()
