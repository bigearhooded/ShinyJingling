"""Strategy simulator V2 — uses *real* 4.16 ball prices and reward scales
(see docs/08_economic_params.md).

Major revisions vs Stage 4:
    - Catch-ball price scale 1000x larger (15,000–80,000 RMB-equiv per ball)
    - v_0 (per-shiny lookup credit) ≈ 5,000 with [1000, 10000] sensitivity
    - Catch probability q_cap differs by ball type:
        normal       free, q ≈ 0.40
        attribute    free (mining), q ≈ 0.95 against typed
        high-grade   15000, q ≈ 0.70
        capture      80000, q = 1.00
    - Strategy choice now depends on (v_0, ball_type) — not just (p_0, mech)

Outputs:
    figures/17_strategy_v2_ball_grid.png      — heatmap profit(ball, p0)
    figures/18_v0_breakeven_curve.png         — break-even v_0 by ball
    figures/19_prism_option_value.png         — option value of prism upgrade
    figures/20_realistic_strategy_choice.png  — best strategy by (v_0, p_0)
    data/processed/strategy_v2_summary.md
"""
from __future__ import annotations

from pathlib import Path

from _plotting import setup as _setup_plotting; _setup_plotting()
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(".")
FIG = PROJ / "figures"
OUT = PROJ / "data/processed/strategy_v2_summary.md"

K_MAX = 80


# ───────────────────────── ball catalog ──────────────────────────────────────


BALLS = {
    "normal":   {"price": 0,      "q_cap": 0.40,  "label": "普通球 (免费)"},
    "attribute":{"price": 0,      "q_cap": 0.95,  "label": "属性球 (免费, 同属性)"},
    "high":     {"price": 15000,  "q_cap": 0.70,  "label": "高级球 (15k)"},
    "capture":  {"price": 80000,  "q_cap": 1.00,  "label": "捕光球 (80k, 100%)"},
}

CAPTURE_BALL = "capture"   # forced for shiny capture (don't risk losing it)


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


def survive(pk):
    out = np.empty(K_MAX, dtype=float)
    out[0] = 1.0
    out[1:] = np.cumprod(1.0 - pk[:-1])
    return out


def expected_battles(pk):
    return float(np.sum(survive(pk)))


def expected_profit(pk, alpha, v0, ball_for_pollution, m_poll=5.0, m_shiny=10.0):
    """Per-cycle expected profit using *separate* ball for pollution catches
    and a single capture-ball for the final shiny."""
    p_ball = BALLS[ball_for_pollution]
    s_ball = BALLS[CAPTURE_BALL]
    s = survive(pk)
    per_pollution = alpha * (p_ball["q_cap"] * m_poll * v0 - p_ball["price"])
    per_shiny = pk * (s_ball["q_cap"] * m_shiny * v0 - s_ball["price"])
    return float(np.sum((per_pollution + per_shiny) * s))


# ───────────────────────── plotting ─────────────────────────────────────────


def plot_strategy_grid(p0=0.062, mech="M1"):
    """Heatmap: net profit S-F as function of v_0 × pollution_ball."""
    pk = pk_M1(p0) if mech == "M1" else pk_M2(p0)
    v0_grid = np.array([1000, 2000, 3000, 5000, 7500, 10000])
    ball_keys = ["normal", "attribute", "high", "capture"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    profits = np.zeros((len(ball_keys), len(v0_grid)))
    for i, b in enumerate(ball_keys):
        for j, v0 in enumerate(v0_grid):
            profits[i, j] = expected_profit(
                pk, np.ones(K_MAX), v0, ball_for_pollution=b)
    im = ax.imshow(profits, aspect="auto", cmap="RdBu_r",
                    vmin=-2e6, vmax=2e6)
    ax.set_xticks(range(len(v0_grid)))
    ax.set_xticklabels([f"{v}" for v in v0_grid])
    ax.set_yticks(range(len(ball_keys)))
    ax.set_yticklabels([BALLS[b]["label"] for b in ball_keys])
    ax.set_xlabel("v0 (per-shiny lookup credit, 洛克贝)")
    ax.set_ylabel("ball used for pollution catches")
    ax.set_title(
        f"17 · S-F net profit per cycle (p0={p0}, {mech}); "
        f"shiny always caught with capture-ball (80k)"
    )
    plt.colorbar(im, ax=ax, label="洛克贝 / cycle (red=亏)")
    for i in range(profits.shape[0]):
        for j in range(profits.shape[1]):
            v = profits[i, j]
            ax.text(j, i, f"{v/1e3:+.0f}k", ha="center", va="center",
                    color="white" if abs(v) > 1e6 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "17_strategy_v2_ball_grid.png", bbox_inches="tight")
    plt.close(fig)
    return profits, v0_grid, ball_keys


def plot_v0_breakeven(p0=0.062):
    """For each ball type, what v_0 makes E[Pi_SF] = 0?"""
    pk = pk_M1(p0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    v0s = np.linspace(0, 12000, 200)
    for b in ["normal", "attribute", "high", "capture"]:
        ys = [expected_profit(pk, np.ones(K_MAX), v, b) / 1e3 for v in v0s]
        ax.plot(v0s, ys, lw=2, label=BALLS[b]["label"])
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("v0 (洛克贝 per shiny)")
    ax.set_ylabel("net profit per cycle  (千洛克贝)")
    ax.set_title(
        f"18 · break-even v0 by ball type (p0={p0}, M1, S-F all-catch)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "18_v0_breakeven_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_realistic_strategy(p0=0.062, mech="M1"):
    """For each (v_0, p_0) what's the BEST strategy (ball choice, S-P/S-F)?"""
    pk = pk_M1(p0) if mech == "M1" else pk_M2(p0)
    v0_grid = np.linspace(500, 12000, 30)
    p0_grid = np.linspace(0.018, 0.10, 30)
    best = np.zeros((len(p0_grid), len(v0_grid)), dtype=int)
    label_map = {0: "S-P", 1: "S-F normal", 2: "S-F attr", 3: "S-F high"}
    for j, p0v in enumerate(p0_grid):
        pk_v = pk_M1(p0v)
        alpha_F = np.ones(K_MAX); alpha_P = np.zeros(K_MAX)
        for i, v0 in enumerate(v0_grid):
            options = {
                0: expected_profit(pk_v, alpha_P, v0, "normal"),  # SP doesn't matter ball-wise
                1: expected_profit(pk_v, alpha_F, v0, "normal"),
                2: expected_profit(pk_v, alpha_F, v0, "attribute"),
                3: expected_profit(pk_v, alpha_F, v0, "high"),
            }
            best[j, i] = max(options, key=lambda k: options[k])
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("Set3", 4)
    im = ax.imshow(best, aspect="auto", origin="lower", cmap=cmap,
                    extent=[v0_grid.min(), v0_grid.max(),
                            p0_grid.min(), p0_grid.max()])
    cbar = plt.colorbar(im, ax=ax, ticks=range(4))
    cbar.set_ticklabels([label_map[i] for i in range(4)])
    ax.set_xlabel("v0  (per-shiny credit, 洛克贝)")
    ax.set_ylabel("p0  (per-trial base prob)")
    ax.axvline(80000 / (10 * 0.95), color="red", linestyle="--", lw=1,
               label="capture-ball break-even v0")
    ax.legend(loc="upper right")
    ax.set_title(f"20 · optimal strategy by (v0, p0) — {mech}")
    fig.tight_layout()
    fig.savefig(FIG / "20_realistic_strategy_choice.png", bbox_inches="tight")
    plt.close(fig)


def plot_prism_option_value(v0=5000):
    """Option value of holding shiny for prism upgrade.

    Simple model: prism upgrade gives shiny -> chromatic shiny with collection
    premium phi · v0 (phi multiplier of base lookup value).
    Option value = phi · v0 - prism_cost.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    phi_grid = np.linspace(0, 200, 200)  # collection multiplier
    prism_costs = {
        "通行证 (S1: 3 个)":         0,           # free if you do BP
        "远行商人 (~320万 洛克贝)":  3_200_000,
        "氪金 160 RMB":             3_200_000,    # equiv
    }
    for label, cost in prism_costs.items():
        ax.plot(phi_grid, phi_grid * v0 - cost, lw=2, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(80, color="orange", linestyle="--", lw=1,
               label=r"$\phi=80$ (collection 80×)")
    ax.set_xlabel(r"collection premium multiplier  $\phi$  (chromatic vs shiny)")
    ax.set_ylabel("option value  (洛克贝)")
    ax.set_title(
        f"19 · prism upgrade option value (v0={v0})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "19_prism_option_value.png", bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── narrative ────────────────────────────────────────


def write_summary():
    pk_post = pk_M1(0.062)   # use M2/M3 posterior median for default scenarios
    pk_off = pk_M1(0.018)    # player-derived
    et_post = expected_battles(pk_post)
    et_off = expected_battles(pk_off)

    lines = [
        "# 04v2 · Realistic strategy table (real ball prices)\n",
        "Ball catalog (洛克贝-equiv):",
        "",
        "| Ball | Price | $q_\\text{cap}$ |",
        "|---|---|---|",
    ]
    for k, b in BALLS.items():
        lines.append(f"| {b['label']} | {b['price']} | {b['q_cap']:.2f} |")

    lines += [
        "",
        f"## Key per-cycle profits  (v0=5000, capture-ball for shiny)",
        "",
        "Posterior $p_0=0.062$ (M2/M3), M1 mechanism, $E[\\tau]={:.1f}$:".format(
            et_post),
        "",
        "| Strategy / pollution ball | net profit / cycle (洛克贝) |",
        "|---|---|",
    ]
    for ball in ["normal", "attribute", "high", "capture"]:
        prof = expected_profit(pk_post, np.ones(K_MAX), 5000, ball)
        lines.append(f"| S-F + {BALLS[ball]['label']} | **{prof:+,.0f}** |")
    sp_prof = expected_profit(pk_post, np.zeros(K_MAX), 5000, "normal")
    lines.append(f"| S-P (纯保底) | **{sp_prof:+,.0f}** |")

    lines += [
        "",
        f"Player-guess $p_0=0.018$ (M1, $E[\\tau]={et_off:.1f}$):",
        "",
        "| Strategy / pollution ball | net profit / cycle |",
        "|---|---|",
    ]
    for ball in ["normal", "attribute", "high", "capture"]:
        prof = expected_profit(pk_off, np.ones(K_MAX), 5000, ball)
        lines.append(f"| S-F + {BALLS[ball]['label']} | **{prof:+,.0f}** |")
    sp_prof = expected_profit(pk_off, np.zeros(K_MAX), 5000, "normal")
    lines.append(f"| S-P (纯保底) | **{sp_prof:+,.0f}** |")

    lines += [
        "",
        "## Reversal vs Stage 4 (toy v0=50, c_ball=5):",
        "- 之前结论 'S-F 全抓必赢' 在 *任意球价* 下成立, 因为 v0/c_ball 比例偏离现实.",
        "- 真实经济下:",
        "  - **捕光球抓污染严重亏损** (8w 球 vs 25k 单池利润, 净 -55k/只)",
        "  - **属性球 (免费, q=0.95) 是最优污染抓取选择**",
        "  - **高级球 (1.5w, q=0.7) 边际可行**",
        "  - **普通球 (免费, q=0.4) 也优于 S-P 纯保底**",
        "  - **S-P 纯保底亏损** (异色用 80k 捕光球, 但单只回顾仅 50k = -30k)",
        "",
        "## Prism upgrade option (棱镜球升格炫彩异色)",
        "棱镜球纯洛克贝兑换 320 万 ⇒ 1000+ 天纯肝, 实际只能 *氪金 160 RMB*",
        "或 *通行证奖励 3 个/赛季*. ",
        "升格收益依赖炫彩异色相对异色的 collection premium $\\phi$.",
        "",
        "若 $\\phi v_0 > $ 棱镜球获取成本, 升格有正期望. ",
        "对 v_0 = 5000:",
        "- 通行证棱镜球 (相当 free): 任何 $\\phi > 0$ 都正期望",
        "- 远行商人/氪金 320万 等价: 需要 $\\phi > 640$ 才回本",
        "- 即炫彩异色需要比异色再值 640× 才回本 — 实际很难, 主要是 *status good*",
        "",
        "## 90% reversal: 之前 Stage 4 结论需要修正",
        "1. ❌ '$c^* = 5 v_0$, S-F 永远赢' → ✅ '$c_\\text{ball} < q \\cdot m_\\text{poll} \\cdot v_0$ 才有利, 球选择关键'",
        "2. ❌ '非酋上限 +19000 洛克贝' (用 v_0=50, c=5) → ✅ 真实非酋下限 用属性球 +1.87M, 用高级球 +0.17M, 用捕光球 -4.4M",
        "3. ✅ 多池均分仍优于集中 (与球选择无关)",
        "4. ✅ 70-75 冲刺无效 (M1 supports)",
        "5. ✅ 综合概率 0.02 被强烈拒绝 (与经济无关, Bayesian 结论稳健)",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


def main():
    plot_strategy_grid()
    plot_v0_breakeven()
    plot_realistic_strategy()
    plot_prism_option_value()
    write_summary()


if __name__ == "__main__":
    main()
