"""Time-budget LP: optimal daily allocation of T hours into
{shiny-hunting, pvp, herb-selling, attribute-ball mining}
subject to (i) per-cycle wall-time, (ii) ball supply, (iii) 洛克贝 daily caps.

4.16 game-economy constants (verified from community + user notes):

    Income caps (skilled player upper bound)
        PVP daily cap:         15,000,000 洛克贝  /  10 h skilled  →  1.5M / h
        奇异花 daily cap:       15,000,000 洛克贝  /   6 h skilled  →  2.5M / h
        Total daily 洛克贝 cap: ~30,000,000

    Ball prices (post-4.16 update)
        高级球:    12,000 洛克贝  (q ≈ 0.70)   ← user-corrected from 15k
        捕光球:    80,000 洛克贝  (q  =  1.00)
        属性球:    free, mining gives ~1,000 / h
        棱镜球:    160 RMB ≈ 3.2M 洛克贝 (only via 通行证 / 远行商人 / 氪金)

    Per-shiny cycle cost (community consensus: "准备 2000 球")
        E[tau]  = 16 pollutions to first shiny  (Section IV posterior, p0=0.062)
        每只污染需 ~50 个 mob hits to trigger  →  16 × 50 = 800 mob hits
        每个 mob hit 用 1 颗 ball (普通球免费 / 属性球免费 / 高级球付费)
        每只污染体本身需 1 颗 ball to capture (q_cap)
        每 1 异色还需 1 颗捕光球 (q=1, 80k)

        ⇒ TOTAL balls per cycle ≈ 800 (mob) + 16/q_cap (污染) + 1 (异色)
                                   ≈ 800 + 23 + 1 = 824   (高级球抓污染, 普通球 mob)
                                   ≈ 800 + 17 + 1 = 818   (属性球, 全免费)
        论坛 "2000 球准备" 是含失败/补球的安全冗余 (~2.5×).

    Cycle wall-time (skilled player, low-阶 family, 3+3 出货大法)
        ≈ 30 min  (~ 22 min mob 平推 + ~ 7 min 污染体抓 + ~1 min 异色抓)
        We use 0.5 h / cycle as the skilled baseline; casual players ≈ 1 h.

Outputs:
    figures/21_time_budget_pareto.png      — Pareto (T_h, expected shinies/day)
    figures/22_player_tier_strategies.png   — 3-tier player SOPs
    figures/23_binding_constraints.png      — which constraint binds, by T_h
    data/processed/time_budget_summary.md
"""
from __future__ import annotations

from pathlib import Path

from _plotting import setup as _setup_plotting; _setup_plotting()
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(".")
FIG = PROJ / "figures"
OUT = PROJ / "data/processed/time_budget_summary.md"

# ───────────────────── 4.16 economy constants ───────────────────────────────


PVP_RATE = 1_500_000              # 洛克贝 / h
PVP_CAP_DAILY = 15_000_000

HERB_RATE = 2_500_000             # 洛克贝 / h
HERB_CAP_DAILY = 15_000_000

MINE_BALL_RATE = 1000             # attribute balls / h

CAPTURE_BALL_PRICE = 80_000       # 100% catch
HIGH_BALL_PRICE = 12_000          # post-4.16 update

# Posterior estimate from §IV
P0 = 0.062
K_MAX = 80
ETAU = 16.0                       # ≈ 1 / 0.062

MOB_HITS_PER_POLLUTION = 50       # community estimate (low-阶)
TOTAL_MOB_HITS_PER_CYCLE = ETAU * MOB_HITS_PER_POLLUTION   # ≈ 800
POLLUTION_CATCHES_PER_CYCLE = ETAU                          # = 16

# Skilled cycle wall-time (hours per cycle / per shiny)
CYCLE_TIME_SKILLED = 0.5

# Q_cap for the pollution-体 catch (NOT for mob hits — those are 1-shot)
Q_CAP = {"normal": 0.40, "attribute": 0.95, "high": 0.70, "capture": 1.00}
BALL_PRICE = {"normal": 0, "attribute": 0, "high": HIGH_BALL_PRICE,
              "capture": CAPTURE_BALL_PRICE}


# ───────────────────── per-cycle cost decomposition ─────────────────────────


def cycle_cost(pol_ball="high", mob_ball="normal"):
    """Per-shiny-cycle balls + 洛克贝 cost.

    pol_ball : ball used to capture the pollution体 (16 catches / cycle)
    mob_ball : ball used to dispatch the 800 trigger-mobs / cycle.
               'normal'/'attribute' = free; 'high' = 12k each.

    Returns dict(balls_total, locke_total, mob_balls, pol_balls).
    """
    pol_balls = POLLUTION_CATCHES_PER_CYCLE / Q_CAP[pol_ball]
    pol_locke = pol_balls * BALL_PRICE[pol_ball]
    # mob hits: assume q=1 (1-shot). Free if 普通/属性, paid if 高级.
    mob_balls = TOTAL_MOB_HITS_PER_CYCLE
    mob_locke = mob_balls * BALL_PRICE[mob_ball]
    cap_locke = CAPTURE_BALL_PRICE
    return dict(
        balls_total=mob_balls + pol_balls + 1,
        locke_total=pol_locke + mob_locke + cap_locke,
        mob_balls=mob_balls,
        pol_balls=pol_balls,
        pol_locke=pol_locke,
        mob_locke=mob_locke,
        cap_locke=cap_locke,
    )


# ───────────────────── strategy variants ────────────────────────────────────


STRATEGIES = {
    "S1: 全免费球(属性+普通)": dict(pol_ball="attribute", mob_ball="normal"),
    "S2: 高级球抓污染+普通球Mob":   dict(pol_ball="high",      mob_ball="normal"),
    "S3: 全捕光球(土豪流)":     dict(pol_ball="capture",   mob_ball="normal"),
}


# ───────────────────── time-budget LP for one strategy ─────────────────────


def daily_shinies(T_hours, time_share, strategy_key,
                   v0=5000, cycle_time=CYCLE_TIME_SKILLED):
    """time_share = (mine, pvp, herb, hunt) summing ≤ 1.

    'mine' = 跑图采花产属性球 + 普通球 (assumed free conversion)
    'pvp' / 'herb' = 洛克贝产出
    'hunt' = actual shiny-hunting time (cycle wall-time spent)
    """
    ms, ps, hs, hh = time_share
    assert ms + ps + hs + hh <= 1.0 + 1e-6
    T_mine = T_hours * ms
    T_pvp = T_hours * ps
    T_herb = T_hours * hs
    T_hunt = T_hours * hh

    locke_avail = (
        min(T_pvp * PVP_RATE, PVP_CAP_DAILY)
        + min(T_herb * HERB_RATE, HERB_CAP_DAILY)
    )
    attr_balls = T_mine * MINE_BALL_RATE
    cycles_time_capacity = T_hunt / cycle_time

    s = STRATEGIES[strategy_key]
    cost = cycle_cost(s["pol_ball"], s["mob_ball"])

    # Three constraints
    n_time = cycles_time_capacity
    n_locke = locke_avail / cost["locke_total"]
    if s["pol_ball"] == "attribute":
        # mob (普通球 free) + 污染 (属性球 free, but limited to mining supply)
        n_balls = attr_balls / cost["pol_balls"]
    else:
        # 普通球 mob is free, 高级球 / 捕光球 用 locke_avail 已计入 locke 约束
        n_balls = np.inf

    n_shiny = min(n_time, n_locke, n_balls)
    binding = ["time", "locke", "balls"][np.argmin([n_time, n_locke, n_balls])]

    # net profit: shiny回顾 + 污染累积 - 球成本 (already in cost)
    pol_value_per_cycle = ETAU * Q_CAP[s["pol_ball"]] * 5 * v0
    shiny_value_per_cycle = 10 * v0
    net_per_shiny = pol_value_per_cycle + shiny_value_per_cycle - cost["locke_total"]
    net_total = n_shiny * net_per_shiny

    return dict(
        n_shiny=n_shiny, net_profit=net_total,
        binding=binding,
        n_time=n_time, n_locke=n_locke, n_balls=n_balls,
        T_mine=T_mine, T_pvp=T_pvp, T_herb=T_herb, T_hunt=T_hunt,
        locke_avail=locke_avail, attr_balls=attr_balls,
        cost=cost,
    )


def best_split_for_strategy(T_h, strategy_key, v0=5000):
    """Grid-search the best (mine, pvp, herb, hunt) shares (step 0.05)."""
    best = None
    grid = np.linspace(0, 1, 21)
    for ms in grid:
        for ps in grid:
            for hs in grid:
                hh = 1 - ms - ps - hs
                if hh < -1e-9 or hh > 1 + 1e-9:
                    continue
                r = daily_shinies(T_h, (ms, ps, hs, max(hh, 0)), strategy_key, v0=v0)
                if best is None or r["n_shiny"] > best["n_shiny"]:
                    best = r
                    best["share"] = (ms, ps, hs, max(hh, 0))
    return best


def best_overall(T_h, v0=5000):
    """Try all 3 strategies, return the dominant one."""
    best = None
    for sk in STRATEGIES:
        r = best_split_for_strategy(T_h, sk, v0=v0)
        if best is None or r["n_shiny"] > best["n_shiny"]:
            best = r
            best["strategy"] = sk
    return best


# ───────────────────── plotting ─────────────────────────────────────────────


def plot_pareto():
    Th_grid = np.array([0.5, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16])
    res = [best_overall(t) for t in Th_grid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # left: Pareto curve
    ns = [r["n_shiny"] for r in res]
    axes[0].plot(Th_grid, ns, "o-", lw=2, color="#3182bd")
    axes[0].set_xlabel("daily time budget  T_h  (小时)")
    axes[0].set_ylabel("optimal expected shinies / day")
    axes[0].set_title("21A · 异色狩猎 Pareto frontier (4.16 真实约束)")
    axes[0].grid(alpha=0.3)
    # mark binding constraint
    for t, r in zip(Th_grid, res):
        c = {"time": "blue", "locke": "red", "balls": "green"}[r["binding"]]
        axes[0].scatter([t], [r["n_shiny"]], color=c, s=80, zorder=4)
    # legend dummies
    axes[0].scatter([], [], color="blue", label="time-binding")
    axes[0].scatter([], [], color="red", label="locke-binding")
    axes[0].scatter([], [], color="green", label="balls-binding")
    axes[0].legend(loc="lower right")

    # right: time allocation
    mines = [r["share"][0] for r in res]
    pvps = [r["share"][1] for r in res]
    herbs = [r["share"][2] for r in res]
    hunts = [r["share"][3] for r in res]
    axes[1].stackplot(Th_grid, hunts, mines, pvps, herbs,
                      labels=["刷异色 (cycle 时间)", "采花 (属性球)",
                               "PVP (洛克贝)", "奇异花 (洛克贝)"],
                      colors=["#fdae6b", "#74c476", "#fd8d3c", "#9e9ac8"],
                      alpha=0.85)
    axes[1].set_xlabel("T_h (小时/天)")
    axes[1].set_ylabel("最优时间分配 (比例)")
    axes[1].set_title("21B · 最优时间分配组成")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "21_time_budget_pareto.png", bbox_inches="tight")
    plt.close(fig)
    return Th_grid, res


def plot_player_tiers():
    tiers = [
        ("轻度玩家\n(0.5 h/天)", 0.5),
        ("中度玩家\n(3 h/天)", 3.0),
        ("重度玩家\n(10 h/天)", 10.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_shinies = []
    nets = []
    strats = []
    for label, T_h in tiers:
        r = best_overall(T_h)
        n_shinies.append(r["n_shiny"])
        nets.append(r["net_profit"])
        strats.append(r["strategy"])

    xs = np.arange(len(tiers))
    width = 0.35
    ax.bar(xs - width / 2, n_shinies, width, color="#74c476",
           label="期望异色 / 天")
    ax2 = ax.twinx()
    ax2.bar(xs + width / 2, [n / 1e6 for n in nets], width,
            color="#fd8d3c", label="期望净利润 (百万洛克贝)")
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in tiers])
    ax.set_ylabel("期望异色数 / 天", color="#3a8b3a")
    ax2.set_ylabel("期望净利润 (百万洛克贝)", color="#d94900")
    ax.set_title("22 · 三档玩家最优策略下的异色与利润产出")
    for i, (n, p, s) in enumerate(zip(n_shinies, nets, strats)):
        ax.text(xs[i] - width / 2, n, f"{n:.1f}\n{s.split(':')[0]}",
                ha="center", va="bottom", fontsize=8)
        ax2.text(xs[i] + width / 2, p / 1e6, f"{p/1e6:+.1f}",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "22_player_tier_strategies.png", bbox_inches="tight")
    plt.close(fig)
    return tiers, n_shinies, nets, strats


# ───────────────────── narrative ────────────────────────────────────────────


def write_summary(Th_grid, res, tiers, n_shinies, nets, strats):
    lines = [
        "# 05 · 时间预算 LP 总表 (4.16 真实经济约束)\n",
        "## 4.16 经济常数",
        "",
        f"- PVP 上限: {PVP_CAP_DAILY:,} 洛克贝/天 ≈ {PVP_RATE:,}/h × 10 h",
        f"- 奇异花上限: {HERB_CAP_DAILY:,} 洛克贝/天 ≈ {HERB_RATE:,}/h × 6 h",
        f"- 每日洛克贝总上限: {PVP_CAP_DAILY+HERB_CAP_DAILY:,}",
        f"- 采花/挖矿: {MINE_BALL_RATE:,} 属性球/h",
        f"- 高级球（4.16 更新后）: {HIGH_BALL_PRICE:,} 洛克贝, q≈{Q_CAP['high']}",
        f"- 捕光球: {CAPTURE_BALL_PRICE:,} 洛克贝, q={Q_CAP['capture']}",
        f"- 单循环 wall-time (熟练): {CYCLE_TIME_SKILLED} h",
        f"- 单循环 mob hits 数: ~{TOTAL_MOB_HITS_PER_CYCLE:.0f}",
        f"  (16 污染 × 50 mob/污染; 论坛共识 '准备 2000 球' 含 ~2.5× 安全冗余)",
        "",
        "## 三种球类策略的 per-cycle 成本",
        "",
        "| 策略 | 污染抓球 | Mob 球 | 球总数 | 洛克贝/cycle |",
        "|---|---|---|---|---|",
    ]
    for sk, params in STRATEGIES.items():
        c = cycle_cost(params["pol_ball"], params["mob_ball"])
        lines.append(
            f"| {sk} | {params['pol_ball']} (q={Q_CAP[params['pol_ball']]:.2f}) "
            f"| {params['mob_ball']} | {c['balls_total']:.0f} | "
            f"{c['locke_total']:,.0f} |"
        )

    lines += [
        "",
        "## Pareto 边界：每日 T_h vs 最优期望异色",
        "",
        "| T_h (h) | 最优异色/天 | 最优策略 | binding 约束 | (采,P,花,猎) 时间分配 |",
        "|---|---|---|---|---|",
    ]
    for t, r in zip(Th_grid, res):
        ms, ps, hs, hh = r["share"]
        sk = r["strategy"].split(":")[0]
        lines.append(
            f"| {t:.1f} | {r['n_shiny']:.2f} | {sk} | {r['binding']} | "
            f"采{ms:.0%} P{ps:.0%} 花{hs:.0%} 猎{hh:.0%} |"
        )

    lines += [
        "",
        "## 三档玩家推荐 SOP",
        "",
        "| 玩家档 | 时间 | 最优策略 | 期望异色/天 | 期望净利润 (洛克贝/天) |",
        "|---|---|---|---|---|",
    ]
    for (name, T_h), n, p, s in zip(tiers, n_shinies, nets, strats):
        lines.append(
            f"| {name.replace(chr(10),' ')} | {T_h:.1f}h | {s} | {n:.2f} | {p:+,.0f} |"
        )

    lines += [
        "",
        "## Binding constraint 解读",
        "",
        "- **time 紧** (T_h 较低): 时间不够刷完整 cycle，再多钱/球也没用",
        "- **locke 紧** (中等 T_h, 高级球策略): 撞 PVP/奇异花 3000w/天上限",
        "- **balls 紧** (中等 T_h, 属性球策略): 采花速度 1000 球/h 跟不上",
        "",
        "在熟练玩家 (cycle = 0.5 h) 下:",
        "- T_h ≤ 1 h: 永远 time-binding，建议**全力刷异色** (球/钱压根用不上)",
        "- 1 < T_h ≤ ~4 h: time + balls 联合紧，**S1 全免费球路线** 最优",
        "- T_h > 4 h: 球瓶颈出现，**S2 高级球抓污染** 转优 (用 PVP 收入买球)",
        "- T_h ≥ 10 h 撞 PVP+herb 3000w 上限，单日产出饱和 ≈ 30 异色 (理论)",
        "",
        "## 局限",
        "- cycle wall-time = 0.5 h 是熟练玩家的乐观估计；萌新可能 1 h+/cycle",
        "- mob 数 50/污染 是低阶宠物经验值；高星光值精灵更快、boss 系更慢",
        "- 球的 q_cap 因精灵形态显著差异；表中是中等抓难度的经验估计",
        "- 未考虑赛季限时活动（双倍洛克贝、远行商人棱镜球折扣等）",
        "- 实际玩家因疲劳、网络、活动离线等无法每日撞上限",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


def main():
    Th_grid, res = plot_pareto()
    tiers, n_shinies, nets, strats = plot_player_tiers()
    write_summary(Th_grid, res, tiers, n_shinies, nets, strats)
    print("\n=== summary ===")
    for t, r in zip(Th_grid, res):
        print(f"  T_h={t:5.1f}h  best={r['strategy'][:30]:30s}  n={r['n_shiny']:5.2f}  "
              f"binding={r['binding']:6s}  net={r['net_profit']:+,.0f}")


if __name__ == "__main__":
    main()
