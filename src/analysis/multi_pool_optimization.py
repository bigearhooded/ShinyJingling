"""Multi-pool allocation optimization (V2 §5).

Question: given a daily budget of T total battles and N parallel pools,
how should we allocate T_n battles per pool to maximize expected
shinies (or expected profit)?

Two regimes:
    A. Fresh start (all pools at k=0):    expected shinies E_n(T_n) is concave
                                           ⇒ even split is optimal.
    B. Heterogeneous progress (k_n>0):   prefer pools closer to 80 (greedy
                                           on marginal shiny per battle).

Outputs:
    figures/14_marginal_shiny_per_battle.png  — marginal value curve
    figures/15_optimal_allocation_T100.png    — optimal split for T=100, N=4
    figures/16_concentrate_vs_split.png       — compare (T,1) vs (T/N,N)
    data/processed/multi_pool_summary.md
"""
from __future__ import annotations

from pathlib import Path

from _plotting import setup as _setup_plotting; _setup_plotting()
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(".")
FIG = PROJ / "figures"
OUT = PROJ / "data/processed/multi_pool_summary.md"

K_MAX = 80


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


def expected_shinies_one_pool(pk, T, k0=0):
    """Expected shinies from running T battles on a pool currently at k=k0
    (i.e., k0 battles already done without shiny).

    A pool can only fire one shiny per cycle; after a shiny the player
    typically resets the pollution counter (the family is captured). So
    we model: after a shiny, the remaining (T - cycle_len) battles can
    start a NEW pool from k=0 — but for *one specific pool* we cap shinies at 1.

    For the per-pool budget allocation, treat E_n(T) = P(shiny within T
    battles starting from k=k0).
    """
    # P(no shiny in next t battles starting from k0)
    if T == 0:
        return 0.0
    cond_pks = pk[k0:k0 + T]
    if len(cond_pks) == 0:
        return 0.0
    no_shiny = np.prod(1 - cond_pks)
    return 1.0 - no_shiny


def optimal_allocation(pk, T_total, k0_list, debug=False):
    """Greedy allocation: at each marginal battle, give it to the pool with
    the highest marginal shiny gain."""
    N = len(k0_list)
    allocations = [0] * N
    state_k = list(k0_list)
    while sum(allocations) < T_total:
        # marginal gain at next battle for each pool
        gains = []
        for n in range(N):
            k = state_k[n]
            if k >= K_MAX:
                gains.append(-np.inf)
            else:
                # P(shiny on next battle | survived to k) = pk[k]
                # but greedy on marginal of cumulative. Approximate:
                # gain ≈ pk[k] · prob_no_shiny_so_far
                # Since each pool reset on shiny, and we treat each pool
                # as 'one shot' until shiny, the marginal P(shiny on next
                # battle) starting from k is just pk[k] · already_no_shiny_prob.
                gains.append(pk[k])
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            break
        allocations[best] += 1
        state_k[best] += 1
    return allocations


def plot_marginal_shiny(p0=0.062, mechanism="M2"):
    """Show why pity-near pools are preferred: marginal shiny prob per
    battle as function of current k."""
    pk = pk_M2(p0) if mechanism == "M2" else pk_M1(p0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, K_MAX + 1), pk, lw=2, color="#1f77b4")
    ax.axvline(78, color="orange", linestyle="--", lw=1, label="k*=78 (M2 ramp start)")
    ax.axvline(80, color="red", linestyle="--", lw=1, label="hard pity")
    ax.set_xlabel("k  (current pollution count on pool)")
    ax.set_ylabel(r"$p_k$  (per-battle shiny prob)")
    ax.set_title(f"14 · marginal shiny prob per battle (p0={p0}, {mechanism})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "14_marginal_shiny_per_battle.png", bbox_inches="tight")
    plt.close(fig)


def plot_optimal_allocation(p0=0.062, T_total=100, k0_list=(0, 0, 30, 70)):
    pk = pk_M2(p0)
    alloc = optimal_allocation(pk, T_total, k0_list)
    fig, ax = plt.subplots(figsize=(8, 4))
    pool_names = [f"pool{i+1} k0={k}" for i, k in enumerate(k0_list)]
    bars = ax.bar(pool_names, alloc, color=["#9ecae1", "#9ecae1", "#fdae6b", "#e6550d"])
    for bar, v in zip(bars, alloc):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                ha="center", va="bottom")
    ax.set_ylabel("# battles allocated")
    ax.set_title(
        f"15 · optimal allocation (greedy on marginal p_k) — "
        f"T={T_total}, p0={p0}, M2"
    )
    fig.tight_layout()
    fig.savefig(FIG / "15_optimal_allocation_T100.png", bbox_inches="tight")
    plt.close(fig)
    return alloc


def plot_concentrate_vs_split(p0=0.062, T_total=160, N_grid=(1, 2, 4, 8, 16)):
    """For fresh-start pools, compare concentrating T into 1 pool vs splitting
    evenly into N pools."""
    pk = pk_M2(p0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mech_name, pk_v in [("M1", pk_M1(p0)), ("M2 (k*=78)", pk_M2(p0))]:
        e_vals = []
        for N in N_grid:
            T_per = T_total // N
            e_vals.append(N * expected_shinies_one_pool(pk_v, T_per))
        ax.plot(N_grid, e_vals, marker="o", lw=2, label=mech_name)
    ax.set_xscale("log")
    ax.set_xlabel("N  (number of parallel pools, fresh start k=0)")
    ax.set_ylabel("E[total shinies]")
    ax.set_title(
        f"16 · concentrate vs split — T={T_total} battles split N ways "
        f"(p0={p0})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "16_concentrate_vs_split.png", bbox_inches="tight")
    plt.close(fig)


def main():
    plot_marginal_shiny()
    alloc = plot_optimal_allocation()
    plot_concentrate_vs_split()

    # narrative
    pk = pk_M2(0.062)
    # E[shinies] for fresh-start pool at various T
    T_grid = [10, 20, 40, 80, 160]
    row = [(T, expected_shinies_one_pool(pk, T)) for T in T_grid]

    lines = [
        "# 04 · 多池并行优化总表 (§5)\n",
        "## 单池期望异色 E_n(T)  (M2, p0=0.062, fresh start)",
        "",
        "| T (battles) | E[shinies] |",
        "|---|---|",
    ]
    for T, e in row:
        lines.append(f"| {T} | {e:.3f} |")

    lines += [
        "",
        "## 关键结论 1：均分多池 vs 集中单池",
        "",
        "在 fresh start (所有池 k=0) 下, $E_n(T) = 1 - (1-p_0)^T$ 是**凹**的,",
        "因此对 N 个独立同质池均分 $T$ 总预算 (每池 $T/N$):",
        "$$",
        "\\text{total shinies} = N \\cdot E(T/N)",
        "$$",
        "",
        "- 当 $T \\ll 80$ (球成本/作物 hard cap), 增大 $N$ 几乎线性增加产出",
        "- 当 $T \\gg 80$, 增大 $N$ 边际收益递减 (单池已逼近 1)",
        "",
        "见 `figures/16_concentrate_vs_split.png`. ",
        "",
        "## 关键结论 2：异质池下的贪心策略",
        "",
        f"测试 4 池, k0 = (0, 0, 30, 70), T={100}, p0=0.062, M2.",
        f"贪心分配 (每步选边际 $p_k$ 最大的池) 结果: {alloc}",
        "",
        "**直觉**: 接近保底 (k=70) 的池每步出货概率 = $p_0$ = 0.062 (在 M2 下 k<78 平坦),",
        "新开池 (k=0) 也是 0.062, 所以在 M2 下贪心**不偏向冲保底** — 唯一例外是冲到 78 后",
        "$p_k$ 飙升, 那时所有 budget 都给即将爆的池. 这与民间'冲保底'直觉一致.",
        "",
        "## 关键结论 3：'70–75 冲刺大法'被证伪",
        "",
        "见 `figures/13_pity_sprint_test.png`:",
        "- M1: 70-79 步线性, 80 步必出",
        "- M2 (k*=78): 78 步前线性, 79-80 飙升",
        "- M3 (eta=19): 70-75 区间累计 ~50% 出货 (略好于 M1, 因为凸曲线已经开始上升)",
        "",
        "数据偏好 M1/M2, 即 70-75 冲刺并不显著比 1-5 重新开始好",
        "(因为 $p_{70} \\approx p_1 \\approx p_0$). **冲刺主要是心理安慰**.",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
