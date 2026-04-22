# 02 · Bayesian fit summary (M1 / M2 / M3)

- Data: 266 cleaned single_pool reports (`pity_clean.jsonl`).
- Sampler: PyMC NUTS, 4 chains x 1500 draws (1500 tune).
- Likelihood is a Categorical over 80 outcomes whose probability mass is the per-model PMF reweighted by `w(k) = sigmoid(b0 + b1·1[k<30] + b2·1[k>=78])` (V2 §2.3 selection model).

## LOO comparison
```
    rank    elpd_loo     p_loo  elpd_diff        weight         se       dse  warning scale
M1     0 -789.254789  4.484910   0.000000  1.000000e+00  30.232740  0.000000    False   log
M2     1 -802.475546  4.499537  13.220756  1.094680e-13  29.637673  4.265273    False   log
M3     2 -819.706483  4.878252  30.451694  1.966205e-13  30.287810  8.921119    False   log
```

## Official benchmark tests
Note: 0.018 is **player-derived** per-trial estimate (NOT official).
The actual official anchor is the 'comprehensive' rate **0.02**, which under M1 corresponds to per-trial p0 ≈ 0.0143 (since 1/E[tau] equals the comprehensive rate).

| 模型 | $\hat p_0$ | $1/\hat E[\tau]$ (comprehensive rate) | $P(p_0 < 0.0143)$ | $P(p_0 < 0.018)$ |
|---|---|---|---|---|
| M1 | 0.0921 | 0.0922 | **0.00%** | **0.00%** |
| M2 | 0.0621 | 0.0625 | **0.00%** | **0.00%** |
| M3 | 0.0617 | 0.0624 | **0.00%** | **0.00%** |

*Interpretation*: 后验 $p_0$ 显著超出官方综合概率 0.02 蕴含的 0.0143, 也超出玩家反推的 0.018. 综合出货率被强烈拒绝.

## Posterior summaries

### M1
```
          mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p0       0.092  0.009   0.075    0.107      0.000    0.000     794.0     463.0    1.0
b0      -0.017  3.099  -5.948    5.792      0.063    0.054    2417.0    2395.0    1.0
b_first  0.657  0.394   0.000    1.313      0.014    0.012     828.0     800.0    1.0
b_low   -1.325  0.328  -1.896   -0.725      0.011    0.010     890.0     897.0    1.0
b_pity   3.564  0.588   2.390    4.579      0.018    0.012    1041.0     750.0    1.0
```

### M2
```
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p0        0.062  0.006   0.051    0.072      0.000    0.000    2576.0    2494.0    1.0
b0        0.031  2.014  -3.805    3.763      0.028    0.029    5063.0    3738.0    1.0
b_first   2.295  0.186   1.970    2.666      0.003    0.003    2919.0    3225.0    1.0
b_low     0.086  0.083   0.000    0.233      0.002    0.002    2506.0    2233.0    1.0
b_pity    2.305  0.540   1.273    3.306      0.010    0.008    2748.0    2424.0    1.0
k_star   78.177  0.275  77.734   78.732      0.004    0.005    4290.0    3097.0    1.0
```

### M3
```
           mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p0        0.062  0.005   0.052    0.071      0.000    0.000    3896.0    3952.0    1.0
b0        0.016  2.016  -3.723    3.844      0.027    0.029    5564.0    3894.0    1.0
b_first   2.309  0.181   1.982    2.654      0.003    0.002    4044.0    3816.0    1.0
b_low     0.092  0.089   0.000    0.257      0.001    0.002    3099.0    2482.0    1.0
b_pity    4.821  0.806   3.285    6.283      0.015    0.011    2972.0    3043.0    1.0
eta      19.333  3.118  14.074   25.304      0.055    0.054    3318.0    3188.0    1.0
```

## Figures
- `figures/05_posterior_p0.png` — posterior on per-trial base p0 (3 models)
- `figures/06_posterior_w_function.png` — posting-weight w(k) posterior
- `figures/07_implied_pmf_overlay.png` — implied weighted-PMF vs data
- `figures/08_model_comparison_loo.png` — LOO ranking