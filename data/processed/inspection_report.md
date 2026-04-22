# 01 · single_pool inspection report

- input file: `data/processed/pity_clean.jsonl`
- single_pool rows used (1<=n<=80): **266**

## Posting-bias signature
- n=1 reports: **97** (36.5%)
- n=80 reports (pity trigger): **10** (3.8%)
- low-k region (n<30): **220** (82.7%)
- mid region (30<=n<=77): **34** (12.8%)

**Interpretation.** The empirical histogram is U-shaped, exactly the double-peak posting bias predicted in V2 §1.3: shouty 'first-try' reports + 'finally hit pity' reports, with the middle eaten by selection. This validates the bias-correction weight 
`w(k) = sigma(b0 + b1·1[k<30] + b2·1[k>=78])` baked into M1/M2/M3.

## Sanity check vs official 'comprehensive 0.02'
- Empirical P(N<=10): **0.778**
- Geometric(p=0.018) P(N<=10): 0.166
- Empirical P(N=80): **0.038**
- Geometric(p=0.018) P(N=80): 0.0043

**Interpretation.** Empirical P(N<=10) is grossly inflated relative to a no-bias Geometric draw — quantitative confirmation of the early-k posting spike. The 80-spike is also far above Geometric tail mass, consistent with a hard pity AND with selective posting of pity-trigger events.

## Figures
- `figures/01_single_pool_hist.png` — n histogram (linear + log)
- `figures/02_posting_bias.png` — posting-bias regions overlaid
- `figures/03_empirical_vs_geometric.png` — empirical CDF vs Geometric(0.02 / 0.018)
- `figures/04_family_breakdown.png` — top-15 family mentions