# Formative 3 - Probability Distributions, Bayesian Probability & Gradient Descent

## Part 1 - Probability Distributions: EM Algorithm (mixture of two Gaussians)

**Setup.** We chose **Fathers and Children**: 934 children's heights + 205 fathers' heights (one per family; the raw file repeats each father on every child's row). 1139 heights, labels hidden from the algorithm. Component 1 = Children, component 2 = Fathers (the brief's "Pros/Basketball Player" label comes from a template; here it is Fathers).

**Algorithm** (`notebooks/Part1_Probability_Distributions_EM_Algorithm.ipynb`)
1. **Initialization** - split at the global mean (67.21 in); each pile gives a starting mean, variance and mixing coefficient.
2. **E-step** - responsibility r = πₖ·N(x | μₖ, σₖ²) / Σⱼ πⱼ·N(x | μⱼ, σⱼ²): the probability each height belongs to each group (soft assignment).
3. **M-step** - new μₖ, σₖ², πₖ as responsibility-weighted averages.
4. **Stop** when the log-likelihood improves by less than 10⁻⁶ → converged after **309 iterations**.

**Tracking table** (printed by the code)

| Iteration | μ1 (Children) | μ2 (Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 (Initialization) | 64.2598 | 70.2147 | 4.1428 | 3.5711 | 0.5048 | 0.4952 | −3071.1165 |
| 1 | 64.3784 | 70.0758 | 4.9168 | 4.2979 | 0.5033 | 0.4967 | −3053.5777 |
| 2 | 64.4368 | 69.9988 | 5.3058 | 4.6724 | 0.5017 | 0.4983 | −3049.7088 |
| 309 (final) | 64.1772 | 69.5186 | 5.3024 | 6.0410 | 0.4325 | 0.5675 | −3046.9895 |

**Should we split at the global mean and average each pile? No.**
- The cut misplaces **44 of 205 fathers** and **403 of 934 children**: the two bell curves overlap, so no single line separates them.
- Cutting chops off tails, so the piles look too narrow (sd 2.04 / 1.89 vs EM 2.30 / 2.46).
- The global mean is the wrong boundary: EM's 50/50 point is **66.54 in**, not 67.21.
- Hard labels hide uncertainty: 66.71 in is "100% child" for the cut, but **46.1% / 53.9%** for EM.
- EM starts from that split and improves the log-likelihood from −3071.12 to −3046.99.

**What EM found** (hidden labels used only to check): the Children component is 92% children (mostly daughters); the Fathers component holds 81% of the fathers but also most sons - Galton's children are adults, and sons (69.2 in) are as tall as fathers (69.3 in).

**Test-height demo:** type the coach's value into `TEST_HEIGHT` (last cell of the Part 1 notebook) and run it, e.g. 70 in → P(Children) = 0.0328, P(Fathers) = 0.9672.

![EM fit](images/part1_em_fit.png)
