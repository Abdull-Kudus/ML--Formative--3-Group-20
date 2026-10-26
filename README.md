# Formative 3 - Probability Distributions, Bayesian Probability & Gradient Descent

ML Formative 3 - Group 20.

## Team

| Member | Role | Responsible for |
|---|---|---|
| Abdul Kudus Zakaria Mukhtaru | Team Lead - project management, Git branch oversight, code review, mathematical validation, documentation | Part 3 and Part 4 - Gradient Descent |

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # needs SciPy >= 1.15 for scipy.differentiate
jupyter notebook                   # open a notebook in notebooks/ and choose Run All
```

The notebooks also run in Google Colab: the first code cell copies this repository into Colab.

---

## Part 3 - Gradient Descent Manual Calculation

Handwritten, one full iteration per member: `docs/Part3_Gradient_Descent_Manual_Calculation_Handwritten.pdf`.

| Pages | Written by | Content |
|---|---|---|
| 1-2 | Abdul Kudus Zakaria Mukhtaru | Setup, chain-rule derivation, Iteration 1 |
| 3 | Nitonde Binthia | Iteration 2 |
| 4 | Bior Aguer Kuir | Iteration 3 |
| 5 | Abdul Kudus Zakaria Mukhtaru | Summary table and trend |

**Model in matrix form:** ŷ = Xm + b with X = [[1, 3], [4, 10]], y = [5, 6], initial m = [−1, 2], b = [1, 1] (one shared bias, written once per data point).
**Cost:** J = (1/n) Σ (ŷ − y)² (Mean Squared Error), n = 2.
**Chain rule:** ∂J/∂m = ∂J/∂ŷ · ∂ŷ/∂m = (2/n) Xᵀ(ŷ − y) and ∂J/∂b = (2/n) Σ(ŷ − y).
**Update rule:** new value = old value − α × gradient, with learning rate α = 0.01 (not given in the brief; stable because α < 2/127.5 ≈ 0.0157, while α = 0.1 makes J jump from 61 to 8089.7).

| Iteration | By | m at start | b at start | J | ∂J/∂m | ∂J/∂b | m result | b result |
|---|---|---|---|---|---|---|---|---|
| 1 | Kudus | [−1, 2] | 1 | 61 | [45, 113] | 12 | [−1.45, 0.87] | 0.88 |
| 2 | Binthia | [−1.45, 0.87] | 0.88 | 6.845 | [−11.84, −31.08] | −5.18 | [−1.3316, 1.1808] | 0.9318 |
| 3 | Bior | [−1.3316, 1.1808] | 0.9318 | 2.72381716 | [3.7962, 8.5618] | −0.444 | [−1.369562, 1.095182] | 0.93624 |

**Trend.** J falls every iteration (61 → 6.845 → 2.7238 → 2.3904, a 96% drop), so m and b move in the direction that reduces the error. Each value moves opposite to its gradient; steps shrink as the error shrinks; m₂ zig-zags (2 → 0.87 → 1.1808 → 1.0952) because x₂ = 10 makes the cost steep along m₂, but each overshoot is smaller than the last.

## Part 4 - Gradient Descent in Code

Notebook: `notebooks/Part4_Gradient_Descent_in_Code.ipynb`

- `derivative_of(equation, point)` passes any one-variable equation to `scipy.differentiate.derivative` (SciPy ≥ 1.15). Checked on f(x) = 3x² + 2x: f′(2) = 14.
- `partial_derivatives` calls it once per parameter (m₁, m₂, b) to build the gradient.
- Every iteration prints ŷ = Xm + b, the error, the cost, the SciPy gradient beside the chain-rule gradient (identical), and the update of m and b. The three iterations reproduce Part 3 exactly.
- Predictions with the final m and b: [2.852224, 6.409812] (targets [5, 6]); J = 2.39044381.

![Parameters over 3 iterations](images/part4_gd_parameters.png)
![Error over 3 iterations](images/part4_gd_error.png)

Continuing the same code for 2000 iterations: m ≈ [−2.2258, 1.0968], b ≈ 3.9353, J ≈ 0, predictions ≈ [5, 6] - the error keeps falling towards zero, as Part 3's trend predicted.

![Parameters over 2000 iterations](images/part4_gd_parameters_2000.png)
![Error over 2000 iterations](images/part4_gd_error_2000.png)
