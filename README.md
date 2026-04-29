# Genetic Algorithm — Investment Portfolio Optimisation

A Python implementation of a **Genetic Algorithm (GA)** applied to the classic portfolio optimisation problem. The goal is to find the allocation of assets that **maximises the Sharpe Ratio** — the industry-standard measure that balances return against risk.

---

## Problem Statement

A moderate-risk investor needs to allocate capital across **5 assets**. The challenge is walking the fine line between:

- **Minimising risk** (portfolio volatility / standard deviation)
- **Maximising return** (expected portfolio return above the risk-free rate)

The Sharpe Ratio captures both dimensions in a single metric:

$$\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$$

| Symbol | Meaning |
|--------|---------|
| $E(R_p)$ | Expected portfolio return |
| $R_f$ | Risk-free rate |
| $\sigma_p$ | Portfolio standard deviation (risk) |

A higher Sharpe Ratio means more return per unit of risk taken.

---

## How the Genetic Algorithm Works

```
Initialise random population of portfolios
        │
        ▼
Evaluate fitness (Sharpe Ratio) for each portfolio
        │
        ▼
Target reached? ──Yes──► Return best portfolio
        │ No
        ▼
Elitism  ──► keep top-k individuals unchanged
        │
        ▼
Tournament Selection  ──► pick parents by competition
        │
        ▼
Uniform Crossover  ──► blend parent genes randomly
        │
        ▼
Gaussian Mutation  ──► perturb weights with noise
        │
        ▼
Re-normalise weights (sum = 1, no short selling)
        │
        └──────────────────────────────────► next generation
```

| Operator | Details |
|----------|---------|
| **Representation** | Weight vector $w \in \mathbb{R}^n$, $\sum w_i = 1$, $w_i \geq 0$ |
| **Fitness** | Sharpe Ratio |
| **Selection** | $k$-way tournament |
| **Crossover** | Uniform (random binary mask) |
| **Mutation** | Gaussian noise + clip negatives + re-normalise |
| **Elitism** | Top-$k$ individuals survive unchanged |

---

## Results

The GA consistently outperforms the naive equal-weight baseline:

| Portfolio | Sharpe Ratio |
|-----------|-------------|
| Equal-weight (baseline) | ~2.3 |
| GA-optimised | **≥ 15.0** |

The convergence chart and final allocation pie are saved to `results.png` after each run.

---

## Project Structure

```
TechChallenge02/
├── genetic_portfolio_optimizer.py   # Main source — fully self-contained
├── results.png                      # Generated chart (created on first run)
├── TechChallenge2.ipynb             # Original Jupyter notebook (academic submission)
└── README.md
```

---

## Quick Start

**Requirements:** Python 3.10+ with `numpy` and `matplotlib`.

```bash
pip install numpy matplotlib
python genetic_portfolio_optimizer.py
```

### Tuning the algorithm

All hyperparameters are constants at the top of the file:

```python
POPULATION_SIZE = 100   # individuals per generation
ELITE_SIZE      = 5     # top individuals preserved unchanged
TOURNAMENT_SIZE = 5     # competitors per selection round
MUTATION_RATE   = 0.02  # std-dev of Gaussian noise
MAX_GENERATIONS = 200   # hard stop
TARGET_SHARPE   = 15.0  # early-stop threshold
```

---

## Key Design Decisions

- **No short selling** — weights are clipped to `[0, 1]` and re-normalised after mutation, ensuring every allocation is non-negative.
- **Elitism** — the best portfolios are copied verbatim to the next generation so the GA never loses its best solution.
- **Reproducible runs** — `np.random.seed(42)` is set at entry point; remove it for stochastic exploration.
- **Baseline comparison** — an equally-weighted portfolio is computed automatically so results are always contextualised.

---

## Academic Context

Developed as **Tech Challenge 02** for the Postgraduate Programme in Artificial Intelligence — FIAP (2024).  
Author: **Guilherme Augusto da Silva** — RM 354300
