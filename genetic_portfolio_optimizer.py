"""
Genetic Algorithm for Investment Portfolio Optimization
=======================================================
Problem: Find a portfolio allocation that maximizes the Sharpe Ratio,
balancing risk (volatility) and return for a moderate-risk investor.

Success criteria: evolve a population of portfolios until the best
individual reaches a target Sharpe Ratio or the generation limit is hit.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_ASSETS = 5
POPULATION_SIZE = 100
ELITE_SIZE = 5
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.02
MAX_GENERATIONS = 200
TARGET_SHARPE = 15.0
RISK_FREE_RATE = 0.01

# Simulated historical daily returns for 5 assets (rows = time steps)
HISTORICAL_RETURNS = np.array([
    [0.01,  0.02,  0.015, -0.005,  0.007],
    [0.012, 0.018, 0.016, -0.004,  0.009],
    [0.015, 0.017, 0.014, -0.006,  0.008],
    [0.013, 0.019, 0.015, -0.003,  0.007],
    [0.014, 0.016, 0.013, -0.002,  0.009],
])

ASSET_LABELS = [f"Asset {i + 1}" for i in range(NUM_ASSETS)]


# ---------------------------------------------------------------------------
# Core genetic algorithm operators
# ---------------------------------------------------------------------------

def generate_portfolio(num_assets: int) -> np.ndarray:
    """Return a random portfolio: weights sum to 1 (fully invested)."""
    weights = np.random.rand(num_assets)
    return weights / weights.sum()


def fitness_function(
    weights: np.ndarray,
    returns: np.ndarray,
    risk_free_rate: float,
) -> float:
    """
    Compute the Sharpe Ratio for a given portfolio weight vector.

    Sharpe Ratio = (E[Rp] - Rf) / sigma_p
    where E[Rp] is expected portfolio return, Rf is the risk-free rate,
    and sigma_p is portfolio standard deviation (volatility / risk).
    """
    mean_returns = returns.mean(axis=0)
    cov_matrix = np.cov(returns.T)

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_std = np.sqrt(portfolio_variance)

    if portfolio_std == 0:
        return 0.0

    return (portfolio_return - risk_free_rate) / portfolio_std


def tournament_selection(
    population: list,
    fitness_scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Select one individual via k-way tournament (highest fitness wins)."""
    indices = np.random.choice(len(population), k, replace=False)
    best = max(indices, key=lambda i: fitness_scores[i])
    return population[best]


def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Produce a child by randomly inheriting each gene from one of the parents.
    The resulting weights are re-normalised so they still sum to 1.
    """
    mask = np.random.rand(len(parent1)) < 0.5
    child = np.where(mask, parent1, parent2)
    return child / child.sum()


def gaussian_mutation(weights: np.ndarray, mutation_rate: float) -> np.ndarray:
    """
    Perturb weights with Gaussian noise then re-normalise.
    Negative weights are clipped to zero before normalisation so all
    allocations stay non-negative (no short selling).
    """
    mutated = weights + np.random.normal(0, mutation_rate, len(weights))
    mutated = np.clip(mutated, 0, None)
    total = mutated.sum()
    # If all weights collapsed to zero, fall back to a fresh random portfolio
    return mutated / total if total > 0 else generate_portfolio(len(weights))


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

def run_genetic_algorithm(
    returns: np.ndarray = HISTORICAL_RETURNS,
    num_assets: int = NUM_ASSETS,
    population_size: int = POPULATION_SIZE,
    elite_size: int = ELITE_SIZE,
    tournament_size: int = TOURNAMENT_SIZE,
    mutation_rate: float = MUTATION_RATE,
    max_generations: int = MAX_GENERATIONS,
    target_sharpe: float = TARGET_SHARPE,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[np.ndarray, float, list]:
    """
    Evolve a population of portfolios and return the best solution found.

    Returns
    -------
    best_weights   : weight vector of the best portfolio
    best_sharpe    : Sharpe Ratio of the best portfolio
    history        : list of (generation, best_sharpe) for plotting
    """
    population = [generate_portfolio(num_assets) for _ in range(population_size)]
    history: list[tuple[int, float]] = []
    best_weights = population[0]
    best_sharpe = -np.inf

    for generation in range(max_generations):
        fitness_scores = np.array([
            fitness_function(p, returns, risk_free_rate) for p in population
        ])

        gen_best_idx = np.argmax(fitness_scores)
        gen_best_sharpe = fitness_scores[gen_best_idx]

        if gen_best_sharpe > best_sharpe:
            best_sharpe = gen_best_sharpe
            best_weights = population[gen_best_idx].copy()

        history.append((generation, best_sharpe))

        if best_sharpe >= target_sharpe:
            print(f"Target Sharpe Ratio reached at generation {generation}.")
            break

        # Elitism: carry the top-k individuals unchanged
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[i].copy() for i in elite_indices]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores, tournament_size)
            parent2 = tournament_selection(population, fitness_scores, tournament_size)
            child = uniform_crossover(parent1, parent2)
            child = gaussian_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    else:
        print(f"Max generations ({max_generations}) reached.")

    return best_weights, best_sharpe, history


# ---------------------------------------------------------------------------
# Baseline: equally-weighted portfolio (naive benchmark)
# ---------------------------------------------------------------------------

def equal_weight_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Compute the Sharpe Ratio of an equally-weighted portfolio."""
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    return fitness_function(weights, returns, risk_free_rate)


# ---------------------------------------------------------------------------
# Reporting & visualisation
# ---------------------------------------------------------------------------

def print_portfolio(weights: np.ndarray, sharpe: float, labels: list) -> None:
    print("\n=== Optimised Portfolio ===")
    for label, w in zip(labels, weights):
        print(f"  {label}: {w * 100:.2f}%")
    print(f"  Sharpe Ratio : {sharpe:.4f}")


def plot_results(
    history: list,
    best_weights: np.ndarray,
    labels: list,
    baseline_sharpe: float,
) -> None:
    generations = [h[0] for h in history]
    sharpes = [h[1] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Genetic Algorithm — Portfolio Optimisation", fontsize=14, fontweight="bold")

    # Convergence curve
    axes[0].plot(generations, sharpes, color="steelblue", linewidth=1.8, label="Best Sharpe")
    axes[0].axhline(baseline_sharpe, color="tomato", linestyle="--", label=f"Equal-weight baseline ({baseline_sharpe:.2f})")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Convergence Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Portfolio allocation pie chart
    axes[1].pie(
        best_weights,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.tab10.colors[:len(labels)],
    )
    axes[1].set_title("Best Portfolio Allocation")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nChart saved to results.png")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)  # reproducibility

    print("Running Genetic Algorithm for Portfolio Optimisation...\n")

    best_weights, best_sharpe, history = run_genetic_algorithm()

    baseline = equal_weight_sharpe(HISTORICAL_RETURNS)
    print(f"Equal-weight baseline Sharpe Ratio : {baseline:.4f}")

    print_portfolio(best_weights, best_sharpe, ASSET_LABELS)

    final_gen = history[-1][0]
    print(f"\nEvolution completed in {final_gen + 1} generation(s).")
    print(f"Improvement over baseline: {best_sharpe - baseline:+.4f}")

    plot_results(history, best_weights, ASSET_LABELS, baseline)
