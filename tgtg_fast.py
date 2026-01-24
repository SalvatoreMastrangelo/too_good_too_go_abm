"""
tgtg_fast.py

Fast, reproducible simulation core for a TooGoodToGo-style (TGTG) mechanism
in a bakery agent-based model.

Key design choices:
- Consumers are represented as an integer preference matrix prefs[N, L].
- All randomness is pre-generated (common random numbers) and passed in:
  - perm[D, N]: arrival order per day
  - visit_u[D, N]: uniforms for visit decisions (compare to alpha_t)
  - walk_u[D, N, L]: uniforms for walk-out decisions at each preference rank
- The core epoch simulator is numba-compiled and deterministic given inputs.
- Outer loops (population evaluation, scenario sweeps) can be parallelised safely.

This file is meant to be imported from a notebook or used in scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any, List
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn): return fn
        return wrap


# -----------------------------
# Configs / simple containers
# -----------------------------
@dataclass(frozen=True)
class EnvironmentConfig:
    N: int
    L: int
    r: float           # walk-out probability after a stockout at a preference rank
    chi: float         # unit cost
    rho: float         # regular unit price
    tau: float         # TGTG "bag" unit price (per unit, in your current model)
    # NOTE: in your PDF, tau is the price per bag; here we treat it as per unit
    # because your current simulator sells "tgtg_sales" as units. Adapt as needed.


@dataclass
class BakerAgent:
    q: np.ndarray      # shape (L,), non-negative integers (or floats rounded later)
    b: float           # share reserved for TGTG in [0,1]
    gamma: float       # risk aversion
    fitness: float = 0.0


# -----------------------------
# Preference generation
# -----------------------------
def generate_preferences(
    N: int,
    L: int,
    mode: Literal["random", "correlated"] = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Returns prefs[N, L] as a permutation of goods indices per consumer.

    - random: each row is an independent permutation
    - correlated: a few goods are more likely to appear early in the ranking
    """
    if rng is None:
        rng = np.random.default_rng()

    prefs = np.empty((N, L), dtype=np.int32)

    if mode == "random":
        base = np.arange(L, dtype=np.int32)
        for i in range(N):
            prefs[i] = rng.permutation(base)
        return prefs

    if mode == "correlated":
        # Popularity weights: early goods more likely to be chosen first.
        # You can tune "pop_strength" to control correlation.
        pop_strength = 2.0
        weights = np.exp(-pop_strength * np.arange(L, dtype=np.float64))
        weights = weights / weights.sum()

        goods = np.arange(L, dtype=np.int32)
        for i in range(N):
            remaining = goods.copy()
            w = weights.copy()
            row = np.empty(L, dtype=np.int32)
            for k in range(L):
                # sample one good without replacement with current weights
                idx = rng.choice(len(remaining), p=w[:len(remaining)] / w[:len(remaining)].sum())
                row[k] = remaining[idx]
                remaining = np.delete(remaining, idx)
                w = np.delete(w, idx)
            prefs[i] = row
        return prefs

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Demand models (alpha paths)
# -----------------------------
def alpha_path_constant(D: int, alpha: float) -> np.ndarray:
    return np.full(D, float(alpha), dtype=np.float64)

def alpha_path_beta_shocks(
    D: int,
    alpha_mean: float,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Day-level i.i.d. shocks: alpha_t ~ Beta(a,b) with mean alpha_mean
    and concentration a+b = concentration.
    Lower concentration => higher volatility.
    """
    a = max(1e-6, alpha_mean * concentration)
    b = max(1e-6, (1.0 - alpha_mean) * concentration)
    return rng.beta(a, b, size=D).astype(np.float64)

def alpha_path_logit_ar1(
    D: int,
    alpha0: float,
    phi: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Persistent demand volatility:
    logit(alpha_t) = phi * logit(alpha_{t-1}) + eps_t, eps_t ~ N(0, sigma^2)
    """
    def logit(x: float) -> float:
        x = min(max(x, 1e-6), 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    def logistic(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    z = np.empty(D, dtype=np.float64)
    z[0] = logit(alpha0)
    eps = rng.normal(0.0, sigma, size=D).astype(np.float64)
    for t in range(1, D):
        z[t] = phi * z[t - 1] + eps[t]
    return np.array([logistic(v) for v in z], dtype=np.float64)


# -----------------------------
# Common random numbers
# -----------------------------
def make_common_random_draws(
    D: int,
    N: int,
    L: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-generate all randomness used by the simulator:
      perm[d, :]    arrival permutation for day d
      visit_u[d, :] uniforms for visit decisions (compare to alpha_t)
      walk_u[d,:,:] uniforms for walk-out decisions at each pref rank

    This enables:
    - reproducibility
    - lower selection noise in evolutionary training (common random numbers)
    - thread-safe parallel evaluation (no RNG used in compiled core)
    """
    rng = np.random.default_rng(seed)
    perm = np.empty((D, N), dtype=np.int32)
    for d in range(D):
        perm[d] = rng.permutation(N).astype(np.int32)
    visit_u = rng.random((D, N), dtype=np.float64)
    walk_u = rng.random((D, N, L), dtype=np.float64)
    return perm, visit_u, walk_u


# -----------------------------
# Fast core simulation (numba)
# -----------------------------
@njit(cache=True)
def _simulate_epoch_precomputed(
    q: np.ndarray,              # (L,)
    b: float,
    gamma: float,
    prefs: np.ndarray,          # (N, L) int
    alpha_path: np.ndarray,     # (D,) float
    perm: np.ndarray,           # (D, N) int
    visit_u: np.ndarray,        # (D, N) float
    walk_u: np.ndarray,         # (D, N, L) float
    r: float,
    chi: float,
    rho: float,
    tau: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns:
      fitness, mean_profit, std_profit, total_regular_sales, total_tgtg_sales, total_waste, total_production

    Notes:
    - Shop closes for regular sales once total_inventory <= reserved.
    - A sale is allowed only if selling would not push total_inventory below reserved.
    """
    D = alpha_path.shape[0]
    N = prefs.shape[0]
    L = prefs.shape[1]

    qsum = 0
    for k in range(L):
        # treat q as non-negative (floor); you can also round outside.
        if q[k] > 0:
            qsum += int(q[k])
    unit_cost = qsum * chi

    # store daily profits for variance
    profits = np.empty(D, dtype=np.float64)

    total_regular_sales = 0.0
    total_tgtg_sales = 0.0
    total_waste = 0.0

    for d in range(D):
        # initialize inventory
        inventory = np.empty(L, dtype=np.int32)
        total_inv = 0
        for k in range(L):
            val = int(q[k]) if q[k] > 0 else 0
            inventory[k] = val
            total_inv += val

        reserved = int(b * total_inv)
        regular_sales = 0
        walkouts = 0

        alpha_t = alpha_path[d]

        # iterate consumers in random arrival order
        for pos in range(N):
            i = perm[d, pos]
            if visit_u[d, i] >= alpha_t:
                continue  # does not visit

            # shop closed for regular sales?
            if total_inv <= reserved:
                # remaining visiting consumers would all walk out; but we don't know how many
                # are left among the remaining positions, so we count walkout only for this visitor
                walkouts += 1
                continue

            bought_or_left = False
            # go through preference ranking
            for rank in range(L):
                g = prefs[i, rank]
                if inventory[g] > 0:
                    # can we sell without violating reserved stock?
                    if total_inv - 1 < reserved:
                        walkouts += 1
                    else:
                        inventory[g] -= 1
                        total_inv -= 1
                        regular_sales += 1
                    bought_or_left = True
                    break
                else:
                    # stockout at this rank => possible walkout
                    if walk_u[d, i, rank] < r:
                        walkouts += 1
                        bought_or_left = True
                        break

            if not bought_or_left:
                # exhausted preference list
                walkouts += 1

        # TGTG: guaranteed sale up to reserved, but can't exceed remaining inventory
        tgtg_sales = reserved if reserved <= total_inv else total_inv
        waste = total_inv - tgtg_sales

        profit = regular_sales * rho + tgtg_sales * tau - unit_cost
        profits[d] = profit

        total_regular_sales += regular_sales
        total_tgtg_sales += tgtg_sales
        total_waste += waste

    # mean & std
    mean_profit = profits.mean()
    var_profit = profits.var()
    std_profit = np.sqrt(var_profit)

    fitness = D * (mean_profit - gamma * std_profit)
    total_production = float(qsum) * float(D)

    return (fitness, mean_profit, std_profit,
            total_regular_sales, total_tgtg_sales, total_waste, total_production)


def simulate_epoch_fast(
    agent: BakerAgent,
    prefs: np.ndarray,
    env: EnvironmentConfig,
    alpha_path: np.ndarray,
    perm: np.ndarray,
    visit_u: np.ndarray,
    walk_u: np.ndarray,
) -> Dict[str, float]:
    """
    Python wrapper returning a dict of key metrics.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is not available; install numba or use your original python simulator.")

    q = agent.q.astype(np.float64)
    # Keep q non-negative integers
    q = np.maximum(0.0, np.floor(q))

    b = float(min(max(agent.b, 0.0), 1.0))

    out = _simulate_epoch_precomputed(
        q=q,
        b=b,
        gamma=float(agent.gamma),
        prefs=prefs.astype(np.int32),
        alpha_path=alpha_path.astype(np.float64),
        perm=perm.astype(np.int32),
        visit_u=visit_u.astype(np.float64),
        walk_u=walk_u.astype(np.float64),
        r=float(env.r),
        chi=float(env.chi),
        rho=float(env.rho),
        tau=float(env.tau),
    )
    fitness, mean_profit, std_profit, reg_sales, tgtg_sales, waste, production = out
    return {
        "fitness": float(fitness),
        "mean_profit": float(mean_profit),
        "std_profit": float(std_profit),
        "regular_sales": float(reg_sales),
        "tgtg_sales": float(tgtg_sales),
        "waste": float(waste),
        "production": float(production),
    }
