"""Probabilistic convenience layer for Monte Carlo uncertainty analysis.

Draws samples from user-specified distributions, evaluates LCOE via
``CostModel.batch_lcoe``, and returns summary statistics (P10/P50/P90,
confidence intervals).

Two entry points:

* ``run_uncertainty`` -- fast path for engineering/financial params only.
* ``run_uncertainty_full`` -- slow path that also varies CostingConstants
  fields (blanket_unit_cost, shield_unit_cost, etc.) via discretised bins.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from costingfe.defaults import CostingConstants, load_costing_constants
from costingfe.model import CostModel
from costingfe.types import ConfinementConcept, Fuel

# ---------------------------------------------------------------------------
# Distribution types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Normal:
    """Gaussian distribution."""

    mean: float
    sigma: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.sigma, size=n)


@dataclass(frozen=True)
class Uniform:
    """Bounded uniform distribution."""

    low: float
    high: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)


@dataclass(frozen=True)
class Triangular:
    """Triangular distribution for expert elicitation."""

    low: float
    mode: float
    high: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=n)


@dataclass(frozen=True)
class LogNormal:
    """Log-normal distribution (right-skewed cost uncertainty).

    Parameters are the mean and std-dev of the *underlying* normal,
    i.e. ``exp(N(mu, sigma_log))``.
    """

    mu: float
    sigma_log: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma_log, size=n)


Distribution = Normal | Uniform | Triangular | LogNormal


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    """Summary statistics from a Monte Carlo uncertainty run."""

    lcoe_samples: np.ndarray  # (N,) raw LCOE values
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    param_samples: dict[str, np.ndarray]  # drawn samples per parameter
    n_samples: int

    @property
    def p5(self) -> float:
        return float(np.percentile(self.lcoe_samples, 5))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.lcoe_samples, 95))

    @property
    def ci_80(self) -> tuple[float, float]:
        """80 % confidence interval (P10, P90)."""
        return (self.p10, self.p90)

    @property
    def ci_90(self) -> tuple[float, float]:
        """90 % confidence interval (P5, P95)."""
        return (self.p5, self.p95)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_samples(
    distributions: dict[str, Distribution],
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw *n* independent samples from each distribution."""
    return {name: dist.sample(n, rng) for name, dist in distributions.items()}


def _apply_correlation(
    samples: dict[str, np.ndarray],
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Induce rank correlation via the Iman-Conover method.

    Generates correlated standard normals using Cholesky decomposition,
    then rank-matches each marginal to the target rank order.  Preserves
    exact marginal distributions while inducing desired rank correlation.
    """
    names = list(samples.keys())
    m = len(names)
    n = len(samples[names[0]])

    if corr_matrix.shape != (m, m):
        raise ValueError(
            f"correlation_matrix shape {corr_matrix.shape} does not match "
            f"number of uncertain parameters ({m})"
        )

    # Step 1: Cholesky factor of target correlation matrix
    L = np.linalg.cholesky(corr_matrix)

    # Step 2: Generate independent standard normals, then correlate
    z = rng.standard_normal((n, m))
    z_corr = z @ L.T  # shape (n, m)

    # Step 3: For each variable, rank-match the original marginal samples
    #         to the rank order implied by z_corr.
    result = {}
    for j, name in enumerate(names):
        target_ranks = np.argsort(np.argsort(z_corr[:, j]))
        sorted_marginal = np.sort(samples[name])
        result[name] = sorted_marginal[target_ranks]

    return result


def _build_result(
    lcoe_values: np.ndarray,
    param_samples: dict[str, np.ndarray],
) -> UncertaintyResult:
    """Construct an ``UncertaintyResult`` from raw LCOE values."""
    return UncertaintyResult(
        lcoe_samples=lcoe_values,
        mean=float(np.mean(lcoe_values)),
        std=float(np.std(lcoe_values)),
        p10=float(np.percentile(lcoe_values, 10)),
        p50=float(np.percentile(lcoe_values, 50)),
        p90=float(np.percentile(lcoe_values, 90)),
        param_samples=param_samples,
        n_samples=len(lcoe_values),
    )


# ---------------------------------------------------------------------------
# Fast path: only engineering / financial params
# ---------------------------------------------------------------------------


def run_uncertainty(
    model: CostModel,
    base_params: dict,
    param_distributions: dict[str, Distribution],
    n_samples: int = 1000,
    cost_overrides: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
    seed: int | None = None,
) -> UncertaintyResult:
    """Run Monte Carlo uncertainty analysis on engineering/financial params.

    Uses a single ``batch_lcoe`` call for all *n_samples* rows.

    Args:
        model: A constructed ``CostModel``.
        base_params: Base parameter dict (typically from ``forward().params``).
        param_distributions: Mapping of parameter name to ``Distribution``.
        n_samples: Number of Monte Carlo draws (default 1000).
        cost_overrides: Optional CAS account overrides passed through to
            ``batch_lcoe``.
        correlation_matrix: Optional (M, M) rank-correlation matrix for the
            M uncertain parameters, applied via Iman-Conover.  Must be
            symmetric positive-definite with ones on the diagonal.
        seed: Random seed for reproducibility.

    Returns:
        ``UncertaintyResult`` with LCOE samples and summary statistics.

    Raises:
        ValueError: If a parameter name is not traceable by ``batch_lcoe``.
    """
    rng = np.random.default_rng(seed)

    # Validate parameter names against the model's continuous keys
    _, keys, _ = model._build_lcoe_fn(base_params, cost_overrides)
    for name in param_distributions:
        if name not in keys:
            raise ValueError(
                f"Parameter '{name}' is not a traceable continuous parameter. "
                f"Valid parameters: {keys}"
            )

    # Draw samples
    samples = _draw_samples(param_distributions, n_samples, rng)

    # Apply correlation if requested
    if correlation_matrix is not None:
        samples = _apply_correlation(samples, correlation_matrix, rng)

    # Build param_sets for batch_lcoe: each value must be a list
    param_sets = {name: vals.tolist() for name, vals in samples.items()}

    lcoe_list = model.batch_lcoe(param_sets, base_params, cost_overrides)
    lcoe_arr = np.array(lcoe_list)

    return _build_result(lcoe_arr, samples)


# ---------------------------------------------------------------------------
# Slow path: also varies CostingConstants
# ---------------------------------------------------------------------------


def run_uncertainty_full(
    concept: ConfinementConcept,
    fuel: Fuel,
    base_params: dict,
    param_distributions: dict[str, Distribution],
    cc_distributions: dict[str, Distribution] | None = None,
    base_cc: CostingConstants | None = None,
    n_samples: int = 1000,
    cost_overrides: dict[str, float] | None = None,
    correlation_matrix: np.ndarray | None = None,
    n_cc_bins: int = 5,
    seed: int | None = None,
) -> UncertaintyResult:
    """Monte Carlo uncertainty that can also vary CostingConstants fields.

    CostingConstants are baked into the ``CostModel`` at construction, so
    they cannot be traced through ``batch_lcoe``.  This function discretises
    each CC distribution into *n_cc_bins* quantile bins, creates one
    ``CostModel`` per grid cell, and runs ``batch_lcoe`` once per cell.

    Falls back to ``run_uncertainty`` when *cc_distributions* is empty/None.

    Args:
        concept: Confinement concept (e.g. ``ConfinementConcept.TOKAMAK``).
        fuel: Fuel type (e.g. ``Fuel.DT``).
        base_params: Base parameter dict (typically from ``forward().params``).
        param_distributions: Engineering/financial distributions.
        cc_distributions: Distributions for CostingConstants fields
            (e.g. ``{"blanket_unit_cost_dt": Uniform(0.8, 1.5)}``).
        base_cc: Base CostingConstants (defaults to ``load_costing_constants()``).
        n_samples: Total number of Monte Carlo draws.
        cost_overrides: CAS account overrides.
        correlation_matrix: Rank-correlation matrix for ``param_distributions``
            only (CC params are binned independently).
        n_cc_bins: Number of quantile bins per CC parameter (default 5).
        seed: Random seed for reproducibility.

    Returns:
        ``UncertaintyResult`` with LCOE samples and summary statistics.
    """
    if not cc_distributions:
        model = CostModel(concept=concept, fuel=fuel, costing_constants=base_cc)
        return run_uncertainty(
            model,
            base_params,
            param_distributions,
            n_samples=n_samples,
            cost_overrides=cost_overrides,
            correlation_matrix=correlation_matrix,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    if base_cc is None:
        base_cc = load_costing_constants()

    cc_names = list(cc_distributions.keys())
    n_cc = len(cc_names)

    if n_cc > 3:
        warnings.warn(
            f"Varying {n_cc} CostingConstants parameters creates {n_cc_bins}^{n_cc} "
            f"= {n_cc_bins**n_cc} grid cells. Consider reducing n_cc_bins or "
            f"the number of CC parameters.",
            stacklevel=2,
        )

    # Validate CC field names
    for name in cc_names:
        if not hasattr(base_cc, name):
            raise ValueError(f"'{name}' is not a field of CostingConstants.")

    # Draw CC samples (full n_samples) for bookkeeping
    cc_samples = _draw_samples(cc_distributions, n_samples, rng)

    # Build quantile bin edges for each CC param
    # For Q bins, we need Q+1 edges; assign each sample to a bin
    bin_assignments = {}  # name -> array of bin indices (0..Q-1)
    bin_centers = {}  # name -> array of Q representative values
    for name in cc_names:
        quantiles = np.linspace(0, 100, n_cc_bins + 1)
        edges = np.percentile(cc_samples[name], quantiles)
        # Assign each sample to a bin via digitize (1-indexed), clamp to [0, Q-1]
        assignments = np.digitize(cc_samples[name], edges[1:-1])  # 0-indexed bins
        bin_assignments[name] = assignments
        # Compute bin centers as mean of samples in each bin
        centers = np.empty(n_cc_bins)
        for b in range(n_cc_bins):
            mask = assignments == b
            if np.any(mask):
                centers[b] = np.mean(cc_samples[name][mask])
            else:
                # Empty bin — use midpoint of edges
                centers[b] = (edges[b] + edges[b + 1]) / 2.0
        bin_centers[name] = centers

    # Draw engineering/financial samples
    eng_samples = _draw_samples(param_distributions, n_samples, rng)
    if correlation_matrix is not None:
        eng_samples = _apply_correlation(eng_samples, correlation_matrix, rng)

    # Build combined grid index for each sample
    # grid_key[i] = tuple of bin indices for sample i
    grid_keys = np.column_stack([bin_assignments[name] for name in cc_names])

    # Group samples by grid cell
    from collections import defaultdict

    cell_indices: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n_samples):
        key = tuple(grid_keys[i])
        cell_indices[key].append(i)

    # Evaluate each grid cell
    lcoe_all = np.empty(n_samples)
    for cell_key, indices in cell_indices.items():
        # Build CostingConstants for this cell
        cc_overrides = {}
        for j, name in enumerate(cc_names):
            cc_overrides[name] = float(bin_centers[name][cell_key[j]])
        cc = base_cc.replace(**cc_overrides)
        cell_model = CostModel(concept=concept, fuel=fuel, costing_constants=cc)

        # Validate param names on first cell
        if cell_key == next(iter(cell_indices)):
            _, keys, _ = cell_model._build_lcoe_fn(base_params, cost_overrides)
            for name in param_distributions:
                if name not in keys:
                    raise ValueError(
                        f"Parameter '{name}' is not a traceable continuous "
                        f"parameter. Valid parameters: {keys}"
                    )

        # Build param_sets for this cell's subset
        param_sets = {}
        for name, arr in eng_samples.items():
            param_sets[name] = [float(arr[i]) for i in indices]

        cell_lcoes = cell_model.batch_lcoe(param_sets, base_params, cost_overrides)
        for k, idx in enumerate(indices):
            lcoe_all[idx] = cell_lcoes[k]

    # Merge all samples for result
    all_samples = {**eng_samples, **cc_samples}
    return _build_result(lcoe_all, all_samples)
