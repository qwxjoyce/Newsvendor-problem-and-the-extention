

import os
import sys
import time
import json
import hashlib
import platform
import shutil
from datetime import datetime, timezone
from copy import copy
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
if os.getenv("SHOW_PLOTS", "0") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from scipy.special import betainc, gammaincc
from scipy.stats import (
    beta as beta_distribution,
    gamma as gamma_distribution,
    laplace,
    norm,
    triang,
)

# Pyomo is imported conditionally so paper_check can run without Pyomo.
try:
    from pyomo.environ import (
        ConcreteModel,
        Constraint,
        NonNegativeReals,
        Objective,
        Param,
        RangeSet,
        Var,
        maximize,
        value,
    )
    from pyomo.opt import SolverFactory, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False


# 1. USER SETTINGS

RUN_MODE = os.getenv("RUN_MODE", "full")   # test, paper_check, or full
SOLVER_NAME = "scipy-highs"   # Four-variable LP, equivalent to scenario recourse LP.
# Paper Section 3.2: 405 Normal + 405 Exponential/RSE combinations.
DEMAND_GROUP = os.getenv("DEMAND_GROUP", "all")  # all, normal, exponential
if DEMAND_GROUP not in {"all", "normal", "exponential"}:
    raise ValueError("DEMAND_GROUP must be all, normal, or exponential.")
group_ids = list(range(1, 811)) if DEMAND_GROUP == "all" else (
    list(range(1, 406)) if DEMAND_GROUP == "normal" else list(range(406, 811)))
SELECTED_CASES = group_ids
if RUN_MODE == "test":
    SELECTED_CASES = [i for i in [1, 203, 405, 406, 608, 810] if i in group_ids]
if os.getenv("CASE_IDS"):
    SELECTED_CASES = [int(i.strip()) for i in os.environ["CASE_IDS"].split(",")]
    if not SELECTED_CASES or len(set(SELECTED_CASES)) != len(SELECTED_CASES) or any(i not in group_ids for i in SELECTED_CASES):
        raise ValueError("CASE_IDS must contain unique case numbers in the selected demand group.")
TRAIN_SEED = 20
VALIDATION_SEED = 2026
BASE_FOLDER = Path(__file__).resolve().parent
SAVE_FILES = True
SHOW_PLOTS = os.getenv("SHOW_PLOTS", "0") == "1"
LAMBDA_TIME_REPEATS = 5
HEURISTIC_TIME_REPEATS = 100
CAPACITY_FRACTIONS = np.arange(40, 101, 4, dtype=float) / 100.0
# Compatibility name used by the existing plots: these are capacity fractions.
SERVICE_LEVELS = CAPACITY_FRACTIONS

if RUN_MODE == "test":
    RUN_SAA = True
    SAA_N = 200
    SAA_REPLICATIONS = 1
    VALIDATION_N = 5_000
elif RUN_MODE == "paper_check":
    RUN_SAA = False
    SAA_N = 0
    SAA_REPLICATIONS = 0
    VALIDATION_N = 0
elif RUN_MODE == "full":
    RUN_SAA = True
    SAA_N = 10_000
    SAA_REPLICATIONS = 20
    VALIDATION_N = 100_000
else:
    raise ValueError("RUN_MODE must be 'test', 'paper_check', or 'full'.")

OUTPUT_FOLDER = BASE_FOLDER / "erlebacher_full_general_piecewise_results" / RUN_MODE
EXCEL_FILENAME = "full_general_summary.xlsx"
NUMBER_OF_PRODUCTS = 2
PRICE = np.array([4.0, 3.0])
COST = np.array([2.0, 2.0])
SALVAGE = np.array([1.0, 0.0])
SHORTAGE_PENALTY = np.zeros(2)
RESOURCE = np.ones(2)
CRITICAL_RATIO = (PRICE - COST) / (PRICE - SALVAGE)
BETA_COST = RESOURCE / (PRICE - SALVAGE)
METHOD_ORDER = ["Exact Lambda", "H1", "H2", "H3", "SAA"]


# 3. DEMAND DISTRIBUTION OBJECT

class DemandDistribution:
    """
    Small container for one demand distribution.

    ppf(probability):
        inverse CDF

    sampler(rng, n):
        random sample

    shortage(q):
        E[max(D-q, 0)]
    """

    def __init__(
        self,
        name,
        mean,
        std,
        ppf_function,
        sampler_function,
        shortage_function,
    ):
        self.name = name
        self.mean = float(mean)
        self.std = float(std)
        self._ppf = ppf_function
        self._sampler = sampler_function
        self._shortage = shortage_function

    def ppf(self, probability):
        probability = float(
            np.clip(probability, 1e-12, 1.0 - 1e-12)
        )
        return float(self._ppf(probability))

    def sample(self, rng, n):
        return np.asarray(self._sampler(rng, n), dtype=float)

    def expected_shortage(self, q):
        return float(self._shortage(float(q)))


# 4. DISTRIBUTION FACTORIES

def make_normal(mean=50.0, std=10.0):
    def ppf(probability):
        return norm.ppf(probability, loc=mean, scale=std)

    def sampler(rng, n):
        return rng.normal(mean, std, n)

    def shortage(q):
        z = (q - mean) / std
        return (
            std * norm.pdf(z)
            - (q - mean) * (1.0 - norm.cdf(z))
        )

    return DemandDistribution(
        f"Normal(mean={mean:g}, std={std:g})",
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_uniform(mean=50.0, std=10.0):
    half_width = std * np.sqrt(3.0)
    lower = mean - half_width
    upper = mean + half_width

    def ppf(probability):
        return lower + probability * (upper - lower)

    def sampler(rng, n):
        return rng.uniform(lower, upper, n)

    def shortage(q):
        if q <= lower:
            return mean - q
        if q >= upper:
            return 0.0
        return (upper - q) ** 2 / (2.0 * (upper - lower))

    return DemandDistribution(
        f"Uniform(mean={mean:g}, std={std:g})",
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_double_exponential(mean=50.0, std=10.0):
    """
    Double exponential = Laplace distribution.

    Laplace standard deviation = sqrt(2) * scale.
    """
    scale = std / np.sqrt(2.0)

    def ppf(probability):
        return laplace.ppf(probability, loc=mean, scale=scale)

    def sampler(rng, n):
        return rng.laplace(mean, scale, n)

    def shortage(q):
        if q >= mean:
            return 0.5 * scale * np.exp(-(q - mean) / scale)

        return (
            mean - q
            + 0.5 * scale * np.exp((q - mean) / scale)
        )

    return DemandDistribution(
        f"Double Exponential(mean={mean:g}, std={std:g})",
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_gamma(mean=50.0, std=10.0):
    """
    Gamma parameters from mean and standard deviation:

        shape = (mean/std)^2
        scale = std^2/mean
    """
    shape = (mean / std) ** 2
    scale = std ** 2 / mean

    def ppf(probability):
        return gamma_distribution.ppf(
            probability,
            a=shape,
            scale=scale,
        )

    def sampler(rng, n):
        return rng.gamma(shape=shape, scale=scale, size=n)

    def shortage(q):
        if q <= 0.0:
            return mean - q

        probability_tail = gammaincc(shape, q / scale)
        first_moment_tail = mean * gammaincc(
            shape + 1.0,
            q / scale,
        )

        return first_moment_tail - q * probability_tail

    return DemandDistribution(
        f"Gamma(mean={mean:g}, std={std:g})",
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_beta(alpha_shape, beta_shape, mean=50.0, std=10.0):
    """
    Shift and scale a standard Beta(alpha,beta) distribution so that it has
    the requested mean and standard deviation.
    """
    standard_mean = alpha_shape / (alpha_shape + beta_shape)

    standard_variance = (
        alpha_shape * beta_shape
        / (
            (alpha_shape + beta_shape) ** 2
            * (alpha_shape + beta_shape + 1.0)
        )
    )

    scale = std / np.sqrt(standard_variance)
    lower = mean - scale * standard_mean
    upper = lower + scale

    def ppf(probability):
        return beta_distribution.ppf(
            probability,
            alpha_shape,
            beta_shape,
            loc=lower,
            scale=scale,
        )

    def sampler(rng, n):
        standard_sample = rng.beta(
            alpha_shape,
            beta_shape,
            size=n,
        )
        return lower + scale * standard_sample

    def shortage(q):
        if q <= lower:
            return mean - q
        if q >= upper:
            return 0.0

        y = (q - lower) / scale

        tail_probability = 1.0 - betainc(
            alpha_shape,
            beta_shape,
            y,
        )

        standard_first_moment_tail = (
            alpha_shape
            / (alpha_shape + beta_shape)
            * (
                1.0
                - betainc(
                    alpha_shape + 1.0,
                    beta_shape,
                    y,
                )
            )
        )

        return (
            (lower - q) * tail_probability
            + scale * standard_first_moment_tail
        )

    return DemandDistribution(
        (
            f"Beta(alpha={alpha_shape:g}, beta={beta_shape:g}, "
            f"mean={mean:g}, std={std:g})"
        ),
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_triangular(left, mode, right, label=""):
    """
    Triangular distribution using the actual left, mode, and right values.

    The paper does not print these three values directly.
    They are recovered from the reported mean/std and the geometry in Figure 2.
    """
    left = float(left)
    mode = float(mode)
    right = float(right)

    mean = (left + mode + right) / 3.0
    variance = (
        left ** 2
        + mode ** 2
        + right ** 2
        - left * mode
        - left * right
        - mode * right
    ) / 18.0
    std = np.sqrt(variance)

    c = (mode - left) / (right - left)

    def ppf(probability):
        return triang.ppf(
            probability,
            c=c,
            loc=left,
            scale=right - left,
        )

    def sampler(rng, n):
        return rng.triangular(
            left=left,
            mode=mode,
            right=right,
            size=n,
        )

    def shortage(q):
        if q <= left:
            return mean - q
        if q >= right:
            return 0.0

        if q < mode:
            left_part = (
                (mode - q)
                - (
                    (mode - left) ** 3
                    - (q - left) ** 3
                )
                / (
                    3.0
                    * (right - left)
                    * (mode - left)
                )
            )

            right_part = (
                (right - mode) ** 2
                / (3.0 * (right - left))
            )

            return left_part + right_part

        return (
            (right - q) ** 3
            / (
                3.0
                * (right - left)
                * (right - mode)
            )
        )

    return DemandDistribution(
        (
            f"Triangular({label}, left={left:.4f}, "
            f"mode={mode:.4f}, right={right:.4f})"
        ),
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )

def make_exponential(mean=50.0):
    std = mean

    def ppf(probability):
        return -mean * np.log(1.0 - probability)

    def sampler(rng, n):
        return rng.exponential(mean, n)

    def shortage(q):
        if q <= 0.0:
            return mean - q
        return mean * np.exp(-q / mean)

    return DemandDistribution(
        f"Exponential(mean={mean:g}, std={std:g})",
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


def make_reverse_shifted_exponential(mean=350.0, std=50.0):
    """
    D = upper_point - X, where X ~ Exponential(mean=std).

    The paper uses upper_point=400, producing mean 350 and std 50.
    """
    upper_point = mean + std

    def ppf(probability):
        return upper_point + std * np.log(probability)

    def sampler(rng, n):
        return upper_point - rng.exponential(std, n)

    def shortage(q):
        if q >= upper_point:
            return 0.0

        distance = upper_point - q

        return (
            distance
            - std
            + std * np.exp(-distance / std)
        )

    return DemandDistribution(
        (
            f"Reverse Shifted Exponential("
            f"mean={mean:g}, std={std:g})"
        ),
        mean,
        std,
        ppf,
        sampler,
        shortage,
    )


# 5. BUILD BOTH COMPLETE GENERAL-COST PROBLEM SETS

def build_cases():
    """Full factorial reconstruction of BOTH problem sets in paper Section 3.2.

    This is the published design, not the author's unavailable raw output data.
    Normal IDs 1..405; Exponential/RSE IDs 406..810.
    """
    cases = {}
    grid = list(product([25.0, 50.0, 100.0], [0.2, 0.4, 0.6, 0.8, 1.0],
                        [3.0, 4.0, 6.0], [0.0, 1.0, 1.5], [0.5, 1.0, 2.0]))
    for group_index, group in enumerate(["Normal", "Exponential_RSE"]):
        for index, (mean_2, cv_2, price_2, salvage_2, resource_2) in enumerate(grid, 1):
            number = group_index * 405 + index
            d1 = make_normal(50.0, 20.0) if group == "Normal" else make_exponential(50.0)
            d2 = (make_normal(mean_2, mean_2 * cv_2) if group == "Normal"
                  else make_reverse_shifted_exponential(mean_2, mean_2 * cv_2))
            d1.family = "normal" if group == "Normal" else "exponential"
            d2.family = "normal" if group == "Normal" else "reverse_exponential"
            cases[number] = {
                "case_number": number, "case_label": f"{'N' if group == 'Normal' else 'E'}{index:03d}",
                "case_name": f"{group} {index:03d}: mu2={mean_2:g}, CV2={cv_2:g}, p2={price_2:g}, s2={salvage_2:g}, v2={resource_2:g}",
                "demand_group": group, "mean_2": mean_2, "cv_2": cv_2,
                "product_1": d1, "product_2": d2,
                "price": np.array([4.0, price_2]), "cost": np.array([2.0, 2.0]),
                "salvage": np.array([1.0, salvage_2]), "resource": np.array([1.0, resource_2]),
                "negative_policy": "repair_nonnegative",
            }
    return cases


def distribution_cdf(distribution, q):
    if distribution.family == "normal":
        return float(norm.cdf(q, loc=distribution.mean, scale=distribution.std))
    if distribution.family == "exponential":
        return float(-np.expm1(-max(q, 0.0) / distribution.mean))
    upper = distribution.mean + distribution.std
    return float(np.exp((q - upper) / distribution.std)) if q < upper else 1.0


def negative_part_mean(distribution):
    """E[max(-D,0)], for consistent censored-demand profit diagnostics."""
    if distribution.family == "normal":
        z = distribution.mean / distribution.std
        return float(distribution.std * norm.pdf(z) - distribution.mean * norm.cdf(-z))
    if distribution.family == "exponential":
        return 0.0
    return float(distribution.std * np.exp(-(distribution.mean + distribution.std) / distribution.std))


def activate_case(case):
    """Set case-specific costs before using the original algorithm functions."""
    global PRICE, COST, SALVAGE, RESOURCE, CRITICAL_RATIO, BETA_COST
    PRICE, COST = case["price"].copy(), case["cost"].copy()
    SALVAGE, RESOURCE = case["salvage"].copy(), case["resource"].copy()
    CRITICAL_RATIO = (PRICE - COST) / (PRICE - SALVAGE)
    BETA_COST = RESOURCE / (PRICE - SALVAGE)


# 6. ANALYTICAL PROFIT

def analytical_profit_item(distribution, product, q):
    """
    Expected profit for one product.

    E[min(D,Q)] = E[D] - E[(D-Q)+]
    E[(Q-D)+]   = Q - E[min(D,Q)]
    """
    shortage = distribution.expected_shortage(q)
    sold = distribution.mean - shortage
    leftover = q - sold

    return float(
        PRICE[product] * sold
        + SALVAGE[product] * leftover
        - SHORTAGE_PENALTY[product] * shortage
        - COST[product] * q
    )


def analytical_total_profit(case, q):
    q = np.asarray(q, dtype=float)

    return float(
        analytical_profit_item(case["product_1"], 0, q[0])
        + analytical_profit_item(case["product_2"], 1, q[1])
    )


# 7. CAPACITY GENERATION

def capacity_from_service_level(case, service_level):
    """C(kappa) = kappa * sum_i v_i Q_i^u; kappa is not a fractile."""
    distributions = [case["product_1"], case["product_2"]]
    q_u = np.array([max(d.ppf(CRITICAL_RATIO[i]), 0.0)
                    for i, d in enumerate(distributions)])
    return float(service_level * np.dot(RESOURCE, q_u))


# 8. EXACT LAMBDA

def exact_orders_given_lambda(case, lambda_value):
    q = np.zeros(2)

    distributions = [
        case["product_1"],
        case["product_2"],
    ]

    for i in range(2):
        adjusted_probability = (
            CRITICAL_RATIO[i]
            - lambda_value * BETA_COST[i]
        )

        q[i] = max(
            distributions[i].ppf(adjusted_probability),
            0.0,
        )

    return q


def solve_exact_lambda(
    case,
    capacity,
    tolerance=1e-10,
    max_iterations=200,
):
    distributions = [
        case["product_1"],
        case["product_2"],
    ]

    q_unconstrained = np.array([
        distributions[i].ppf(CRITICAL_RATIO[i])
        for i in range(2)
    ])

    q_unconstrained = np.maximum(q_unconstrained, 0.0)

    if np.dot(RESOURCE, q_unconstrained) <= capacity + tolerance:
        return q_unconstrained, 0.0

    low = 0.0
    high = 1.0

    while (
        np.dot(
            RESOURCE,
            exact_orders_given_lambda(case, high),
        )
        > capacity
    ):
        high *= 2.0

        if high > 1e8:
            raise RuntimeError(
                f"Could not bracket lambda in Case "
                f"{case['case_number']}."
            )

    middle = None
    q_middle = None

    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        q_middle = exact_orders_given_lambda(case, middle)
        used = float(np.dot(RESOURCE, q_middle))

        if abs(used - capacity) <= tolerance:
            break

        if used > capacity:
            low = middle
        else:
            high = middle

    return q_middle, float(middle)


# 9. H1, H2, H3

def apply_paper_negative_policy(case, q, capacity):
    """Repair negative quantities while preserving the capacity allocation.

    The four screenshot cases do not need this repair for H1/H2/H3.
    Raw heuristic quantities are nevertheless saved in diagnostics.
    """
    q = np.asarray(q, dtype=float).copy()
    if q[0] < 0.0:
        q[:] = [0.0, capacity / RESOURCE[1]]
    elif q[1] < 0.0:
        q[:] = [capacity / RESOURCE[0], 0.0]
    return q


def solve_h1(case, capacity):
    mean = np.array([
        case["product_1"].mean,
        case["product_2"].mean,
    ])

    std = np.array([
        case["product_1"].std,
        case["product_2"].std,
    ])

    denominator = float(np.sum(RESOURCE * std))
    capacity_gap = float(
        capacity - np.dot(RESOURCE, mean)
    )

    q = mean + std / denominator * capacity_gap

    return apply_paper_negative_policy(case, q, capacity)


def solve_h2(case, capacity):
    distributions = [
        case["product_1"],
        case["product_2"],
    ]

    std = np.array([
        distributions[0].std,
        distributions[1].std,
    ])

    q_unconstrained = np.array([
        distributions[i].ppf(CRITICAL_RATIO[i])
        for i in range(2)
    ])

    denominator = float(
        np.sum(RESOURCE * BETA_COST * std)
    )

    capacity_gap = float(
        capacity - np.dot(RESOURCE, q_unconstrained)
    )

    q = (
        q_unconstrained
        + BETA_COST * std / denominator * capacity_gap
    )

    return apply_paper_negative_policy(case, q, capacity)


def solve_h3(case, capacity):
    mean = np.array([
        case["product_1"].mean,
        case["product_2"].mean,
    ])

    std = np.array([
        case["product_1"].std,
        case["product_2"].std,
    ])

    denominator = float(
        np.sum(RESOURCE * BETA_COST * std)
    )

    capacity_gap = float(
        capacity - np.dot(RESOURCE, mean)
    )

    q = (
        mean
        + BETA_COST * std / denominator * capacity_gap
    )

    return apply_paper_negative_policy(case, q, capacity)


def average_cpu(function, repeats):
    result = None
    start = time.perf_counter()

    for _ in range(repeats):
        result = function()

    elapsed = (time.perf_counter() - start) / repeats

    return result, float(elapsed)


# 10. SIMULATION / SAA-LP

def generate_case_demands(case, rng, n):
    d1 = case["product_1"].sample(rng, n)
    d2 = case["product_2"].sample(rng, n)

    # The LP uses nonnegative demand balances.
    # Normal and reverse-shifted exponential distributions can generate a
    # very small number of negative observations, so simulation clips them.
    d1 = np.maximum(d1, 0.0)
    d2 = np.maximum(d2, 0.0)

    return d1, d2


def prepare_saa_lp(demand_1, demand_2):
    """Exact expected-profit envelopes, with only q1,q2,t1,t2 as variables.

    For sorted demand and j observations below q:
      profit_i(q) = [(p-c) - (p-s)*j/N]*q + (p-s)*sum(d[:j])/N.
    The minimum of these affine pieces is the sample-average profit.
    Maximizing t1+t2 subject to all pieces is equivalent to the recourse LP.
    """
    n = len(demand_1)
    row_indices, col_indices, coefficients, rhs = [], [], [], []
    for i, demand in enumerate([demand_1, demand_2]):
        ordered = np.sort(demand)
        slopes = PRICE[i] - COST[i] - (PRICE[i] - SALVAGE[i]) * np.arange(n + 1) / n
        intercepts = (PRICE[i] - SALVAGE[i]) * np.r_[0.0, np.cumsum(ordered)] / n
        rows = np.arange(i * (n + 1), (i + 1) * (n + 1))
        row_indices.extend(np.repeat(rows, 2))
        col_indices.extend(np.tile([i, i + 2], n + 1))
        coefficients.extend(np.column_stack([-slopes, np.ones(n + 1)]).ravel())
        rhs.extend(intercepts)
    row_indices.extend([2 * (n + 1)] * 2)
    col_indices.extend([0, 1])
    coefficients.extend(RESOURCE)
    rhs.append(0.0)
    matrix = csr_matrix((coefficients, (row_indices, col_indices)),
                        shape=(2 * (n + 1) + 1, 4))
    return matrix, np.asarray(rhs, dtype=float)


def solve_saa_lp(demand_1, demand_2, capacity, solver=None, prepared=None):
    """Solve the scenario LP and retain its capacity dual and solver diagnostics."""
    start = time.perf_counter()
    matrix, rhs = prepared if prepared is not None else prepare_saa_lp(demand_1, demand_2)
    rhs = rhs.copy()
    rhs[-1] = capacity
    solution = linprog(
        [0.0, 0.0, -1.0, -1.0], A_ub=matrix, b_ub=rhs,
        bounds=[(0.0, None), (0.0, None), (None, None), (None, None)],
        method="highs", options={"dual_feasibility_tolerance": 1e-8,
                                 "primal_feasibility_tolerance": 1e-8},
    )
    if not solution.success:
        raise RuntimeError(f"SAA LP failed: {solution.message}")
    elapsed = time.perf_counter() - start
    diagnostics = {
        "CapacityDual": float(-solution.ineqlin.marginals[-1]),
        "SolverStatus": int(solution.status), "SolverMessage": solution.message,
        "SolverIterations": int(solution.nit), "LPVariableCount": 4,
        "LPConstraintCount": int(len(rhs)),
    }
    arrays = {"x": solution.x, "inequality_dual": solution.ineqlin.marginals,
              "inequality_count": np.array(len(rhs)),
              "lower_dual": solution.lower.marginals,
              "upper_dual": solution.upper.marginals}
    dual = arrays.pop("inequality_dual")
    arrays["inequality_dual_indices"] = np.flatnonzero(dual)
    arrays["inequality_dual_values"] = dual[dual != 0]
    return float(-solution.fun), solution.x[:2], float(elapsed), diagnostics, arrays


def validation_profit(q, demand_1, demand_2):
    q = np.asarray(q, dtype=float)
    demand = np.vstack([demand_1, demand_2])

    sold = np.minimum(q[:, None], demand)
    leftover = np.maximum(q[:, None] - demand, 0.0)
    shortage = np.maximum(demand - q[:, None], 0.0)

    profit_by_scenario = np.sum(
        PRICE[:, None] * sold
        + SALVAGE[:, None] * leftover
        - SHORTAGE_PENALTY[:, None] * shortage
        - COST[:, None] * q[:, None],
        axis=0,
    )

    return float(np.mean(profit_by_scenario))


# 11. METRICS

def order_metrics(q, q_exact):
    q = np.asarray(q, dtype=float)
    q_exact = np.asarray(q_exact, dtype=float)

    error = q - q_exact
    absolute_error = np.abs(error)

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(absolute_error)

    mape = (
        np.mean(
            absolute_error
            / np.maximum(np.abs(q_exact), 1e-10)
        )
        * 100.0
    )

    wmape = (
        np.sum(absolute_error)
        / max(np.sum(np.abs(q_exact)), 1e-10)
        * 100.0
    )

    return (
        float(rmse),
        float(mae),
        float(mape),
        float(wmape),
    )


def build_result_row(
    case_number,
    case,
    service_level,
    capacity,
    method,
    q,
    final_profit,
    q_exact,
    exact_profit,
    cpu_time,
    train_profit=np.nan,
    validation=np.nan,
    q1_std=0.0,
    q2_std=0.0,
    final_profit_std=0.0,
):
    q = np.asarray(q, dtype=float)

    gap = exact_profit - final_profit
    gap_pct = 100.0 * gap / max(abs(exact_profit), 1e-10)

    rmse, mae, mape, wmape = order_metrics(
        q,
        q_exact,
    )

    capacity_used = float(np.dot(RESOURCE, q))

    return {
        "Case": case_number,
        "CaseName": case["case_name"],
        "Product1Distribution": case["product_1"].name,
        "Product2Distribution": case["product_2"].name,
        "ServiceLevel": service_level,
        "Capacity": capacity,
        "Method": method,
        "Q1": q[0],
        "Q2": q[1],
        "Q1Std": q1_std,
        "Q2Std": q2_std,
        "CapacityUsed": capacity_used,
        "CapacityUtilization": capacity_used / capacity,
        "TrainingProfit": train_profit,
        "ValidationProfit": validation,
        "FinalProfit": final_profit,
        "FinalProfitStd": final_profit_std,
        "ProfitGap": gap,
        "ProfitGapPct": gap_pct,
        "Q_RMSE": rmse,
        "Q_MAE": mae,
        "Q_MAPE": mape,
        "Q_WMAPE": wmape,
        "CPUTimeSec": cpu_time,
    }


# 12. MAIN EXPERIMENT

def write_json_atomic(path, data):
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, (np.integer, np.floating)):
            obj = obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_npz_atomic(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def initialize_run(cases):
    """Isolate incompatible runs, and preserve source/configuration/environment."""
    global OUTPUT_FOLDER
    import scipy
    configuration = {
        "schema": "full-general-piecewise-v1", "mode": RUN_MODE,
        "selected_cases": SELECTED_CASES, "capacity_fractions": CAPACITY_FRACTIONS,
        "scenario_size": SAA_N, "replications": SAA_REPLICATIONS,
        "validation_size": VALIDATION_N, "train_seed": TRAIN_SEED,
        "validation_seed": VALIDATION_SEED,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "solver": SOLVER_NAME, "negative_demand_policy": "censor_at_zero",
        "profit_benchmark": "original distribution; gap normalized by abs(exact profit); censored profit also saved",
        "timing_scope": "SAA CPUTimeSec includes solve; preprocessing recorded separately",
        "cases": [{k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in case.items() if not k.startswith("product_")}
                  for case in cases.values()],
    }
    # Serialize arrays before hashing; the complete configuration is also stored.
    encoded = json.dumps(configuration, default=lambda x: x.tolist(), sort_keys=True)
    run_id = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    OUTPUT_FOLDER = OUTPUT_FOLDER / run_id
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    for name in ["samples", "checkpoints", "solver_arrays"]:
        (OUTPUT_FOLDER / name).mkdir(exist_ok=True)
    manifest_path = OUTPUT_FOLDER / "run_manifest.json"
    if not manifest_path.exists():
        write_json_atomic(manifest_path, {
            "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "python_executable": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "pandas_version": pd.__version__,
            "matplotlib_version": matplotlib.__version__,
        })
        shutil.copyfile(__file__, OUTPUT_FOLDER / "source_snapshot.py")
    return run_id


def load_case_samples(case_number, case):
    """Save actual draws, not just seeds, before beginning any LP solves."""
    path = OUTPUT_FOLDER / "samples" / f"case_{case['case_label']}.npz"
    if not path.exists():
        train_rng = np.random.default_rng(TRAIN_SEED + case_number * 1000)
        validation_rng = np.random.default_rng(VALIDATION_SEED + case_number * 1000)
        training_raw = np.stack([
            np.column_stack([case['product_1'].sample(train_rng, SAA_N),
                             case['product_2'].sample(train_rng, SAA_N)])
            for _ in range(SAA_REPLICATIONS)
        ])
        validation_raw = np.column_stack([
            case['product_1'].sample(validation_rng, VALIDATION_N),
            case['product_2'].sample(validation_rng, VALIDATION_N),
        ])
        save_npz_atomic(path, training_raw=training_raw,
                        training=np.maximum(training_raw, 0.0),
                        validation_raw=validation_raw,
                        validation=np.maximum(validation_raw, 0.0))
    with np.load(path) as data:
        return data['training'].copy(), data['validation'].copy()


def censored_profit_adjustment(case):
    negative_tail = np.array([negative_part_mean(case['product_1']),
                              negative_part_mean(case['product_2'])])
    return float(np.dot(PRICE - SALVAGE, negative_tail))


def add_general_diagnostics(row, case, q, q_exact, lambda_star):
    distributions = [case['product_1'], case['product_2']]
    row['CaseLabel'] = case['case_label']
    row['DemandGroup'] = case['demand_group']
    row['CapacityFraction'] = row.pop('ServiceLevel')
    row['ExactLambda'] = lambda_star
    row['LambdaAdjustedFractile1'] = max(float(CRITICAL_RATIO[0] - lambda_star * BETA_COST[0]), 0.0)
    row['LambdaAdjustedFractile2'] = max(float(CRITICAL_RATIO[1] - lambda_star * BETA_COST[1]), 0.0)
    row['F1Q1'] = distribution_cdf(distributions[0], q[0])
    row['F2Q2'] = distribution_cdf(distributions[1], q[1])
    row['CensoredF1Q1'] = row['F1Q1']
    row['CensoredF2Q2'] = row['F2Q2']
    row['CensoredAnalyticalProfit'] = row['FinalProfit'] + censored_profit_adjustment(case)
    row['ProfitBenchmark'] = 'original_unbounded_distribution'
    row['RawExactProfitNonpositive'] = bool(analytical_total_profit(case, q_exact) <= 0.0)
    row['CapacitySlack'] = row['Capacity'] - float(np.dot(RESOURCE, q))
    row['Feasible'] = bool(np.all(np.asarray(q) >= -1e-7) and row['CapacitySlack'] >= -1e-7)
    row['ZeroExactQuantityCount'] = int(np.sum(np.asarray(q_exact) <= 1e-8))
    # MAPE is undefined when an exact quantity is zero; avoid artificial 1e10 errors.
    if np.any(np.asarray(q_exact) <= 1e-8):
        row['Q_MAPE'] = np.nan
    for i in range(2):
        j = i + 1
        shortage = distributions[i].expected_shortage(q[i])
        sold = distributions[i].mean - shortage
        row[f'OriginalAnalyticalProfitItem{j}'] = analytical_profit_item(distributions[i], i, q[i])
        row[f'OriginalExpectedShortageItem{j}'] = shortage
        row[f'OriginalExpectedSalesItem{j}'] = sold
        row[f'OriginalExpectedLeftoverItem{j}'] = q[i] - sold
        row[f'Price{j}'], row[f'Cost{j}'], row[f'Salvage{j}'] = PRICE[i], COST[i], SALVAGE[i]
        row[f'Resource{j}'], row[f'CriticalRatio{j}'] = RESOURCE[i], CRITICAL_RATIO[i]
        row[f'QU{j}'] = max(distributions[i].ppf(CRITICAL_RATIO[i]), 0.0)
        row[f'ExactQ{j}'] = q_exact[i]
    return row


def run_all_cases(cases):
    all_rows, replication_rows, sample_stats = [], [], []
    cache_hits, solves = 0, 0
    start = time.perf_counter()
    for case_number in SELECTED_CASES:
        case = cases[case_number]
        activate_case(case)
        print(f"\n{case['case_name']}", flush=True)
        training, validation = (None, None)
        prepared_cache = {}
        if RUN_SAA:
            training, validation = load_case_samples(case_number, case)
            with np.load(OUTPUT_FOLDER / 'samples' / f"case_{case['case_label']}.npz") as data:
                for name in ['training_raw', 'training', 'validation_raw', 'validation']:
                    sample = data[name]
                    sample_stats.append({'Case': case['case_label'], 'Sample': name,
                        'CountPerItem': sample.size // 2,
                        'Mean1': float(sample[..., 0].mean()), 'Mean2': float(sample[..., 1].mean()),
                        'Std1': float(sample[..., 0].std()), 'Std2': float(sample[..., 1].std()),
                        'NegativeCount': int((sample < 0).sum()), 'ZeroCount': int((sample == 0).sum())})
        for fraction in CAPACITY_FRACTIONS:
            capacity = capacity_from_service_level(case, fraction)
            prefix = f"case_{case['case_label']}_k{int(round(100 * fraction)):03d}"
            deterministic_path = OUTPUT_FOLDER / 'checkpoints' / f'{prefix}_deterministic.json'
            if deterministic_path.exists():
                saved = json.loads(deterministic_path.read_text(encoding='utf-8'))
                deterministic = saved['rows']
                q_exact, lambda_star = np.asarray(saved['q_exact']), saved['lambda']
                cache_hits += 1
            else:
                (q_exact, lambda_star), cpu = average_cpu(
                    lambda: solve_exact_lambda(case, capacity), LAMBDA_TIME_REPEATS)
                exact_profit = analytical_total_profit(case, q_exact)
                deterministic = []
                for method, function in [('Exact Lambda', None), ('H1', solve_h1),
                                         ('H2', solve_h2), ('H3', solve_h3)]:
                    q, timing = (q_exact, cpu) if function is None else average_cpu(
                        lambda: function(case, capacity), HEURISTIC_TIME_REPEATS)
                    row = build_result_row(case_number, case, fraction, capacity, method, q,
                        analytical_total_profit(case, q), q_exact, exact_profit, timing)
                    add_general_diagnostics(row, case, q, q_exact, lambda_star)
                    if RUN_SAA:
                        row['ValidationProfit'] = validation_profit(q, *validation.T)
                    means = np.array([case['product_1'].mean, case['product_2'].mean])
                    stds = np.array([case['product_1'].std, case['product_2'].std])
                    if method == 'H1':
                        raw_q = means + stds / np.dot(RESOURCE, stds) * (capacity - np.dot(RESOURCE, means))
                    elif method == 'H2':
                        qu = np.array([case[f'product_{i+1}'].ppf(CRITICAL_RATIO[i]) for i in range(2)])
                        raw_q = qu + BETA_COST * stds / np.dot(RESOURCE, BETA_COST * stds) * (capacity - np.dot(RESOURCE, qu))
                    elif method == 'H3':
                        raw_q = means + BETA_COST * stds / np.dot(RESOURCE, BETA_COST * stds) * (capacity - np.dot(RESOURCE, means))
                    else:
                        raw_q = q
                    row['RawQ1'], row['RawQ2'] = raw_q
                    row['HeuristicRepaired'] = bool(np.any(raw_q < 0))
                    deterministic.append(row)
                write_json_atomic(deterministic_path, {
                    'q_exact': q_exact, 'lambda': lambda_star, 'rows': deterministic})
            all_rows.extend(deterministic)
            exact_profit = deterministic[0]['FinalProfit']
            current_reps = []
            if RUN_SAA:
                for replication in range(1, SAA_REPLICATIONS + 1):
                    checkpoint = OUTPUT_FOLDER / 'checkpoints' / f'{prefix}_rep{replication:03d}.json'
                    expected_arrays = OUTPUT_FOLDER / 'solver_arrays' / f'{prefix}_rep{replication:03d}.npz'
                    if checkpoint.exists() and expected_arrays.exists():
                        rep = json.loads(checkpoint.read_text(encoding='utf-8'))
                        cache_hits += 1
                    else:
                        sample = training[replication - 1]
                        if replication not in prepared_cache:
                            prep_start = time.perf_counter()
                            prepared_cache[replication] = (prepare_saa_lp(*sample.T),
                                                          time.perf_counter() - prep_start)
                        prepared, prep_cpu = prepared_cache[replication]
                        train_profit, q, cpu, diagnostics, arrays = solve_saa_lp(
                            *sample.T, capacity, prepared=prepared)
                        rep = build_result_row(case_number, case, fraction, capacity, 'SAA', q,
                            analytical_total_profit(case, q), q_exact, exact_profit, cpu,
                            train_profit=train_profit, validation=validation_profit(q, *validation.T))
                        add_general_diagnostics(rep, case, q, q_exact, lambda_star)
                        rep.update(diagnostics)
                        rep.update({'Replication': replication, 'ScenarioSize': SAA_N,
                                    'LPPreprocessingTimeSec': prep_cpu,
                                    'CapacityDualMinusExactLambda': diagnostics['CapacityDual'] - lambda_star,
                                    'SampleFile': f"samples/case_{case['case_label']}.npz",
                                    'TrainingSampleIndex': replication - 1})
                        profit_vector = np.sum(PRICE[:, None] * np.minimum(q[:, None], validation.T)
                            + SALVAGE[:, None] * np.maximum(q[:, None] - validation.T, 0.0)
                            - COST[:, None] * q[:, None], axis=0)
                        rep['ValidationProfitStd'] = float(profit_vector.std(ddof=1))
                        rep['ValidationProfitSE'] = rep['ValidationProfitStd'] / np.sqrt(VALIDATION_N)
                        rep['TrainingObjectiveCheckError'] = train_profit - validation_profit(q, *sample.T)
                        if not rep['Feasible'] or abs(rep['TrainingObjectiveCheckError']) > 1e-5:
                            raise RuntimeError(f'LP verification failed: {prefix}, rep {replication}')
                        if rep['ProfitGapPct'] < -1e-6:
                            raise RuntimeError('SAA outperforms the claimed exact optimum; check formulation.')
                        # Solver vectors are saved first; JSON is the completion marker.
                        arrays_path = OUTPUT_FOLDER / 'solver_arrays' / f'{prefix}_rep{replication:03d}.npz'
                        save_npz_atomic(arrays_path, **arrays)
                        rep['SolverArraysFile'] = str(arrays_path.relative_to(OUTPUT_FOLDER))
                        write_json_atomic(checkpoint, rep)
                        solves += 1
                    current_reps.append(rep)
                replication_rows.extend(current_reps)
                frame = pd.DataFrame(current_reps)
                q_mean = frame[['Q1', 'Q2']].mean().to_numpy()
                row = build_result_row(case_number, case, fraction, capacity, 'SAA', q_mean,
                    frame['FinalProfit'].mean(), q_exact, exact_profit, frame['CPUTimeSec'].mean(),
                    train_profit=frame['TrainingProfit'].mean(), validation=frame['ValidationProfit'].mean(),
                    q1_std=frame['Q1'].std(ddof=1) if len(frame) > 1 else 0.0,
                    q2_std=frame['Q2'].std(ddof=1) if len(frame) > 1 else 0.0,
                    final_profit_std=frame['FinalProfit'].std(ddof=1) if len(frame) > 1 else 0.0)
                add_general_diagnostics(row, case, q_mean, q_exact, lambda_star)
                # FinalProfit is mean profit of individual decisions, not profit of mean Q.
                for key in ['Q_RMSE', 'Q_MAE', 'Q_MAPE', 'Q_WMAPE', 'CensoredAnalyticalProfit',
                            'OriginalAnalyticalProfitItem1', 'OriginalAnalyticalProfitItem2',
                            'OriginalExpectedShortageItem1', 'OriginalExpectedShortageItem2',
                            'OriginalExpectedSalesItem1', 'OriginalExpectedSalesItem2',
                            'OriginalExpectedLeftoverItem1', 'OriginalExpectedLeftoverItem2',
                            'F1Q1', 'F2Q2', 'CensoredF1Q1', 'CensoredF2Q2', 'CapacityDual',
                            'CapacityDualMinusExactLambda']:
                    row[key] = frame[key].mean()
                row['AnalyticalProfitAtMeanQ'] = analytical_total_profit(case, q_mean)
                row['F1AtMeanQ1'] = distribution_cdf(case['product_1'], q_mean[0])
                row['F2AtMeanQ2'] = distribution_cdf(case['product_2'], q_mean[1])
                all_rows.append(row)
            print(f"Case {case['case_label']} | kappa={fraction:.2f} | C={capacity:.4f} | "
                  f"lambda={lambda_star:.6f} | Q*=({q_exact[0]:.4f}, {q_exact[1]:.4f}) | "
                  f"new LPs={solves}, cache hits={cache_hits}", flush=True)
    results = pd.DataFrame(all_rows)
    results['Method'] = pd.Categorical(results['Method'], categories=METHOD_ORDER, ordered=True)
    results = results.sort_values(['Case', 'CapacityFraction', 'Method']).reset_index(drop=True)
    pd.DataFrame(sample_stats, columns=['Case', 'Sample', 'CountPerItem', 'Mean1', 'Mean2',
                                      'Std1', 'Std2', 'NegativeCount', 'ZeroCount']).to_csv(
        OUTPUT_FOLDER / 'sample_statistics.csv', index=False)
    write_json_atomic(OUTPUT_FOLDER / 'execution_status.json', {
        'status': 'computed', 'new_lp_solves': solves, 'cache_hits': cache_hits,
        'elapsed_seconds': time.perf_counter() - start,
        'completed_utc': datetime.now(timezone.utc).isoformat()})
    return results, pd.DataFrame(replication_rows)


# 13. SUMMARIES

def make_case_method_summary(results):
    summary = (
        results
        .groupby(
            ["Case", "Method"],
            observed=True,
        )
        .agg(
            CapacityCases=("Capacity", "count"),
            AverageFinalProfit=("FinalProfit", "mean"),
            AverageProfitGapPct=("ProfitGapPct", "mean"),
            MaximumProfitGapPct=("ProfitGapPct", "max"),
            AverageCPUTimeSec=("CPUTimeSec", "mean"),
            MaximumCPUTimeSec=("CPUTimeSec", "max"),
            AverageQ_RMSE=("Q_RMSE", "mean"),
            AverageQ_MAE=("Q_MAE", "mean"),
            AverageQ_MAPE=("Q_MAPE", "mean"),
            AverageQ_WMAPE=("Q_WMAPE", "mean"),
            AverageCapacityUtilization=(
                "CapacityUtilization",
                "mean",
            ),
            AverageFinalProfitStd=(
                "FinalProfitStd",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["Method"] = pd.Categorical(
        summary["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return summary.sort_values(
        ["Case", "Method"]
    ).reset_index(drop=True)


def make_overall_method_summary(case_method_summary):
    overall = (
        case_method_summary
        .groupby("Method", observed=True)
        .agg(
            NumberOfCases=("Case", "count"),
            MeanOfCaseAverageGapPct=(
                "AverageProfitGapPct",
                "mean",
            ),
            WorstCaseMaximumGapPct=(
                "MaximumProfitGapPct",
                "max",
            ),
            MeanCPUTimeSec=(
                "AverageCPUTimeSec",
                "mean",
            ),
            MeanOrderRMSE=(
                "AverageQ_RMSE",
                "mean",
            ),
            MeanOrderWMAPE=(
                "AverageQ_WMAPE",
                "mean",
            ),
        )
        .reset_index()
    )

    overall["Method"] = pd.Categorical(
        overall["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    return overall.sort_values("Method").reset_index(drop=True)


def save_results_to_excel(
    results,
    case_method_summary,
    overall_summary,
    paper_comparison,
    saa_details,
):
    """Save all experiment tables in one formatted Excel workbook."""
    excel_path = OUTPUT_FOLDER / EXCEL_FILENAME

    sheets = {
        "All Results": results,
        "Case Method Summary": case_method_summary,
        "Overall Summary": overall_summary,
        "Paper Comparison": paper_comparison,
    }

    if RUN_SAA and not saa_details.empty:
        sheets["SAA Replications"] = saa_details

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            export_data = dataframe.copy()

            # Excel does not need pandas' categorical metadata.
            for column in export_data.select_dtypes(
                include="category"
            ).columns:
                export_data[column] = export_data[column].astype(str)

            export_data.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                font.color = "FFFFFF"
                cell.font = font
                fill = copy(cell.fill)
                fill.fill_type = "solid"
                fill.fgColor = "1F4E78"
                cell.fill = fill

            for column_cells in worksheet.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells
                )
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 10),
                    28,
                )

            percent_columns = {
                "CapacityUtilization",
                "ProfitGapPct",
                "Q_MAPE",
                "Q_WMAPE",
                "AverageProfitGapPct",
                "MaximumProfitGapPct",
                "AverageQ_MAPE",
                "AverageQ_WMAPE",
                "AverageCapacityUtilization",
                "MeanOfCaseAverageGapPct",
                "WorstCaseMaximumGapPct",
                "MeanOrderWMAPE",
            }

            for cell in worksheet[1]:
                if cell.value in percent_columns:
                    # These columns already contain percentage-point values,
                    # except utilization, which is stored as a fraction.
                    number_format = (
                        "0.00%"
                        if "Utilization" in cell.value
                        else "0.0000"
                    )
                    for data_cell in worksheet[cell.column_letter][1:]:
                        data_cell.number_format = number_format

    return excel_path


# 14. PAPER TABLE 3 BENCHMARK

def export_input_parameters(cases):
    rows = []
    for number, case in cases.items():
        activate_case(case)
        qu = np.array([max(case[f'product_{i+1}'].ppf(CRITICAL_RATIO[i]), 0.0) for i in range(2)])
        for i in range(2):
            distribution = case[f'product_{i+1}']
            rows.append({'Case': number, 'CaseLabel': case['case_label'],
                'DemandGroup': case['demand_group'], 'SelectedForRun': number in SELECTED_CASES,
                'Item': i+1, 'Distribution': distribution.name, 'Mean': distribution.mean,
                'Std': distribution.std, 'CV': distribution.std/distribution.mean,
                'Price': PRICE[i], 'Cost': COST[i], 'Salvage': SALVAGE[i], 'Resource': RESOURCE[i],
                'CriticalRatio': CRITICAL_RATIO[i], 'QU': qu[i],
                'UnconstrainedCapacity': np.dot(RESOURCE, qu),
                'CapacityAtKappa040': .4*np.dot(RESOURCE, qu)})
    return pd.DataFrame(rows)


def write_data_guide(run_id):
    guide = f'''# Full Section 3.2 general-cost design, piecewise SAA

## Material Passport

Source: user-provided Erlebacher PDF, Section 3.2 (PDF page 13), Table 4 (page 14).
Task: user-authorized separate implementation using the full published design.
Material: reconstructed factorial design and generated samples, not the author's raw data.
Run ID: {run_id}; mode: {RUN_MODE}; selected combinations: {len(SELECTED_CASES)}.

## Published design

Each problem set has 3*5*3*3*3=405 combinations:
item 2 mean=[25,50,100], CV=[.2,.4,.6,.8,1], price=[3,4,6],
salvage=[0,1,1.5], resource=[.5,1,2]; unit cost=2.
Normal set: item 1 Normal(50,20), item 2 Normal(mu2,mu2*CV2).
Exponential set: item 1 Exponential(mean=std=50),
item 2 D=mu2+std2-X, X~Exponential(mean=std2), std2=mu2*CV2.
Item 1 cost structure: price=4,cost=2,salvage=1,resource=1 in both sets.
IDs 1..405=N001..N405; 406..810=E001..E405.

The paper says approximately 16 capacity levels from 40% to 100% of unconstrained
capacity. It does not list exact capacity grids or raw per-instance results.
This reconstruction uses exactly kappa=.40,.44,...,1.00 and C=kappa*sum(v*max(Q_u,0)).
It reconstructs both full parameter grids, without claiming exact Table 4 replication.
The original 13 distribution pairs belong to Section 3.1 and are not part of this grid.

## Sampling and benchmark conventions

SAA is an extension for comparison, not a method specified by the original author.
Current mode: N={SAA_N}, R={SAA_REPLICATIONS}, validation={VALIDATION_N}.
Full mode: 810*16=12960 instances; 259200 SAA solves at R=20.
Test mode selects 3 combinations per family, retains all 16 capacities, N=200,R=1,V=5000.
Use DEMAND_GROUP=normal/exponential/all and optional CASE_IDS=1,405,... to select runs.

Raw demand and max(D,0) samples are both saved. SAA/validation use max(D,0).
At high CV, negative mass is substantial and must not be described as negligible.
Original-distribution and censored analytical profits are both saved. Their difference
is constant with respect to nonnegative Q, preserving the optimizer, not profit ratios.
ProfitGapPct=100*(exact_profit-profit)/max(abs(exact_profit),1e-10).
RawExactProfitNonpositive flags problematic original-distribution profit denominators.
CensoredProfitGapPct is additionally exported for a nonnegative-demand comparison.
MAPE is missing at zero exact quantities; RMSE/MAE/WMAPE are saved.
H1/H2/H3 raw quantities are saved, alongside repaired nonnegative feasible decisions;
negative quantities are repaired by allocating all capacity to the other item with v accounted for.
This explicit repair policy may differ from unspecified details in the paper's grid study.

## Exact piecewise SAA LP and persistence

Variables q1,q2,t1,t2; all N+1 empirical affine profit pieces per item are retained.
HiGHS solves this LP via scipy.optimize.linprog. No scenario averaging of demand,
breakpoint subsampling or smoothing is used. CPUTimeSec covers solve; reusable
envelope preparation is separately recorded in LPPreprocessingTimeSec.

samples/*.npz: training_raw/training shape (R,N,2), validation_raw/validation shape (V,2).
checkpoints/*.json: each completed deterministic instance or SAA replication.
solver_arrays/*.npz: x=(q1,q2,t1,t2), bound duals, inequality_count and sparse
inequality_dual_indices/inequality_dual_values. Reconstruct full dual as zeros(count)
and assign values at saved indices. No nonzero dual values are discarded.
Inequality order: item 1 pieces j=0..N, item 2 pieces j=0..N, capacity last.
Capacity maximization dual is minus the last inequality minimization dual.
Slacks follow from saved samples, x, and prepare_saa_lp; scenario sales=min(q,d),
leftover=max(q-d,0), shortage=max(d-q,0). No new optimization is needed.

run_manifest.json, source_snapshot.py, execution_status.json retain configuration,
environment, original CPU timings and source hash. Incompatible runs are isolated.
Repeated identical runs read checkpoints and never replace original solve timings.
The sample files and sparse dual encoding limit redundant storage for the full grid.

all_results.csv and saa_replication_details.csv are full-detail exports;
all_810_input_parameters.csv lists BOTH grids even for a small selected test.
Excel contains summaries and parameters; large per-replication data stay in CSV/NPZ.
Group plots are saved without creating thousands of per-case figures.
Table4 comparisons are contextual; denominator, censoring, capacity-grid and
nonnegative-repair conventions are recorded rather than assumed to reproduce the paper.
'''
    (OUTPUT_FOLDER/'DATA_GUIDE.md').write_text(guide,encoding='utf-8')


def export_full_grid_results(results, saa_details, cases):
    parameters = export_input_parameters(cases)
    parameters.to_csv(OUTPUT_FOLDER/'all_810_input_parameters.csv',index=False)
    case_summary = make_case_method_summary(results)
    group_summary = results.groupby(['DemandGroup','Method'],observed=True).agg(
        Instances=('Capacity','count'), AverageProfitGapPct=('ProfitGapPct','mean'),
        MaximumProfitGapPct=('ProfitGapPct','max'),
        AverageCensoredProfitGapPct=('CensoredProfitGapPct','mean'),
        AverageCPUTimeSec=('CPUTimeSec','mean'), AverageRMSE=('Q_RMSE','mean'),
        NonpositiveOriginalExactProfitInstances=('RawExactProfitNonpositive','sum')).reset_index()
    paper_values = {'Normal':{'H1':(2.60,55.97),'H2':(.06,4.41),'H3':(1.80,23.95)},
                    'Exponential_RSE':{'H1':(5.77,45.13),'H2':(.28,28.71),'H3':(4.52,35.66)}}
    comparison = group_summary.copy()
    comparison['PaperAverageGapPct'] = [paper_values[g].get(str(m),(np.nan,np.nan))[0]
                                      for g,m in zip(comparison.DemandGroup,comparison.Method)]
    comparison['PaperMaximumGapPct'] = [paper_values[g].get(str(m),(np.nan,np.nan))[1]
                                      for g,m in zip(comparison.DemandGroup,comparison.Method)]
    comparison['ScopeNote'] = 'Published full design; explicit grid/censoring/repair conventions; not exact numeric reproduction'
    for name, frame in [('all_results',results),('saa_replication_details',saa_details),
                        ('case_method_summary',case_summary),('group_method_summary',group_summary),
                        ('table4_context',comparison)]:
        if not frame.empty:
            frame.to_csv(OUTPUT_FOLDER/f'{name}.csv',index=False)
    with pd.ExcelWriter(OUTPUT_FOLDER/EXCEL_FILENAME,engine='openpyxl') as writer:
        group_summary.to_excel(writer,sheet_name='Group Method Summary',index=False)
        comparison.to_excel(writer,sheet_name='Table4 Context',index=False)
        parameters.to_excel(writer,sheet_name='All 810 Parameters',index=False)
        case_summary.to_excel(writer,sheet_name='Case Method Summary',index=False)
    plot_dir = OUTPUT_FOLDER/'plots'
    plot_dir.mkdir(exist_ok=True)
    for group in results.DemandGroup.unique():
        frame = results[results.DemandGroup == group]
        for metric in ['ProfitGapPct','CensoredProfitGapPct']:
            plot_data = frame.groupby(['CapacityFraction','Method'],observed=True)[metric].mean().unstack()
            plot_data.plot(figsize=(9,5),marker='o',markersize=3)
            plt.xlabel('Capacity fraction kappa')
            plt.ylabel(f'Mean {metric} (%)')
            plt.title(f'{group}: {frame.Case.nunique()} combinations')
            plt.grid(alpha=.3)
            plt.tight_layout()
            plt.savefig(plot_dir/f'{group}_{metric}.png',dpi=180)
            if SHOW_PLOTS:
                plt.show()
            plt.close()
    return group_summary


def add_censored_gap(frame):
    if frame.empty:
        return
    correction = frame['CensoredAnalyticalProfit']-frame['FinalProfit']
    # exact raw profit = final profit + gap; correction applies to either decision.
    censored_exact = frame['FinalProfit']+frame['ProfitGap']+correction
    frame['CensoredExactProfit'] = censored_exact
    frame['CensoredProfitGapPct'] = 100*frame['ProfitGap']/np.maximum(np.abs(censored_exact),1e-10)


if __name__ == '__main__':
    all_cases = build_cases()
    assert len(all_cases) == 810
    assert sum(c['demand_group']=='Normal' for c in all_cases.values()) == 405
    run_id = initialize_run(all_cases)
    write_data_guide(run_id)
    # Save complete input design before any long-running solves begin.
    export_input_parameters(all_cases).to_csv(OUTPUT_FOLDER/'all_810_input_parameters.csv',index=False)
    print('FULL ERLEBACHER SECTION 3.2 DESIGN — PIECEWISE SAA',flush=True)
    print(f'Mode={RUN_MODE}; groups={DEMAND_GROUP}; combinations={len(SELECTED_CASES)}; '
          f'capacities={len(CAPACITY_FRACTIONS)}; N={SAA_N}; R={SAA_REPLICATIONS}',flush=True)
    print(f'Total SAA solves: {len(SELECTED_CASES)*len(CAPACITY_FRACTIONS)*SAA_REPLICATIONS}',flush=True)
    print(f'Results: {OUTPUT_FOLDER}',flush=True)
    results, saa_details = run_all_cases(all_cases)
    add_censored_gap(results)
    add_censored_gap(saa_details)
    expected = len(SELECTED_CASES)*len(CAPACITY_FRACTIONS)*(5 if RUN_SAA else 4)
    assert len(results) == expected
    assert results.Feasible.all()
    assert len(saa_details) == len(SELECTED_CASES)*len(CAPACITY_FRACTIONS)*SAA_REPLICATIONS
    assert results.ProfitGapPct.min() >= -1e-6
    group_summary = export_full_grid_results(results,saa_details,all_cases)
    verification = {'status':'passed','all_design_combinations':len(all_cases),
        'selected_combinations':len(SELECTED_CASES),'result_rows':len(results),
        'replication_rows':len(saa_details),'all_feasible':bool(results.Feasible.all()),
        'minimum_profit_gap_pct':float(results.ProfitGapPct.min()),
        'max_training_objective_error':float(saa_details.TrainingObjectiveCheckError.abs().max()) if RUN_SAA else None}
    write_json_atomic(OUTPUT_FOLDER/'verification.json',verification)
    status_path = OUTPUT_FOLDER/'execution_status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status['status'] = 'completed'
    write_json_atomic(status_path,status)
    print(group_summary.to_string(index=False))
    print(f'\nDone. Saved in: {OUTPUT_FOLDER}',flush=True)
