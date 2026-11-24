

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
SOLVER_NAME = "glpk"

SELECTED_CASES = list(range(1, 14))

TRAIN_SEED = 20
VALIDATION_SEED = 2026

OUTPUT_FOLDER = Path("all_13_cases_results")
EXCEL_FILENAME = "all_13_cases_results.xlsx"
SAVE_FILES = True
SHOW_PLOTS = True

# Repeated timing reduces timer noise for very fast deterministic methods.
LAMBDA_TIME_REPEATS = 5
HEURISTIC_TIME_REPEATS = 100

if RUN_MODE == "test":
    RUN_SAA = True
    SERVICE_LEVELS = np.array([0.10, 0.30, 0.50, 0.70])
    SAA_N = 200
    SAA_REPLICATIONS = 1
    VALIDATION_N = 5_000

elif RUN_MODE == "paper_check":
    RUN_SAA = False
    SERVICE_LEVELS = np.arange(10, 72, dtype=float) / 100.0
    SAA_N = 0
    SAA_REPLICATIONS = 0
    VALIDATION_N = 0

elif RUN_MODE == "full":
    RUN_SAA = True
    SERVICE_LEVELS = np.arange(10, 72, dtype=float) / 100.0
    SAA_N = 10_000
    SAA_REPLICATIONS = 20
    VALIDATION_N = 100_000

else:
    raise ValueError("RUN_MODE must be 'test', 'paper_check', or 'full'.")



# 2. COMMON COST PARAMETERS

NUMBER_OF_PRODUCTS = 2

PRICE = np.array([50.0, 50.0])
COST = np.array([25.0, 25.0])
SALVAGE = np.array([15.0, 15.0])
SHORTAGE_PENALTY = np.array([0.0, 0.0])
RESOURCE = np.array([1.0, 1.0])

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


# =============================================================================
# 5. BUILD THE 13 CASES
# =============================================================================

def build_cases():
    """
    negative_policy:
        "normal"     = no special negative-order adjustment is needed
        "allow"      = retain negative heuristic Q, matching Cases 6 and 12
        "repair_q1"  = if heuristic Q1<0, set Q1=0 and Q2=Capacity
    """
    cases = {
        1: {
            "product_1": make_normal(),
            "product_2": make_uniform(),
            "negative_policy": "normal",
        },
        2: {
            "product_1": make_normal(),
            "product_2": make_double_exponential(),
            "negative_policy": "normal",
        },
        3: {
            "product_1": make_uniform(),
            "product_2": make_double_exponential(),
            "negative_policy": "normal",
        },
        4: {
            "product_1": make_gamma(std=12.5),
            "product_2": make_gamma(std=7.5),
            "negative_policy": "normal",
        },
        5: {
            "product_1": make_gamma(std=15.0),
            "product_2": make_gamma(std=5.0),
            "negative_policy": "normal",
        },
        6: {
            "product_1": make_gamma(std=100.0),
            "product_2": make_gamma(std=10.0),
            "negative_policy": "allow",
        },
        7: {
            "product_1": make_gamma(std=100.0),
            "product_2": make_gamma(std=10.0),
            "negative_policy": "repair_q1",
        },
        8: {
            "product_1": make_beta(1.5, 5.0),
            "product_2": make_beta(5.0, 1.5),
            "negative_policy": "normal",
        },
        9: {
            "product_1": make_beta(1.5, 5.0),
            "product_2": make_beta(
                5.0,
                1.5,
                mean=85.0,
                std=10.0,
            ),
            "negative_policy": "normal",
        },
        10: {
            "product_1": make_triangular(
                31.4836,
                40.7418,
                77.7746,
                label="skewed left",
            ),
            "product_2": make_triangular(
                22.2254,
                59.2582,
                68.5164,
                label="skewed right",
            ),
            "negative_policy": "normal",
        },
        11: {
            "product_1": make_triangular(
                31.4836,
                40.7418,
                77.7746,
                label="skewed left",
            ),
            "product_2": make_triangular(
                31.4836,
                68.5164,
                77.7746,
                label="skewed right",
            ),
            "negative_policy": "normal",
        },
        12: {
            "product_1": make_exponential(mean=50.0),
            "product_2": make_reverse_shifted_exponential(
                mean=350.0,
                std=50.0,
            ),
            "negative_policy": "allow",
        },
        13: {
            "product_1": make_exponential(mean=50.0),
            "product_2": make_reverse_shifted_exponential(
                mean=350.0,
                std=50.0,
            ),
            "negative_policy": "repair_q1",
        },
    }

    for case_number, case in cases.items():
        case["case_number"] = case_number
        case["case_name"] = (
            f"Case {case_number}: "
            f"{case['product_1'].name} | "
            f"{case['product_2'].name}"
        )

    return cases


# =============================================================================
# 6. ANALYTICAL PROFIT
# =============================================================================

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
    """
    Because both items have the same cost structure, the exact constrained
    solution has the same service level for both products.

    Capacity at level k:
        C(k) = F1^(-1)(k) + F2^(-1)(k)
    """
    q1 = case["product_1"].ppf(service_level)
    q2 = case["product_2"].ppf(service_level)

    return float(q1 + q2)


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
    """
    Cases 6 and 12:
        keep negative heuristic quantities.

    Cases 7 and 13:
        if Q1 is negative, set Q1=0 and Q2=Capacity.
    """
    q = np.asarray(q, dtype=float).copy()

    if (
        case["negative_policy"] == "repair_q1"
        and q[0] < 0.0
    ):
        q[0] = 0.0
        q[1] = capacity

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


def average_computation_time(function, repeats):
    """Return one result and mean elapsed wall-clock seconds per decision."""
    result = None
    start = time.perf_counter()

    for _ in range(repeats):
        result = function()

    elapsed = (time.perf_counter() - start) / repeats

    return result, float(elapsed)


# =============================================================================
# 10. SIMULATION / SAA-LP
# =============================================================================

def generate_case_demands(case, rng, n):
    d1 = case["product_1"].sample(rng, n)
    d2 = case["product_2"].sample(rng, n)

    # The LP uses nonnegative demand balances.
    # Normal and reverse-shifted exponential distributions can generate a
    # very small number of negative observations, so simulation clips them.
    d1 = np.maximum(d1, 0.0)
    d2 = np.maximum(d2, 0.0)

    return d1, d2


def solve_saa_lp(
    demand_1,
    demand_2,
    capacity,
    solver,
):
    n = len(demand_1)

    if len(demand_2) != n:
        raise ValueError(
            "Both products must have the same number of scenarios."
        )

    probability = 1.0 / n

    demand_data = {}

    for s in range(n):
        demand_data[0, s] = float(demand_1[s])
        demand_data[1, s] = float(demand_2[s])

    start = time.perf_counter()

    model = ConcreteModel()
    model.I = RangeSet(0, 1)
    model.S = RangeSet(0, n - 1)

    model.demand = Param(
        model.I,
        model.S,
        initialize=demand_data,
        within=NonNegativeReals,
    )

    model.q = Var(model.I, within=NonNegativeReals)
    model.sold = Var(
        model.I,
        model.S,
        within=NonNegativeReals,
    )
    model.leftover = Var(
        model.I,
        model.S,
        within=NonNegativeReals,
    )
    model.shortage = Var(
        model.I,
        model.S,
        within=NonNegativeReals,
    )

    def inventory_balance(model, i, s):
        return (
            model.q[i]
            == model.sold[i, s]
            + model.leftover[i, s]
        )

    def demand_balance(model, i, s):
        return (
            model.demand[i, s]
            == model.sold[i, s]
            + model.shortage[i, s]
        )

    model.InventoryBalance = Constraint(
        model.I,
        model.S,
        rule=inventory_balance,
    )

    model.DemandBalance = Constraint(
        model.I,
        model.S,
        rule=demand_balance,
    )

    model.Capacity = Constraint(
        expr=sum(
            RESOURCE[i] * model.q[i]
            for i in range(2)
        )
        <= capacity
    )

    expected_recourse_profit = sum(
        probability
        * (
            PRICE[i] * model.sold[i, s]
            + SALVAGE[i] * model.leftover[i, s]
            - SHORTAGE_PENALTY[i] * model.shortage[i, s]
        )
        for i in range(2)
        for s in range(n)
    )

    first_stage_cost = sum(
        COST[i] * model.q[i]
        for i in range(2)
    )

    model.TotalProfit = Objective(
        expr=expected_recourse_profit - first_stage_cost,
        sense=maximize,
    )

    solution = solver.solve(model, tee=False)

    if (
        solution.solver.termination_condition
        != TerminationCondition.optimal
    ):
        raise RuntimeError(
            "SAA LP did not solve optimally. "
            f"Termination: "
            f"{solution.solver.termination_condition}"
        )

    q = np.array([
        value(model.q[0]),
        value(model.q[1]),
    ])

    training_profit = float(value(model.TotalProfit))
    computation_time = float(time.perf_counter() - start)

    return training_profit, q, computation_time


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
    wmape = (
        np.sum(absolute_error)
        / max(np.sum(np.abs(q_exact)), 1e-10)
        * 100.0
    )

    return (
        float(rmse),
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
    computation_time,
    train_profit=np.nan,
    validation=np.nan,
    q1_std=0.0,
    q2_std=0.0,
    final_profit_std=0.0,
    replication_order_metrics=None,
):
    q = np.asarray(q, dtype=float)

    gap = exact_profit - final_profit
    gap_pct = 100.0 * gap / exact_profit

    # SAA Q1/Q2 are descriptive replication means. Its accuracy is the
    # mean of the individual replication errors (Section 2.5).
    if replication_order_metrics is None:
        rmse, wmape = order_metrics(q, q_exact)
    else:
        rmse, wmape = np.mean(replication_order_metrics, axis=0)

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
        "Q_WMAPE": wmape,
        "ComputationTimeSec": computation_time,
    }


# 12. MAIN EXPERIMENT

def run_all_cases(cases):
    if RUN_SAA:
        if not PYOMO_AVAILABLE:
            raise RuntimeError(
                "Pyomo is not installed. Install Pyomo or use "
                "RUN_MODE='paper_check'."
            )

        solver_kwargs = {}

        if SOLVER_NAME.lower() == "glpk":
            glpsol_candidates = [
                Path(sys.prefix) / "Library" / "bin" / "glpsol.exe",
                Path(sys.executable).parent
                / "Library"
                / "bin"
                / "glpsol.exe",
            ]

            glpsol_path = next(
                (
                    candidate
                    for candidate in glpsol_candidates
                    if candidate.is_file()
                ),
                None,
            )

            if glpsol_path is not None:
                solver_kwargs["executable"] = str(glpsol_path)
                print(f"GLPK executable   : {glpsol_path}")

        solver = SolverFactory(SOLVER_NAME, **solver_kwargs)

        if not solver.available(exception_flag=False):
            raise RuntimeError(
                f"Solver '{SOLVER_NAME}' is unavailable. "
                "Install GLPK and make sure glpsol is visible."
            )
    else:
        solver = None

    all_rows = []
    saa_replication_rows = []

    total_capacity_runs = (
        len(SELECTED_CASES) * len(SERVICE_LEVELS)
    )
    completed = 0

    for case_number in SELECTED_CASES:
        case = cases[case_number]

        print("\n" + "=" * 88)
        print(case["case_name"])
        print("=" * 88)

        training_samples = []
        validation_sample = None

        if RUN_SAA:
            rng_train = np.random.default_rng(
                TRAIN_SEED + case_number * 1000
            )

            training_samples = [
                generate_case_demands(
                    case,
                    rng_train,
                    SAA_N,
                )
                for _ in range(SAA_REPLICATIONS)
            ]

            rng_validation = np.random.default_rng(
                VALIDATION_SEED + case_number * 1000
            )

            validation_sample = generate_case_demands(
                case,
                rng_validation,
                VALIDATION_N,
            )

        for service_level in SERVICE_LEVELS:
            capacity = capacity_from_service_level(
                case,
                service_level,
            )

            # Exact Lambda
            exact_result, exact_time = average_computation_time(
                lambda: solve_exact_lambda(case, capacity),
                LAMBDA_TIME_REPEATS,
            )

            q_exact, lambda_star = exact_result
            exact_profit = analytical_total_profit(
                case,
                q_exact,
            )

            all_rows.append(
                build_result_row(
                    case_number,
                    case,
                    service_level,
                    capacity,
                    "Exact Lambda",
                    q_exact,
                    exact_profit,
                    q_exact,
                    exact_profit,
                    exact_time,
                )
            )

            # Preserve the numerical multiplier with the Exact Lambda row.
            all_rows[-1]["LambdaStar"] = lambda_star

            # H1
            q_h1, h1_time = average_computation_time(
                lambda: solve_h1(case, capacity),
                HEURISTIC_TIME_REPEATS,
            )

            profit_h1 = analytical_total_profit(case, q_h1)

            all_rows.append(
                build_result_row(
                    case_number,
                    case,
                    service_level,
                    capacity,
                    "H1",
                    q_h1,
                    profit_h1,
                    q_exact,
                    exact_profit,
                    h1_time,
                )
            )

            # H2
            q_h2, h2_time = average_computation_time(
                lambda: solve_h2(case, capacity),
                HEURISTIC_TIME_REPEATS,
            )

            profit_h2 = analytical_total_profit(case, q_h2)

            all_rows.append(
                build_result_row(
                    case_number,
                    case,
                    service_level,
                    capacity,
                    "H2",
                    q_h2,
                    profit_h2,
                    q_exact,
                    exact_profit,
                    h2_time,
                )
            )

            # H3
            q_h3, h3_time = average_computation_time(
                lambda: solve_h3(case, capacity),
                HEURISTIC_TIME_REPEATS,
            )

            profit_h3 = analytical_total_profit(case, q_h3)

            all_rows.append(
                build_result_row(
                    case_number,
                    case,
                    service_level,
                    capacity,
                    "H3",
                    q_h3,
                    profit_h3,
                    q_exact,
                    exact_profit,
                    h3_time,
                )
            )

            # SAA
            if RUN_SAA:
                q_list = []
                train_profit_list = []
                validation_profit_list = []
                analytical_profit_list = []
                computation_time_list = []
                replication_order_metrics = []

                for replication, (
                    train_d1,
                    train_d2,
                ) in enumerate(training_samples, start=1):
                    (
                        train_profit,
                        q_saa,
                        saa_time,
                    ) = solve_saa_lp(
                        train_d1,
                        train_d2,
                        capacity,
                        solver,
                    )

                    val_profit = validation_profit(
                        q_saa,
                        validation_sample[0],
                        validation_sample[1],
                    )

                    analytical_profit_saa = (
                        analytical_total_profit(
                            case,
                            q_saa,
                        )
                    )

                    q_list.append(q_saa)
                    train_profit_list.append(train_profit)
                    validation_profit_list.append(val_profit)
                    analytical_profit_list.append(
                        analytical_profit_saa
                    )
                    computation_time_list.append(saa_time)

                    rep_rmse, rep_wmape = order_metrics(q_saa, q_exact)
                    replication_order_metrics.append((rep_rmse, rep_wmape))

                    saa_replication_rows.append({
                        "Case": case_number,
                        "ServiceLevel": service_level,
                        "Capacity": capacity,
                        "Replication": replication,
                        "ScenarioSize": SAA_N,
                        "Q1": q_saa[0],
                        "Q2": q_saa[1],
                        "TrainingProfit": train_profit,
                        "ValidationProfit": val_profit,
                        "AnalyticalProfit": analytical_profit_saa,
                        "ProfitGap": exact_profit - analytical_profit_saa,
                        "ProfitGapPct": (
                            100.0
                            * (
                                exact_profit
                                - analytical_profit_saa
                            )
                            / exact_profit
                        ),
                        "Q_RMSE": rep_rmse,
                        "Q_WMAPE": rep_wmape,
                        "ComputationTimeSec": saa_time,
                    })

                q_array = np.vstack(q_list)
                q_mean = np.mean(q_array, axis=0)

                # Fair final comparison:
                # mean exact analytical profit of the SAA decisions.
                saa_final_profit = float(
                    np.mean(analytical_profit_list)
                )

                all_rows.append(
                    build_result_row(
                        case_number,
                        case,
                        service_level,
                        capacity,
                        "SAA",
                        q_mean,
                        saa_final_profit,
                        q_exact,
                        exact_profit,
                        float(np.mean(computation_time_list)),
                        train_profit=float(
                            np.mean(train_profit_list)
                        ),
                        validation=float(
                            np.mean(validation_profit_list)
                        ),
                        q1_std=(
                            float(
                                np.std(
                                    q_array[:, 0],
                                    ddof=1,
                                )
                            )
                            if SAA_REPLICATIONS > 1
                            else 0.0
                        ),
                        q2_std=(
                            float(
                                np.std(
                                    q_array[:, 1],
                                    ddof=1,
                                )
                            )
                            if SAA_REPLICATIONS > 1
                            else 0.0
                        ),
                        final_profit_std=(
                            float(
                                np.std(
                                    analytical_profit_list,
                                    ddof=1,
                                )
                            )
                            if SAA_REPLICATIONS > 1
                            else 0.0
                        ),
                        replication_order_metrics=replication_order_metrics,
                    )
                )

            completed += 1

            print(
                f"{completed:>3}/{total_capacity_runs} | "
                f"Case={case_number:>2} | "
                f"service={service_level:.2f} | "
                f"C={capacity:.4f} | "
                f"lambda={lambda_star:.6f}"
            )

    results = pd.DataFrame(all_rows)

    results["Method"] = pd.Categorical(
        results["Method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    results = results.sort_values(
        ["Case", "ServiceLevel", "Method"]
    ).reset_index(drop=True)

    saa_details = pd.DataFrame(saa_replication_rows)

    return results, saa_details


# 13. SUMMARIES

def make_case_method_summary(results):
    # Average over capacity levels; SAA rows already average replications.
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
            AverageComputationTimeSec=("ComputationTimeSec", "mean"),
            MaximumComputationTimeSec=("ComputationTimeSec", "max"),
            AverageQ_RMSE=("Q_RMSE", "mean"),
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
    # Equal weight per case; maximum gap is over case-capacity mean gaps.
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
            MeanComputationTimeSec=(
                "AverageComputationTimeSec",
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
                cell.font = cell.font.copy(bold=True, color="FFFFFF")
                cell.fill = cell.fill.copy(
                    fill_type="solid",
                    fgColor="1F4E78",
                )

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
                "Q_WMAPE",
                "AverageProfitGapPct",
                "MaximumProfitGapPct",
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

def build_paper_benchmark():
    """
    Table 3 values:
        average suboptimality / maximum suboptimality (%)
    """
    values = {
        1:  (0.02, 0.05, 0.09, 0.18),
        2:  (0.03, 0.05, 0.09, 0.17),
        3:  (0.10, 0.19, 0.36, 0.65),
        4:  (0.001, 0.002, 0.0003, 0.003),
        5:  (0.003, 0.005, 0.001, 0.009),
        6:  (1.42, 4.78, 4.67, 13.25),
        7:  (0.55, 1.56, 0.36, 1.90),
        8:  (0.12, 0.20, 0.04, 0.31),
        9:  (0.08, 0.14, 0.03, 0.21),
        10: (0.07, 0.12, 0.03, 0.20),
        11: (0.06, 0.11, 0.03, 0.17),
        12: (0.54, 0.85, 0.16, 1.12),
        13: (0.53, 0.85, 0.09, 0.41),
    }

    rows = []

    for case_number, numbers in values.items():
        rows.append({
            "Case": case_number,
            "Paper_H1H3_AverageGapPct": numbers[0],
            "Paper_H1H3_MaximumGapPct": numbers[1],
            "Paper_H2_AverageGapPct": numbers[2],
            "Paper_H2_MaximumGapPct": numbers[3],
        })

    return pd.DataFrame(rows)


def compare_with_paper(case_method_summary):
    benchmark = build_paper_benchmark()

    h1 = (
        case_method_summary[
            case_method_summary["Method"].astype(str) == "H1"
        ][
            [
                "Case",
                "AverageProfitGapPct",
                "MaximumProfitGapPct",
            ]
        ]
        .rename(columns={
            "AverageProfitGapPct": "Code_H1_AverageGapPct",
            "MaximumProfitGapPct": "Code_H1_MaximumGapPct",
        })
    )

    h2 = (
        case_method_summary[
            case_method_summary["Method"].astype(str) == "H2"
        ][
            [
                "Case",
                "AverageProfitGapPct",
                "MaximumProfitGapPct",
            ]
        ]
        .rename(columns={
            "AverageProfitGapPct": "Code_H2_AverageGapPct",
            "MaximumProfitGapPct": "Code_H2_MaximumGapPct",
        })
    )

    comparison = (
        benchmark
        .merge(h1, on="Case", how="left")
        .merge(h2, on="Case", how="left")
    )

    comparison["H1_AverageDifference"] = (
        comparison["Code_H1_AverageGapPct"]
        - comparison["Paper_H1H3_AverageGapPct"]
    )

    comparison["H1_MaximumDifference"] = (
        comparison["Code_H1_MaximumGapPct"]
        - comparison["Paper_H1H3_MaximumGapPct"]
    )

    comparison["H2_AverageDifference"] = (
        comparison["Code_H2_AverageGapPct"]
        - comparison["Paper_H2_AverageGapPct"]
    )

    comparison["H2_MaximumDifference"] = (
        comparison["Code_H2_MaximumGapPct"]
        - comparison["Paper_H2_MaximumGapPct"]
    )

    return comparison


# 15. PLOTS

def make_plots(results, case_method_summary):
    plot_folder = OUTPUT_FOLDER / "plots"

    if SAVE_FILES:
        plot_folder.mkdir(parents=True, exist_ok=True)

    available_methods = (
        results["Method"].astype(str).unique()
    )

    # Per-case final profit and gap plots
    for case_number in SELECTED_CASES:
        case_data = results[results["Case"] == case_number]

        profit_table = case_data.pivot(
            index="ServiceLevel",
            columns="Method",
            values="FinalProfit",
        )

        plt.figure(figsize=(9, 5))

        for method in METHOD_ORDER:
            if method in available_methods and method in profit_table:
                plt.plot(
                    profit_table.index,
                    profit_table[method],
                    marker="o",
                    markersize=3,
                    label=method,
                )

        plt.xlabel("Service Level")
        plt.ylabel("Analytical Final Profit")
        plt.title(
            f"Case {case_number}: Final Profit Comparison"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if SAVE_FILES:
            plt.savefig(
                plot_folder
                / f"case_{case_number:02d}_final_profit.png",
                dpi=180,
            )

        if not SHOW_PLOTS:
            plt.close()

        gap_table = case_data.pivot(
            index="ServiceLevel",
            columns="Method",
            values="ProfitGapPct",
        )

        plt.figure(figsize=(9, 5))

        for method in ["H1", "H2", "H3", "SAA"]:
            if method in available_methods and method in gap_table:
                plt.plot(
                    gap_table.index,
                    gap_table[method],
                    marker="o",
                    markersize=3,
                    label=method,
                )

        plt.axhline(0.0, linewidth=1.0)
        plt.xlabel("Service Level")
        plt.ylabel("Profit Gap from Exact Lambda (%)")
        plt.title(
            f"Case {case_number}: Profit Suboptimality"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if SAVE_FILES:
            plt.savefig(
                plot_folder
                / f"case_{case_number:02d}_profit_gap.png",
                dpi=180,
            )

        if not SHOW_PLOTS:
            plt.close()

    # Average gap by case
    plt.figure(figsize=(10, 6))

    for method in ["H1", "H2", "H3", "SAA"]:
        method_data = case_method_summary[
            case_method_summary["Method"].astype(str)
            == method
        ]

        if not method_data.empty:
            plt.plot(
                method_data["Case"],
                method_data["AverageProfitGapPct"],
                marker="o",
                label=method,
            )

    plt.xlabel("Case")
    plt.ylabel("Average Profit Gap (%)")
    plt.title("All 13 Cases: Average Profit Gap")
    plt.xticks(range(1, 14))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_FILES:
        plt.savefig(
            plot_folder / "all_cases_average_profit_gap.png",
            dpi=200,
        )

    if not SHOW_PLOTS:
        plt.close()

    # Maximum gap by case
    plt.figure(figsize=(10, 6))

    for method in ["H1", "H2", "H3", "SAA"]:
        method_data = case_method_summary[
            case_method_summary["Method"].astype(str)
            == method
        ]

        if not method_data.empty:
            plt.plot(
                method_data["Case"],
                method_data["MaximumProfitGapPct"],
                marker="o",
                label=method,
            )

    plt.xlabel("Case")
    plt.ylabel("Maximum Profit Gap (%)")
    plt.title("All 13 Cases: Maximum Profit Gap")
    plt.xticks(range(1, 14))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if SAVE_FILES:
        plt.savefig(
            plot_folder / "all_cases_maximum_profit_gap.png",
            dpi=200,
        )

    if not SHOW_PLOTS:
        plt.close()

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


# 16. MAIN PROGRAM

if __name__ == "__main__":
    print("=" * 90)
    print("13-CASE FIVE-METHOD NEWSVENDOR EXPERIMENT")
    print("=" * 90)
    print(f"Run mode          : {RUN_MODE}")
    print(f"Selected cases    : {SELECTED_CASES}")
    print(f"Capacity levels   : {len(SERVICE_LEVELS)}")
    print(f"SAA enabled       : {RUN_SAA}")

    if RUN_SAA:
        print(f"SAA scenario size : {SAA_N}")
        print(f"SAA replications  : {SAA_REPLICATIONS}")
        print(f"Validation size   : {VALIDATION_N}")
        print(
            "Total LP solves    : "
            f"{len(SELECTED_CASES) * len(SERVICE_LEVELS) * SAA_REPLICATIONS}"
        )

    if SAVE_FILES:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    all_cases = build_cases()

    results, saa_details = run_all_cases(all_cases)

    case_method_summary = make_case_method_summary(results)
    overall_summary = make_overall_method_summary(
        case_method_summary
    )

    paper_comparison = compare_with_paper(
        case_method_summary
    )

    print("\n" + "=" * 90)
    print("CASE-METHOD SUMMARY")
    print("=" * 90)

    print_columns = [
        "Case",
        "Method",
        "AverageProfitGapPct",
        "MaximumProfitGapPct",
        "AverageComputationTimeSec",
        "AverageQ_RMSE",
        "AverageQ_WMAPE",
    ]

    print(
        case_method_summary[print_columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("OVERALL METHOD SUMMARY")
    print("=" * 90)

    print(
        overall_summary
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("CHECK AGAINST PAPER TABLE 3")
    print("=" * 90)

    comparison_columns = [
        "Case",
        "Paper_H1H3_AverageGapPct",
        "Code_H1_AverageGapPct",
        "Paper_H1H3_MaximumGapPct",
        "Code_H1_MaximumGapPct",
        "Paper_H2_AverageGapPct",
        "Code_H2_AverageGapPct",
        "Paper_H2_MaximumGapPct",
        "Code_H2_MaximumGapPct",
    ]

    print(
        paper_comparison[comparison_columns]
        .round(6)
        .to_string(index=False)
    )

    if SAVE_FILES:
        results.to_csv(
            OUTPUT_FOLDER / "all_13_cases_all_results.csv",
            index=False,
        )

        case_method_summary.to_csv(
            OUTPUT_FOLDER / "case_method_summary.csv",
            index=False,
        )

        overall_summary.to_csv(
            OUTPUT_FOLDER / "overall_method_summary.csv",
            index=False,
        )

        paper_comparison.to_csv(
            OUTPUT_FOLDER / "paper_table3_comparison.csv",
            index=False,
        )

        if RUN_SAA:
            saa_details.to_csv(
                OUTPUT_FOLDER / "saa_replication_details.csv",
                index=False,
            )

        excel_path = save_results_to_excel(
            results,
            case_method_summary,
            overall_summary,
            paper_comparison,
            saa_details,
        )

        print("Excel workbook    :", excel_path.resolve())

    make_plots(
        results,
        case_method_summary,
    )

    print("\nDone.")

    if SAVE_FILES:
        print("Saved in:", OUTPUT_FOLDER.resolve())
