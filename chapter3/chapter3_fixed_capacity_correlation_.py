# -*- coding: utf-8 -*-
"""
Case 1 fixed-capacity correlated-demand experiment

Methods:
    Exact Lambda
    H1
    H2
    H3
    SAA-LP

Demand:
    Product 1: Normal(mean=50, std=10)
    Product 2: Uniform(mean=50, std=10)

Correlation environments:
    rho = -0.5, 0.0, +0.5

Main research logic:
1. Fix one shared capacity at service level 0.50, which gives C = 100.
2. Keep the two marginal demand samples unchanged and change only their dependence.
3. Compare Exact Lambda, H1, H2, H3 and SAA.
4. Study SAA convergence across several training scenario sizes.
5. Use 20 independent SAA replications for each training size.
6. Evaluate final decisions on a common 100,000-scenario validation pool for each rho.
7. Select one common SAA scenario size N* that is closest to Exact Lambda.
8. At N*, compare realized-profit risk under rho = -0.5, 0.0 and +0.5.
9. Report mean, standard deviation, profit VaR at 1%, 5%, 10%, and
   lower-tail profit CVaR at 1%, 5%, 10%.
10. Plot frequency polygons using common profit bins so the distribution spread
    can be compared directly across correlations.

"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

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
except ImportError:
    SolverFactory = None  # Additional study does not require Pyomo.



# 1. USER SETTINGS

SOLVER_NAME = "glpk"

RHO_VALUES = [-0.5, 0.0, 0.5]

FIXED_SERVICE_LEVEL = 0.50

SAA_SCENARIO_SIZES = [
    500,
    750,
    1000,
    1250,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
]

SAA_REPLICATIONS = 20
VALIDATION_N = 100_000

TRAIN_SEED = 20
VALIDATION_SEED = 2026

FREQUENCY_BINS = 45

OUTPUT_FOLDER = Path("case1_fixed_capacity_correlation_final_results")
SAVE_FILES = True
SAVE_FULL_SAA_DISTRIBUTION = True
SHOW_PLOTS = False

METHOD_ORDER = ["Exact Lambda", "H1", "H2", "H3", "SAA"]


# 2. CASE 1 PARAMETERS

NUMBER_OF_PRODUCTS = 2

PRICE = np.array([50.0, 50.0])
COST = np.array([25.0, 25.0])
SALVAGE = np.array([15.0, 15.0])
SHORTAGE_PENALTY = np.array([0.0, 0.0])
RESOURCE = np.array([1.0, 1.0])

def refresh_economic_coefficients():
    global CRITICAL_RATIO, BETA_COST
    denominator = PRICE - SALVAGE + SHORTAGE_PENALTY
    if np.any(denominator <= 0):
        raise ValueError("Invalid economic parameters")
    CRITICAL_RATIO = (PRICE - COST + SHORTAGE_PENALTY) / denominator
    BETA_COST = RESOURCE / denominator


refresh_economic_coefficients()


# 3. CASE 1 DEMAND DISTRIBUTIONS

MEAN_1 = 50.0
STD_1 = 10.0

MEAN_2 = 50.0
STD_2 = 10.0

UNIFORM_LOW = MEAN_2 - STD_2 * np.sqrt(3.0)
UNIFORM_HIGH = MEAN_2 + STD_2 * np.sqrt(3.0)


def normal_ppf(probability):
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(norm.ppf(probability, loc=MEAN_1, scale=STD_1))


def uniform_ppf(probability):
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        UNIFORM_LOW
        + probability * (UNIFORM_HIGH - UNIFORM_LOW)
    )


def normal_expected_shortage(q):
    z = (q - MEAN_1) / STD_1

    return float(
        STD_1 * norm.pdf(z)
        - (q - MEAN_1) * (1.0 - norm.cdf(z))
    )


def uniform_expected_shortage(q):
    if q <= UNIFORM_LOW:
        return float(MEAN_2 - q)

    if q >= UNIFORM_HIGH:
        return 0.0

    return float(
        (UNIFORM_HIGH - q) ** 2
        / (2.0 * (UNIFORM_HIGH - UNIFORM_LOW))
    )


# 4. FIXED CAPACITY

def capacity_from_service_level(service_level):
    q1 = normal_ppf(service_level)
    q2 = uniform_ppf(service_level)

    return float(q1 + q2)


FIXED_CAPACITY = capacity_from_service_level(
    FIXED_SERVICE_LEVEL
)


# 5. ANALYTICAL PROFIT

def analytical_profit_product_1(q):
    shortage = normal_expected_shortage(q)
    sold = MEAN_1 - shortage
    leftover = q - sold

    return float(
        PRICE[0] * sold
        + SALVAGE[0] * leftover
        - SHORTAGE_PENALTY[0] * shortage
        - COST[0] * q
    )


def analytical_profit_product_2(q):
    shortage = uniform_expected_shortage(q)
    sold = MEAN_2 - shortage
    leftover = q - sold

    return float(
        PRICE[1] * sold
        + SALVAGE[1] * leftover
        - SHORTAGE_PENALTY[1] * shortage
        - COST[1] * q
    )


def analytical_total_profit(q):
    return float(
        analytical_profit_product_1(q[0])
        + analytical_profit_product_2(q[1])
    )


# 6. EXACT LAMBDA

def exact_orders_given_lambda(lambda_value):
    adjusted_probability = (
        CRITICAL_RATIO
        - lambda_value * BETA_COST
    )

    q1 = max(normal_ppf(adjusted_probability[0]), 0.0)
    q2 = max(uniform_ppf(adjusted_probability[1]), 0.0)

    return np.array([q1, q2], dtype=float)


def solve_exact_lambda(
    capacity,
    tolerance=1e-10,
    max_iterations=200,
):
    q_unconstrained = np.array([
        normal_ppf(CRITICAL_RATIO[0]),
        uniform_ppf(CRITICAL_RATIO[1]),
    ])

    if np.dot(RESOURCE, q_unconstrained) <= capacity + tolerance:
        return q_unconstrained, 0.0

    low = 0.0
    high = 1.0

    while (
        np.dot(
            RESOURCE,
            exact_orders_given_lambda(high),
        )
        > capacity
    ):
        high *= 2.0

        if high > 1e8:
            raise RuntimeError("Could not bracket lambda.")

    middle = 0.0
    q_middle = q_unconstrained

    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        q_middle = exact_orders_given_lambda(middle)

        used = float(np.dot(RESOURCE, q_middle))

        if abs(used - capacity) <= tolerance:
            break

        if used > capacity:
            low = middle
        else:
            high = middle

    return q_middle, float(middle)


# 7. H1, H2, H3

def solve_h1(capacity):
    mean = np.array([MEAN_1, MEAN_2])
    std = np.array([STD_1, STD_2])

    denominator = float(np.sum(RESOURCE * std))

    capacity_gap = float(
        capacity - np.dot(RESOURCE, mean)
    )

    q = mean + std / denominator * capacity_gap

    return q


def solve_h2(capacity):
    std = np.array([STD_1, STD_2])

    q_unconstrained = np.array([
        normal_ppf(CRITICAL_RATIO[0]),
        uniform_ppf(CRITICAL_RATIO[1]),
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

    return q


def solve_h3(capacity):
    mean = np.array([MEAN_1, MEAN_2])
    std = np.array([STD_1, STD_2])

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

    return q


# 8. CORRELATED DEMAND

def generate_base_marginals(n, seed):
    """
    Generate one fixed Normal sample and one fixed Uniform sample.

    The same marginal observations can then be rearranged to create
    different dependence structures without changing the marginals.
    """
    rng = np.random.default_rng(seed)

    demand_1 = rng.normal(
        MEAN_1,
        STD_1,
        n,
    )

    demand_2 = rng.uniform(
        UNIFORM_LOW,
        UNIFORM_HIGH,
        n,
    )

    demand_1 = np.maximum(demand_1, 0.0)
    demand_2 = np.maximum(demand_2, 0.0)

    return demand_1, demand_2


def impose_correlation(
    base_demand_1,
    base_demand_2,
    rho,
    seed,
):
    """
    Gaussian-rank rearrangement.

    The exact marginal observations are preserved. Only the pairing
    between Product 1 and Product 2 is changed.
    """
    rng = np.random.default_rng(seed)

    n = len(base_demand_1)

    z1 = rng.normal(0.0, 1.0, n)
    error = rng.normal(0.0, 1.0, n)

    z2 = (
        rho * z1
        + np.sqrt(1.0 - rho ** 2) * error
    )

    order_1 = np.argsort(z1)
    order_2 = np.argsort(z2)

    sorted_demand_1 = np.sort(base_demand_1)
    sorted_demand_2 = np.sort(base_demand_2)

    demand_1 = np.empty(n)
    demand_2 = np.empty(n)

    demand_1[order_1] = sorted_demand_1
    demand_2[order_2] = sorted_demand_2

    actual_correlation = float(
        np.corrcoef(demand_1, demand_2)[0, 1]
    )

    return demand_1, demand_2, actual_correlation


# 9. SAA TWO-STAGE LP

def find_glpk():
    if SolverFactory is None:
        raise RuntimeError("Original study requires Pyomo; additional study does not.")
    solver_kwargs = {}

    if SOLVER_NAME.lower() == "glpk":
        candidates = [
            Path(sys.prefix) / "Library" / "bin" / "glpsol.exe",
            Path(sys.executable).parent / "Library" / "bin" / "glpsol.exe",
        ]

        path = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
            ),
            None,
        )

        if path is not None:
            solver_kwargs["executable"] = str(path)

    solver = SolverFactory(
        SOLVER_NAME,
        **solver_kwargs,
    )

    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "GLPK is unavailable. Make sure glpsol is installed."
        )

    return solver


def solve_saa_lp(
    demand_1,
    demand_2,
    capacity,
    solver,
):
    n = len(demand_1)

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

    model.q = Var(
        model.I,
        within=NonNegativeReals,
    )

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

    model.inventory_balance = Constraint(
        model.I,
        model.S,
        rule=inventory_balance,
    )

    def demand_balance(model, i, s):
        return (
            model.demand[i, s]
            == model.sold[i, s]
            + model.shortage[i, s]
        )

    model.demand_balance = Constraint(
        model.I,
        model.S,
        rule=demand_balance,
    )

    model.capacity = Constraint(
        expr=sum(
            RESOURCE[i] * model.q[i]
            for i in range(NUMBER_OF_PRODUCTS)
        )
        <= capacity
    )

    first_stage_cost = sum(
        COST[i] * model.q[i]
        for i in range(NUMBER_OF_PRODUCTS)
    )

    average_recourse = (
        1.0 / n
        * sum(
            PRICE[i] * model.sold[i, s]
            + SALVAGE[i] * model.leftover[i, s]
            - SHORTAGE_PENALTY[i] * model.shortage[i, s]
            for i in range(NUMBER_OF_PRODUCTS)
            for s in range(n)
        )
    )

    model.objective = Objective(
        expr=average_recourse - first_stage_cost,
        sense=maximize,
    )

    result = solver.solve(
        model,
        tee=False,
    )

    if (
        result.solver.termination_condition
        != TerminationCondition.optimal
    ):
        raise RuntimeError(
            "SAA LP did not terminate optimally."
        )

    q = np.array([
        value(model.q[0]),
        value(model.q[1]),
    ])

    training_profit = float(
        value(model.objective)
    )

    computation_time = time.perf_counter() - start

    return q, training_profit, computation_time


# 10. REALIZED PROFIT AND RISK METRICS

def scenario_profits(q, demand_1, demand_2):
    q1 = float(q[0])
    q2 = float(q[1])

    sold_1 = np.minimum(q1, demand_1)
    sold_2 = np.minimum(q2, demand_2)

    leftover_1 = np.maximum(q1 - demand_1, 0.0)
    leftover_2 = np.maximum(q2 - demand_2, 0.0)

    shortage_1 = np.maximum(demand_1 - q1, 0.0)
    shortage_2 = np.maximum(demand_2 - q2, 0.0)

    profit_1 = (
        PRICE[0] * sold_1
        + SALVAGE[0] * leftover_1
        - SHORTAGE_PENALTY[0] * shortage_1
        - COST[0] * q1
    )

    profit_2 = (
        PRICE[1] * sold_2
        + SALVAGE[1] * leftover_2
        - SHORTAGE_PENALTY[1] * shortage_2
        - COST[1] * q2
    )

    total_profit = profit_1 + profit_2

    return profit_1, profit_2, total_profit


def lower_tail_cvar(values, percentile_level):
    """Average the worst alpha fraction, using fractional boundary weight."""
    # Thesis Eq. (3.9) instead averages every observation at/below VaR.
    # Keep this implementation: ties or fractional tail sizes can differ.
    ordered = np.sort(np.asarray(values, dtype=float))
    if not ordered.size or not 0 < percentile_level <= 100:
        raise ValueError("Nonempty values and percentile in (0,100] required")
    mass = ordered.size * percentile_level / 100.0
    whole = int(np.floor(mass))
    fraction = mass - whole
    total = float(np.sum(ordered[:whole]))
    if fraction > 0 and whole < ordered.size:
        total += fraction * ordered[whole]
    return total / mass


def profit_distribution_metrics(total_profit):
    total_profit = np.asarray(total_profit, dtype=float)

    mean_profit = float(np.mean(total_profit))
    std_profit = float(np.std(total_profit, ddof=1))

    var_01 = float(np.percentile(total_profit, 1))
    var_05 = float(np.percentile(total_profit, 5))
    var_10 = float(np.percentile(total_profit, 10))

    cvar_01 = lower_tail_cvar(total_profit, 1)
    cvar_05 = lower_tail_cvar(total_profit, 5)
    cvar_10 = lower_tail_cvar(total_profit, 10)

    return {
        "AverageProfit": mean_profit,
        "StandardDeviationProfit": std_profit,
        "VaR01Profit": var_01,
        "VaR05Profit": var_05,
        "VaR10Profit": var_10,
        "CVaR01Profit": cvar_01,
        "CVaR05Profit": cvar_05,
        "CVaR10Profit": cvar_10,
    }


# 11. ORDER ERROR

def order_metrics(q, q_exact):
    q = np.asarray(q, dtype=float)
    q_exact = np.asarray(q_exact, dtype=float)

    error = q - q_exact
    absolute_error = np.abs(error)

    wmape = float(
        100.0
        * np.sum(absolute_error)
        / np.maximum(
            np.sum(np.abs(q_exact)),
            1e-12,
        )
    )

    return wmape


# 12. SAA CONVERGENCE EXPERIMENT

def run_saa_convergence(solver, q_exact, exact_profit):
    replication_rows = []
    summary_rows = []

    for scenario_size in SAA_SCENARIO_SIZES:
        q_list = []
        training_profit_list = []
        analytical_profit_list = []
        computation_time_list = []

        for replication in range(1, SAA_REPLICATIONS + 1):
            train_seed = (
                TRAIN_SEED
                + scenario_size * 100
                + replication
            )

            train_1, train_2 = generate_base_marginals(
                scenario_size,
                train_seed,
            )

            q_saa, training_profit, computation_time = (
                solve_saa_lp(
                    train_1,
                    train_2,
                    FIXED_CAPACITY,
                    solver,
                )
            )

            analytical_profit = analytical_total_profit(
                q_saa
            )

            wmape = order_metrics(
                q_saa,
                q_exact,
            )

            profit_gap_pct = (
                100.0
                * (exact_profit - analytical_profit)
                / exact_profit
            )

            q_list.append(q_saa)
            training_profit_list.append(training_profit)
            analytical_profit_list.append(analytical_profit)
            computation_time_list.append(computation_time)

            replication_rows.append({
                "ScenarioSize": scenario_size,
                "Replication": replication,
                "Capacity": FIXED_CAPACITY,
                "Q1": q_saa[0],
                "Q2": q_saa[1],
                "TrainingProfit": training_profit,
                "AnalyticalProfit": analytical_profit,
                "ProfitGapPct": profit_gap_pct,
                "Q_WMAPE": wmape,
                "ComputationTimeSec": computation_time,
            })

        q_array = np.vstack(q_list)
        q_mean = np.mean(q_array, axis=0)

        mean_analytical_profit = float(
            np.mean(analytical_profit_list)
        )

        mean_profit_gap_pct = (
            100.0
            * (exact_profit - mean_analytical_profit)
            / exact_profit
        )

        # Appendix C.3 uses the mean order, not mean replication WMAPE.
        wmape = order_metrics(
            q_mean,
            q_exact,
        )

        summary_rows.append({
            "ScenarioSize": scenario_size,
            "Capacity": FIXED_CAPACITY,
            "Q1Mean": q_mean[0],
            "Q2Mean": q_mean[1],
            "Q1Std": float(np.std(q_array[:, 0], ddof=1)),
            "Q2Std": float(np.std(q_array[:, 1], ddof=1)),
            "AverageTrainingProfit": float(
                np.mean(training_profit_list)
            ),
            "AverageAnalyticalProfit": mean_analytical_profit,
            "AnalyticalProfitStd": float(
                np.std(analytical_profit_list, ddof=1)
            ),
            "ProfitGapPct": mean_profit_gap_pct,
            "AbsoluteProfitGapPct": abs(mean_profit_gap_pct),
            "Q_WMAPE": wmape,
            "AverageComputationTimeSec": float(np.mean(computation_time_list)),
        })

        print(
            "Finished SAA scenario size",
            scenario_size,
        )

    replications = pd.DataFrame(replication_rows)
    summary = pd.DataFrame(summary_rows)

    return replications, summary


# 13. SELECT COMMON SAA SCENARIO SIZE

def select_best_scenario_size(saa_summary):
    ranked = saa_summary.copy()

    ranked = ranked.sort_values(
        [
            "AbsoluteProfitGapPct",
            "Q_WMAPE",
            "ScenarioSize",
        ]
    ).reset_index(drop=True)

    best_n = int(ranked.loc[0, "ScenarioSize"])

    ranked["SelectedForDistribution"] = (
        ranked["ScenarioSize"] == best_n
    )

    return best_n, ranked


# 14. VALIDATION SETS

def build_validation_sets():
    base_1, base_2 = generate_base_marginals(
        VALIDATION_N,
        VALIDATION_SEED,
    )

    validation_sets = {}

    for rho in RHO_VALUES:
        d1, d2, actual_rho = impose_correlation(
            base_1,
            base_2,
            rho,
            VALIDATION_SEED + 5000,
        )

        validation_sets[rho] = {
            "d1": d1,
            "d2": d2,
            "actual_rho": actual_rho,
        }

    return validation_sets


# 15. FIVE-METHOD RISK EVALUATION

def build_method_risk_summary(
    validation_sets,
    q_exact,
    lambda_star,
    q_h1,
    q_h2,
    q_h3,
    q_saa,
    best_n,
):
    method_orders = {
        "Exact Lambda": q_exact,
        "H1": q_h1,
        "H2": q_h2,
        "H3": q_h3,
        "SAA": q_saa,
    }

    exact_profit = analytical_total_profit(q_exact)

    rows = []
    saa_distribution_frames = []

    for rho in RHO_VALUES:
        d1 = validation_sets[rho]["d1"]
        d2 = validation_sets[rho]["d2"]
        actual_rho = validation_sets[rho]["actual_rho"]

        for method, q in method_orders.items():
            profit_1, profit_2, total_profit = scenario_profits(
                q,
                d1,
                d2,
            )

            analytical_profit = analytical_total_profit(q)
            risk = profit_distribution_metrics(total_profit)

            wmape = order_metrics(
                q,
                q_exact,
            )

            rows.append({
                "RhoTarget": rho,
                "RhoActualValidation": actual_rho,
                "Capacity": FIXED_CAPACITY,
                "Method": method,
                "SelectedScenarioSize": (
                    best_n if method == "SAA" else np.nan
                ),
                "LambdaStar": (
                    lambda_star if method == "Exact Lambda" else np.nan
                ),
                "Q1": q[0],
                "Q2": q[1],
                "CapacityUsed": float(np.dot(RESOURCE, q)),
                "AnalyticalProfit": analytical_profit,
                "AnalyticalProfitGapPct": (
                    100.0
                    * (exact_profit - analytical_profit)
                    / exact_profit
                ),
                "AverageProfitP1": float(np.mean(profit_1)),
                "AverageProfitP2": float(np.mean(profit_2)),
                "Q_WMAPE": wmape,
                **risk,
            })

            if method == "SAA":
                saa_distribution_frames.append(
                    pd.DataFrame({
                        "RhoTarget": rho,
                        "RhoActualValidation": actual_rho,
                        "Scenario": np.arange(1, VALIDATION_N + 1),
                        "ProfitP1": profit_1,
                        "ProfitP2": profit_2,
                        "TotalProfit": total_profit,
                    })
                )

    method_risk_summary = pd.DataFrame(rows)

    saa_profit_distribution = pd.concat(
        saa_distribution_frames,
        ignore_index=True,
    )

    return method_risk_summary, saa_profit_distribution


# 16. FREQUENCY POLYGON HELPERS

def common_bin_edges(profit_dict, bins=45):
    all_values = np.concatenate(
        [
            np.asarray(values, dtype=float)
            for values in profit_dict.values()
        ]
    )

    return np.linspace(
        float(np.min(all_values)),
        float(np.max(all_values)),
        bins + 1,
    )


def frequency_polygon_values(values, bin_edges):
    counts, edges = np.histogram(
        values,
        bins=bin_edges,
    )

    relative_frequency = (
        counts / np.sum(counts)
    )

    midpoints = (
        edges[:-1] + edges[1:]
    ) / 2.0

    return midpoints, relative_frequency


def remove_max_profit_mass(values):
    values = np.asarray(values, dtype=float)
    max_profit = float(np.max(values))

    tolerance = max(1e-8, abs(max_profit) * 1e-10)

    return values[
        values < max_profit - tolerance
    ]


# 17. PLOT HELPERS

def save_or_show(filename):
    plt.tight_layout()

    if SAVE_FILES:
        plt.savefig(
            OUTPUT_FOLDER / filename,
            dpi=200,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def make_plots(
    saa_summary,
    method_risk_summary,
    saa_profit_distribution,
    q_exact,
    q_h1,
    q_h2,
    q_h3,
    q_saa,
    best_n,
):
    d = saa_summary.sort_values("ScenarioSize")

    # Figure C.1: analytical profit gap.
    plt.figure(figsize=(9, 5))
    plt.plot(
        d["ScenarioSize"],
        d["AbsoluteProfitGapPct"],
        marker="o",
    )
    plt.xlabel("SAA Training Scenario Size")
    plt.ylabel("Absolute Analytical Profit Gap (%)")
    plt.title("SAA Profit Gap from Exact Lambda")
    plt.grid(True, alpha=0.3)
    save_or_show("02_saa_profit_gap_convergence.png")

    # Figure C.2: WMAPE of the mean replication order (Appendix C.3).
    plt.figure(figsize=(9, 5))
    plt.plot(
        d["ScenarioSize"],
        d["Q_WMAPE"],
        marker="o",
    )
    plt.xlabel("SAA Training Scenario Size")
    plt.ylabel("Q WMAPE (%)")
    plt.title("SAA Order-Quantity Error")
    plt.grid(True, alpha=0.3)
    save_or_show("03_saa_order_wmape_convergence.png")

    # Figure C.3: SAA stability across replications.
    plt.figure(figsize=(9, 5))
    plt.plot(
        d["ScenarioSize"],
        d["AnalyticalProfitStd"],
        marker="o",
    )
    plt.xlabel("SAA Training Scenario Size")
    plt.ylabel("Std. Dev. of Analytical Profit Across Replications")
    plt.title("SAA Stability Across Replications")
    plt.grid(True, alpha=0.3)
    save_or_show("04_saa_profit_stability.png")

    # Prepare SAA profit arrays by correlation.
    profit_dict = {}

    for rho in RHO_VALUES:
        profit_dict[rho] = (
            saa_profit_distribution[
                np.isclose(
                    saa_profit_distribution["RhoTarget"],
                    rho,
                )
            ]["TotalProfit"]
            .to_numpy()
        )

    # Figure 3.1: complete SAA frequency polygons with common bins.
    full_edges = common_bin_edges(
        profit_dict,
        FREQUENCY_BINS,
    )

    plt.figure(figsize=(9, 5))

    for rho in RHO_VALUES:
        mids, freq = frequency_polygon_values(
            profit_dict[rho],
            full_edges,
        )

        plt.plot(
            mids,
            freq,
            marker="o",
            markersize=3,
            label=f"rho={rho:+.1f}",
        )

    plt.xlabel("Realized Total Profit")
    plt.ylabel("Relative Frequency")
    plt.title(
        f"SAA Profit Frequency under Different Correlations, N={best_n}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show("06_saa_profit_frequency_all_correlations.png")

    # Figure C.4: distribution after removing maximum-profit point mass.
    continuous_dict = {
        rho: remove_max_profit_mass(profit_dict[rho])
        for rho in RHO_VALUES
    }

    continuous_edges = common_bin_edges(
        continuous_dict,
        FREQUENCY_BINS,
    )

    plt.figure(figsize=(9, 5))

    for rho in RHO_VALUES:
        mids, freq = frequency_polygon_values(
            continuous_dict[rho],
            continuous_edges,
        )

        plt.plot(
            mids,
            freq,
            marker="o",
            markersize=3,
            label=f"rho={rho:+.1f}",
        )

    plt.xlabel("Realized Total Profit")
    plt.ylabel("Relative Frequency")
    plt.title(
        "SAA Profit Frequency by Correlation "
        "(Maximum-Profit Mass Excluded)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show("07_saa_profit_frequency_without_max_mass.png")


# 18. SAVE RESULTS

def save_results(
    saa_replications,
    saa_summary,
    scenario_selection,
    method_risk_summary,
    saa_profit_distribution,
):
    if not SAVE_FILES:
        return

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    saa_replications.to_csv(
        OUTPUT_FOLDER / "saa_replications.csv",
        index=False,
    )

    saa_summary.to_csv(
        OUTPUT_FOLDER / "saa_summary_by_scenario_size.csv",
        index=False,
    )

    scenario_selection.to_csv(
        OUTPUT_FOLDER / "scenario_size_selection.csv",
        index=False,
    )

    method_risk_summary.to_csv(
        OUTPUT_FOLDER / "method_risk_summary.csv",
        index=False,
    )

    if SAVE_FULL_SAA_DISTRIBUTION:
        saa_profit_distribution.to_csv(
            OUTPUT_FOLDER / "selected_saa_profit_distribution.csv",
            index=False,
        )


# 19. MAIN

def main():
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCase 1 fixed-capacity correlation experiment")
    print("Product 1: Normal(mean=50, std=10)")
    print(
        "Product 2: Uniform(",
        round(UNIFORM_LOW, 4),
        ",",
        round(UNIFORM_HIGH, 4),
        ")",
    )
    print("Target rho values:", RHO_VALUES)
    print("Fixed service level:", FIXED_SERVICE_LEVEL)
    print("Fixed capacity:", round(FIXED_CAPACITY, 6))
    print("SAA scenario sizes:", SAA_SCENARIO_SIZES)
    print("SAA replications:", SAA_REPLICATIONS)
    print("Validation scenarios per rho:", VALIDATION_N)

    solver = find_glpk()

    q_exact, lambda_star = solve_exact_lambda(
        FIXED_CAPACITY
    )

    q_h1 = solve_h1(FIXED_CAPACITY)
    q_h2 = solve_h2(FIXED_CAPACITY)
    q_h3 = solve_h3(FIXED_CAPACITY)

    exact_profit = analytical_total_profit(q_exact)

    saa_replications, saa_summary = (
        run_saa_convergence(
            solver,
            q_exact,
            exact_profit,
        )
    )

    best_n, scenario_selection = (
        select_best_scenario_size(
            saa_summary
        )
    )

    selected_reps = saa_replications[
        saa_replications["ScenarioSize"] == best_n
    ]

    q_saa = np.array([
        selected_reps["Q1"].mean(),
        selected_reps["Q2"].mean(),
    ])

    validation_sets = build_validation_sets()

    method_risk_summary, saa_profit_distribution = (
        build_method_risk_summary(
            validation_sets,
            q_exact,
            lambda_star,
            q_h1,
            q_h2,
            q_h3,
            q_saa,
            best_n,
        )
    )

    save_results(
        saa_replications,
        saa_summary,
        scenario_selection,
        method_risk_summary,
        saa_profit_distribution,
    )

    make_plots(
        saa_summary,
        method_risk_summary,
        saa_profit_distribution,
        q_exact,
        q_h1,
        q_h2,
        q_h3,
        q_saa,
        best_n,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    print("\nExact Lambda:")
    print("Q =", np.round(q_exact, 6))
    print("Lambda* =", round(lambda_star, 6))
    print("Analytical profit =", round(exact_profit, 6))

    print("\nSelected SAA scenario size:")
    print("N* =", best_n)

    selected_row = scenario_selection[
        scenario_selection["ScenarioSize"] == best_n
    ].iloc[0]

    print(
        "Absolute analytical profit gap (%) =",
        round(selected_row["AbsoluteProfitGapPct"], 6),
    )
    print("Mean SAA Q =", np.round(q_saa, 6))

    print("\nActual validation correlations:")
    for rho in RHO_VALUES:
        print(
            f"target rho={rho:+.1f} -> actual Pearson rho="
            f"{validation_sets[rho]['actual_rho']:.4f}"
        )

    print("\nFive-method risk summary:")
    columns_to_print = [
        "RhoTarget",
        "RhoActualValidation",
        "Method",
        "Q1",
        "Q2",
        "AnalyticalProfit",
        "AverageProfit",
        "StandardDeviationProfit",
        "VaR01Profit",
        "VaR05Profit",
        "VaR10Profit",
        "CVaR01Profit",
        "CVaR05Profit",
        "CVaR10Profit",
    ]

    print(
        method_risk_summary[columns_to_print]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FOLDER.resolve())




# ADDITIONAL EXPERIMENT: Exact Lambda risk evaluation; no SAA optimization.
# Default: run both studies. Use --original-study or --penalty-study for one.
ADDITIONAL_PENALTIES = [0.0, 20.0, 35.0, 40.0]
ADDITIONAL_REPLICATIONS = 20
ADDITIONAL_N = 100_000
ADDITIONAL_OUTPUT = Path(__file__).resolve().parent / "chapter3_penalty_summary"


def run_additional_experiment():
    from scipy.optimize import minimize_scalar
    from scipy.stats import t

    started = time.perf_counter()
    original_penalty, original_cost = SHORTAGE_PENALTY.copy(), COST.copy()
    rows, orders = [], {}
    try:
        # Validate beyond the symmetric case, where formula errors can hide.
        COST[:] = [20.0, 30.0]
        SHORTAGE_PENALTY[:] = 40.0
        refresh_economic_coefficients()
        q_check, _ = solve_exact_lambda(FIXED_CAPACITY)
        numerical = minimize_scalar(
            lambda x: -analytical_total_profit([x, FIXED_CAPACITY-x]),
            bounds=(0, FIXED_CAPACITY), method="bounded")
        assert numerical.success
        assert np.allclose(q_check, [numerical.x, FIXED_CAPACITY-numerical.x], atol=1e-4)
        COST[:] = original_cost
        assert np.isclose(lower_tail_cvar([0, 10, 10, 10], 50), 5)
        assert np.isclose(lower_tail_cvar([0, 10, 20], 50), 10/3)
        for b in ADDITIONAL_PENALTIES:
            SHORTAGE_PENALTY[:] = b
            refresh_economic_coefficients()
            q, lam = solve_exact_lambda(FIXED_CAPACITY)
            assert np.allclose(q, [50, 50], atol=1e-7)
            assert np.isclose(lam, 7.5+b/2, atol=1e-7)
            orders[b] = q.copy(), lam, analytical_total_profit(q)
        for rep in range(ADDITIONAL_REPLICATIONS):
            # rep=0 reuses the main evaluation seeds 2026/7026; Appendix C.5
            # describes separate pools. Preserve the original samples/results.
            seed = VALIDATION_SEED + rep*10007
            base1, base2 = generate_base_marginals(ADDITIONAL_N, seed)
            means = {}
            for rho in RHO_VALUES:
                d1, d2, actual = impose_correlation(base1, base2, rho, seed+5000)
                for b in ADDITIONAL_PENALTIES:
                    SHORTAGE_PENALTY[:] = b
                    q, lam, expected = orders[b]
                    p1, p2, total = scenario_profits(q, d1, d2)
                    cov = float(np.cov(p1, p2, ddof=1)[0, 1])
                    variance = float(np.var(total, ddof=1))
                    assert np.isclose(variance, np.var(p1, ddof=1)+np.var(p2, ddof=1)+2*cov, rtol=1e-12)
                    mean = float(total.mean())
                    if b in means:
                        assert np.isclose(mean, means[b], atol=1e-9, rtol=0)
                    means[b] = mean
                    rows.append(dict(Replication=rep, Penalty=b, LatentRho=rho,
                        ActualPearsonRho=actual, Q1=q[0], Q2=q[1], Lambda=lam,
                        AnalyticalMeanProfit=expected, MeanProfit=mean,
                        SD=np.sqrt(variance), VaR05Profit=float(np.percentile(total, 5)),
                        CVaR05Profit=lower_tail_cvar(total, 5), ProfitCovariance=cov))
                    del p1, p2, total
                del d1, d2
            del base1, base2
            print(f"Completed pool {rep+1}/{ADDITIONAL_REPLICATIONS}", flush=True)
    finally:
        COST[:] = original_cost
        SHORTAGE_PENALTY[:] = original_penalty
        refresh_economic_coefficients()
    records = pd.DataFrame(rows)  # Only 240 aggregate rows, never scenario rows.
    critical = float(t.ppf(.975, ADDITIONAL_REPLICATIONS-1))
    summaries = []
    for b in ADDITIONAL_PENALTIES:
        subset = records[records.Penalty == b]
        for rho in RHO_VALUES:
            group = subset[subset.LatentRho == rho]
            row = group.drop(columns="Replication").mean().to_dict()
            row.update(IndependentPools=ADDITIONAL_REPLICATIONS, ScenariosPerPool=ADDITIONAL_N)
            row["SD_CI95HalfWidth"] = critical*group.SD.std(ddof=1)/np.sqrt(ADDITIONAL_REPLICATIONS)
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    ADDITIONAL_OUTPUT.mkdir(exist_ok=True)
    records.to_csv(ADDITIONAL_OUTPUT/"pool_level_results.csv", index=False)
    summary.to_csv(ADDITIONAL_OUTPUT/"summary.csv", index=False, float_format="%.8f")
    # Figure C.5: shortage-penalty risk at the Exact Lambda order.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for b in ADDITIONAL_PENALTIES:
        g = summary[summary.Penalty == b].sort_values("LatentRho")
        axes[0].errorbar(g.LatentRho, g.SD, yerr=g.SD_CI95HalfWidth,
                         marker="o", capsize=3, label=f"b={b:g}")
        axes[1].plot(g.LatentRho, g.CVaR05Profit, marker="o", label=f"b={b:g}")
    for ax, label in zip(axes, ["Profit standard deviation", "Worst 5% mean profit (higher is better)"]):
        ax.set_xlabel("Latent Gaussian correlation parameter")
        ax.set_ylabel(label)
        ax.set_xticks(RHO_VALUES)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Case 1: shortage penalty and correlation; capacity=100")
    fig.tight_layout()
    fig.savefig(ADDITIONAL_OUTPUT/"penalty_risk.png", dpi=130)
    plt.close(fig)
    print(f"Elapsed: {time.perf_counter()-started:.1f} seconds")
    for path in sorted(ADDITIONAL_OUTPUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run both Chapter 3 studies by default, or select one."
    )
    study = parser.add_mutually_exclusive_group()
    study.add_argument(
        "--original-study", action="store_true", help="Run only the main experiment."
    )
    study.add_argument(
        "--penalty-study", action="store_true", help="Run only the shortage-penalty experiment."
    )
    args = parser.parse_args()

    if not args.penalty_study:
        main()
    if not args.original_study:
        run_additional_experiment()
