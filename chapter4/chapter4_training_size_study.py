# -*- coding: utf-8 -*-
"""Training-size sensitivity study with CSV data and PNG figures.

Use --plots-only to redraw saved CSV data without solving.
--report-only remains an alias; neither option creates a report.
"""

from pathlib import Path
import argparse
import importlib.util
import sys
import time
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
from pyomo.environ import (
    ConcreteModel, Constraint, Expression, NonNegativeReals, Objective,
    Param, RangeSet, Var, maximize, minimize, value,
)
from pyomo.opt import SolverFactory, TerminationCondition
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOLVER_NAME = "glpk"
NUMBER_OF_PRODUCTS = 2
TRAINING_SIZES = [5_000, 10_000, 20_000]
RHO_VALUES = [-0.5, 0.0, 0.5]
ALPHA_VALUES = np.round(np.linspace(0, 1, 11), 2).tolist()
EVALUATION_N = 100_000
TRAIN_SEED = 20
EVALUATION_SEED = 2026
TARGET_PERCENTILE = 20
PRICE = np.array([50.0, 50.0])
COST = np.array([25.0, 25.0])
SALVAGE = np.array([15.0, 15.0])
SHORTAGE_PENALTY = np.array([0.0, 0.0])
RESOURCE = np.array([1.0, 1.0])
SERVICE_LEVEL = 0.50
MEAN_1, STD_1, MEAN_2, STD_2 = 50.0, 10.0, 50.0, 10.0
LOW, HIGH = MEAN_2 - STD_2*np.sqrt(3), MEAN_2 + STD_2*np.sqrt(3)
TAIL_PROBABILITY = 0.05
OUTPUT_FOLDER = Path(__file__).resolve().parent
SAVE_RESULTS = True
MAKE_PLOTS = True
EDR_FEASIBILITY_TOL = 1e-7
ORIGINAL_SCRIPT = Path(r"C:\Users\qiaow\Downloads\case1_downside_risk_alpha_product_metrics_summary_only.py")



def capacity_value():
    return float(norm.ppf(SERVICE_LEVEL, loc=MEAN_1, scale=STD_1)
                 + LOW + SERVICE_LEVEL*(HIGH-LOW))


def demand_pool(n, seed, rho):
    """Exactly the original Gaussian-rank demand construction and RNG sequence."""
    rng = np.random.default_rng(seed)
    d1 = np.maximum(rng.normal(MEAN_1, STD_1, n), 0)
    d2 = np.maximum(rng.uniform(LOW, HIGH, n), 0)
    rank_rng = np.random.default_rng(seed + 5000)
    z1, error = rank_rng.normal(size=n), rank_rng.normal(size=n)
    z2 = rho*z1 + np.sqrt(1-rho*rho)*error
    a, b = np.empty(n), np.empty(n)
    a[np.argsort(z1)], b[np.argsort(z2)] = np.sort(d1), np.sort(d2)
    return np.column_stack((a, b))


def product_profits(q, demand):
    sold = np.minimum(q, demand)
    return PRICE*sold + SALVAGE*np.maximum(q-demand, 0) \
        - SHORTAGE_PENALTY*np.maximum(demand-q, 0) - COST*q


def find_glpk():
    solver_kwargs = {}

    if SOLVER_NAME.lower() == "glpk":
        candidates = [
            Path(sys.prefix)
            / "Library"
            / "bin"
            / "glpsol.exe",

            Path(sys.executable).parent
            / "Library"
            / "bin"
            / "glpsol.exe",
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
            solver_kwargs["executable"] = str(
                path
            )

    solver = SolverFactory(
        SOLVER_NAME,
        **solver_kwargs,
    )

    if not solver.available(
        exception_flag=False
    ):
        raise RuntimeError(
            "GLPK is unavailable. "
            "Make sure glpsol is installed."
        )

    return solver


def solve_unconstrained_saa(
    demand_1,
    demand_2,
    capacity,
    solver,
):
    n = len(demand_1)

    demand_data = {}

    for s in range(n):
        demand_data[0, s] = float(
            demand_1[s]
        )

        demand_data[1, s] = float(
            demand_2[s]
        )

    start = time.perf_counter()

    model = ConcreteModel()

    model.I = RangeSet(
        0,
        NUMBER_OF_PRODUCTS - 1,
    )

    model.S = RangeSet(
        0,
        n - 1,
    )

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

    def inventory_balance(
        model,
        i,
        s,
    ):
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

    def demand_balance(
        model,
        i,
        s,
    ):
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
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )
        <= capacity
    )

    first_stage_cost = sum(
        COST[i] * model.q[i]
        for i in range(
            NUMBER_OF_PRODUCTS
        )
    )

    average_recourse = (
        1.0 / n
        * sum(
            PRICE[i]
            * model.sold[i, s]
            + SALVAGE[i]
            * model.leftover[i, s]
            - SHORTAGE_PENALTY[i]
            * model.shortage[i, s]
            for i in range(
                NUMBER_OF_PRODUCTS
            )
            for s in range(n)
        )
    )

    model.objective = Objective(
        expr=(
            average_recourse
            - first_stage_cost
        ),
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
            "Unconstrained SAA LP "
            "did not terminate optimally."
        )

    q = np.array([
        value(model.q[0]),
        value(model.q[1]),
    ])

    training_profit = float(
        value(model.objective)
    )

    cpu_time = (
        time.perf_counter()
        - start
    )

    return (
        q,
        training_profit,
        cpu_time,
    )


def solve_minimum_edr(
    demand_1,
    demand_2,
    capacity,
    target_z,
    solver,
):
    """
    Auxiliary LP used to find the theoretical minimum training EDR.

    It keeps exactly the same demand scenarios, capacity, target z,
    and Newsvendor recourse structure.

    The only change is the objective:

        minimize average(downside_s)

    where:
        downside_s >= target_z - scenario_profit_s
        downside_s >= 0

    The optimal objective value is EDR_min.
    """
    n = len(demand_1)

    demand_data = {}

    for s in range(n):
        demand_data[0, s] = float(demand_1[s])
        demand_data[1, s] = float(demand_2[s])

    start = time.perf_counter()

    model = ConcreteModel()

    model.I = RangeSet(
        0,
        NUMBER_OF_PRODUCTS - 1,
    )

    model.S = RangeSet(
        0,
        n - 1,
    )

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

    model.downside = Var(
        model.S,
        within=NonNegativeReals,
    )

    def inventory_balance(
        model,
        i,
        s,
    ):
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

    def demand_balance(
        model,
        i,
        s,
    ):
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
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )
        <= capacity
    )

    def scenario_profit_rule(
        model,
        s,
    ):
        recourse_profit = sum(
            PRICE[i]
            * model.sold[i, s]
            + SALVAGE[i]
            * model.leftover[i, s]
            - SHORTAGE_PENALTY[i]
            * model.shortage[i, s]
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )

        first_stage_cost = sum(
            COST[i] * model.q[i]
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )

        return (
            recourse_profit
            - first_stage_cost
        )

    model.scenario_profit = Expression(
        model.S,
        rule=scenario_profit_rule,
    )

    def downside_rule(
        model,
        s,
    ):
        return (
            model.downside[s]
            >= target_z
            - model.scenario_profit[s]
        )

    model.downside_definition = Constraint(
        model.S,
        rule=downside_rule,
    )

    model.objective = Objective(
        expr=(
            1.0 / n
            * sum(
                model.downside[s]
                for s in range(n)
            )
        ),
        sense=minimize,
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
            "Minimum-EDR LP did not terminate optimally."
        )

    q_min = np.array([
        value(model.q[0]),
        value(model.q[1]),
    ])

    minimum_edr = float(
        value(model.objective)
    )

    minimum_edr_profit = float(
        np.mean([
            value(model.scenario_profit[s])
            for s in range(n)
        ])
    )

    cpu_time = (
        time.perf_counter()
        - start
    )

    return (
        q_min,
        minimum_edr,
        minimum_edr_profit,
        cpu_time,
    )


def solve_risk_constrained_saa(
    demand_1,
    demand_2,
    capacity,
    target_z,
    edr_limit,
    solver,
):
    """
    Maximize average SAA profit subject to:

        downside_s >= target_z - scenario_profit_s
        downside_s >= 0

        average(downside_s) <= edr_limit

    The formulation remains a linear program.

    Important:
    downside_s is an auxiliary variable used only to enforce the EDR upper
    bound. Because the objective maximizes profit rather than minimizes
    downside, the solved downside values are not used as the reported
    TrainingEDR. After solving, TrainingEDR is recomputed directly from the
    realized training profit distribution generated by q.
    """
    n = len(demand_1)

    demand_data = {}

    for s in range(n):
        demand_data[0, s] = float(
            demand_1[s]
        )

        demand_data[1, s] = float(
            demand_2[s]
        )

    start = time.perf_counter()

    model = ConcreteModel()

    model.I = RangeSet(
        0,
        NUMBER_OF_PRODUCTS - 1,
    )

    model.S = RangeSet(
        0,
        n - 1,
    )

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

    model.downside = Var(
        model.S,
        within=NonNegativeReals,
    )

    def inventory_balance(
        model,
        i,
        s,
    ):
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

    def demand_balance(
        model,
        i,
        s,
    ):
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
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )
        <= capacity
    )

    def scenario_profit_rule(
        model,
        s,
    ):
        recourse_profit = sum(
            PRICE[i]
            * model.sold[i, s]
            + SALVAGE[i]
            * model.leftover[i, s]
            - SHORTAGE_PENALTY[i]
            * model.shortage[i, s]
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )

        first_stage_cost = sum(
            COST[i] * model.q[i]
            for i in range(
                NUMBER_OF_PRODUCTS
            )
        )

        return (
            recourse_profit
            - first_stage_cost
        )

    model.scenario_profit = Expression(
        model.S,
        rule=scenario_profit_rule,
    )

    def downside_rule(
        model,
        s,
    ):
        return (
            model.downside[s]
            >= target_z
            - model.scenario_profit[s]
        )

    model.downside_definition = Constraint(
        model.S,
        rule=downside_rule,
    )

    model.edr_constraint = Constraint(
        expr=(
            1.0 / n
            * sum(
                model.downside[s]
                for s in range(n)
            )
            <= edr_limit
        )
    )

    model.objective = Objective(
        expr=(
            1.0 / n
            * sum(
                model.scenario_profit[s]
                for s in range(n)
            )
        ),
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
        return None

    q = np.array([
        value(model.q[0]),
        value(model.q[1]),
    ])

    training_profit = float(
        value(model.objective)
    )

    cpu_time = (
        time.perf_counter()
        - start
    )

    return (
        q,
        training_profit,
        cpu_time,
    )


class ScenarioProblem:
    """Thin interface to the unmodified original Pyomo/GLPK functions."""
    def __init__(self,demand,capacity,solver,target_z=None):
        self.demand=demand
        self.capacity=capacity
        self.solver=solver
        self.z=target_z

    def solve(self,objective="profit",limit=None):
        d1,d2=self.demand[:,0],self.demand[:,1]
        if objective=="risk":
            q,edr,profit,elapsed=solve_minimum_edr(d1,d2,self.capacity,self.z,self.solver)
        elif limit is None:
            q,profit,elapsed=solve_unconstrained_saa(d1,d2,self.capacity,self.solver)
            edr=np.nan if self.z is None else float(np.maximum(self.z-product_profits(q,self.demand).sum(axis=1),0).mean())
        else:
            solution=solve_risk_constrained_saa(d1,d2,self.capacity,self.z,limit,self.solver)
            if solution is None:
                return {"Status":"not_optimal"}
            q,profit,elapsed=solution
            edr=float(np.maximum(self.z-product_profits(q,self.demand).sum(axis=1),0).mean())
        return {"Status":"optimal","Q":q,"MeanProfit":profit,"EDR":edr,"CPUTimeSec":elapsed}



def tail_metrics(values):
    k = max(1, int(np.ceil(TAIL_PROBABILITY*len(values))))
    return float(np.quantile(values, TAIL_PROBABILITY)), \
        float(np.partition(values, k-1)[:k].mean())


def metrics(q, demand, z, training=False):
    pp = product_profits(q, demand)
    total = pp.sum(axis=1)
    losses = np.maximum(z-total, 0)
    var, cvar = tail_metrics(total)
    prefix = "Training" if training else ""
    edr = float(losses.mean())
    se = float(losses.std(ddof=1)/np.sqrt(len(losses)))
    result = {
        ("TrainingMeanProfit" if training else "EvaluationMeanProfit"): float(total.mean()),
        ("TrainingEDR" if training else "EvaluationEDR"): edr,
        ("TrainingProbabilityBelowTarget" if training else "ProbabilityBelowTarget"):
            float(np.mean(total < z)),
        ("TrainingTotalStdProfit" if training else "TotalStdProfit"): float(total.std(ddof=1)),
        ("TrainingTotalVaR05Profit" if training else "TotalVaR05Profit"): var,
        ("TrainingTotalCVaR05Profit" if training else "TotalCVaR05Profit"): cvar,
        ("TrainingEDRSE" if training else "EvaluationEDRSE"): se,
    }
    if not training:
        result.update(EvaluationEDRCILow=max(edr-1.96*se, 0),
                      EvaluationEDRCIHigh=edr+1.96*se,
                      P20Profit=float(np.percentile(total, TARGET_PERCENTILE)))
    for i in range(2):
        name = f"{prefix}P{i+1}"
        pv, pc = tail_metrics(pp[:, i])
        sold, leftover, shortage = np.minimum(q[i], demand[:, i]), \
            np.maximum(q[i]-demand[:, i], 0), np.maximum(demand[:, i]-q[i], 0)
        result.update({name+"MeanProfit": float(pp[:, i].mean()),
                       name+"StdProfit": float(pp[:, i].std(ddof=1)),
                       name+"VaR05Profit": pv, name+"CVaR05Profit": pc,
                       name+"FillRate": float(sold.sum()/demand[:, i].sum()),
                       name+"StockoutProbability": float(np.mean(demand[:, i] > q[i])),
                       name+"ExpectedShortage": float(shortage.mean()),
                       name+"ExpectedLeftover": float(leftover.mean())})
    return result


def risk_setup(demand,capacity,solver):
    problem=ScenarioProblem(demand,capacity,solver)
    base=problem.solve()
    baseline_profit=product_profits(base["Q"],demand).sum(axis=1)
    z=float(np.percentile(baseline_profit,TARGET_PERCENTILE))
    problem.z=z
    base["EDR"]=float(np.maximum(z-baseline_profit,0).mean())
    minimum=problem.solve("risk")
    return problem,base,minimum,z



def pct_loss(base, new):
    return 100*(base-new)/base if abs(base) > 1e-12 else np.nan


def result_row(n, rho, alpha, oracle, solution, z, limit, baseline, minimum,
               eval_demand, base_train, base_eval):
    row = {"Study": "Chapter", "TrainingN": n, "TrainSeed": TRAIN_SEED,
           "EvaluationN": len(eval_demand), "EvaluationSeed": EVALUATION_SEED,
           "RhoTarget": rho, "RhoActualTraining": float(np.corrcoef(oracle.demand.T)[0, 1]),
           "RhoActualValidation": float(np.corrcoef(eval_demand.T)[0, 1]),
           "Alpha": alpha, "Status": solution["Status"],
           "Strategy": "Unconstrained" if alpha == 0 else "Risk Constrained",
           "Capacity": oracle.capacity, "TargetZ": z,
           "BaselineTrainingEDR": baseline["EDR"], "MinimumTrainingEDR": minimum["EDR"],
           "ReducibleTrainingEDR": max(baseline["EDR"]-minimum["EDR"], 0),
           "MaximumFeasibleEDRReductionPct": pct_loss(baseline["EDR"], minimum["EDR"]),
           "TrainingEDRLimit": limit, "EDRLimit": limit,
           "EffectiveEDRLimit": limit+EDR_FEASIBILITY_TOL,
           "BaselineQ1": baseline["Q"][0], "BaselineQ2": baseline["Q"][1],
           "MinEDRQ1": minimum["Q"][0], "MinEDRQ2": minimum["Q"][1],
           "MinEDRTrainingProfit": minimum["MeanProfit"],
           "MinEDRCPUTimeSec": minimum["CPUTimeSec"],
           "B1": SHORTAGE_PENALTY[0], "B2": SHORTAGE_PENALTY[1],
           "CPUTimeSec": solution["CPUTimeSec"],
           "Solver": "Pyomo/GLPK", "FeasibilityBuffer": EDR_FEASIBILITY_TOL}
    if solution["Status"] != "optimal":
        return row
    q = solution["Q"]
    row.update(Q1=q[0], Q2=q[1], CapacityUsed=float(RESOURCE@q))
    row.update(metrics(q, oracle.demand, z, True))
    row.update(metrics(q, eval_demand, z))
    row.update(TrainingProfit=row["TrainingMeanProfit"],
               TrainingProfitLossPct=pct_loss(base_train["TrainingMeanProfit"],row["TrainingMeanProfit"]),
               EvaluationProfitLossPct=pct_loss(base_eval["EvaluationMeanProfit"],row["EvaluationMeanProfit"]),
               TrainingEDRReductionPct=pct_loss(base_train["TrainingEDR"],row["TrainingEDR"]),
               EvaluationEDRReductionPct=pct_loss(base_eval["EvaluationEDR"],row["EvaluationEDR"]))
    row["TrainingEDRMinusLimit"] = row["TrainingEDR"]-limit
    row["EvaluationEDRMinusLimit"] = row["EvaluationEDR"]-limit
    row["TrainingEDRAbsGap"] = abs(row["TrainingEDRMinusLimit"])
    row["EvaluationEDRAbsGap"] = abs(row["EvaluationEDRMinusLimit"])
    row["EvaluationEDRRelativeGapPct"] = 100*row["EvaluationEDRMinusLimit"]/limit if limit > 0 else np.nan
    row["EvaluationEDRMinusTraining"] = row["EvaluationEDR"]-row["TrainingEDR"]
    row["GeneralizationAbsGap"] = abs(row["EvaluationEDRMinusTraining"])
    row["TrainingConstraintViolation"] = max(row["TrainingEDR"]-row["EffectiveEDRLimit"], 0)
    row["EvaluationConstraintViolation"] = max(row["EvaluationEDRMinusLimit"], 0)
    row["EvaluationConstraintSlack"] = max(-row["EvaluationEDRMinusLimit"], 0)
    row["EvaluationMeetsLimit"] = row["EvaluationEDR"] <= limit
    row["TrainingConstraintBinding"] = abs(row["TrainingEDRMinusLimit"]) <= 1e-4
    row["EvaluationCICoversLimit"] = row["EvaluationEDRCILow"] <= limit <= row["EvaluationEDRCIHigh"]
    return row


def run_experiment():
    if np.any(PRICE-SALVAGE+SHORTAGE_PENALTY < 0):
        raise ValueError("Need PRICE-SALVAGE+SHORTAGE_PENALTY >= 0 for concavity.")
    if any(n < 2 for n in TRAINING_SIZES):
        raise ValueError("Need N>=2.")
    solver = find_glpk()
    capacity, rows = capacity_value(), []
    start = time.perf_counter()
    print("Original Pyomo/GLPK model; training sample size loop added.", flush=True)
    print(f"N={TRAINING_SIZES}; rho={RHO_VALUES}; training seed={TRAIN_SEED}; M={EVALUATION_N}", flush=True)
    for rho in RHO_VALUES:
        evaluation = demand_pool(EVALUATION_N, EVALUATION_SEED, rho)
        for n in TRAINING_SIZES:
            print(f"  Chapter N={n}: baseline + minimum EDR ...", flush=True)
            demand = demand_pool(n, TRAIN_SEED, rho)
            oracle, base, minimum, z = risk_setup(demand, capacity, solver)
            bt, be = metrics(base["Q"], demand, z, True), metrics(base["Q"], evaluation, z)
            for alpha in ALPHA_VALUES:
                limit = (1-alpha)*base["EDR"]+alpha*minimum["EDR"]
                solution = base if alpha == 0 else oracle.solve(limit=limit+EDR_FEASIBILITY_TOL)
                if solution["Status"] != "optimal":
                    print(f"  alpha={alpha:.1f} did not solve optimally; skipped as in the original script.",flush=True)
                    continue
                row = result_row(n,rho,alpha,oracle,solution,z,limit,base,minimum,evaluation,bt,be)
                rows.append(row)
                print(f"    alpha={alpha:.1f} {solution['Status']}" +
                      (f" trainEDR={row['TrainingEDR']:.4f} limit={limit:.4f} evalEDR={row['EvaluationEDR']:.4f}" if solution["Status"] == "optimal" else ""), flush=True)
    results = pd.DataFrame(rows)
    results = results.sort_values(["RhoTarget", "TrainingN", "Alpha"])
    if SAVE_RESULTS:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        # Persist before publishing, so an open CSV viewer cannot discard
        # completed computations. Pending files disappear after a successful replace.
        results.to_csv(OUTPUT_FOLDER/"case1_training_results.csv.pending", index=False,
                       encoding="utf-8-sig", float_format="%.10g")
        if MAKE_PLOTS:
            make_plots(results, OUTPUT_FOLDER/"plots")
        publish_pending_outputs()
    print_summary(results)
    print(f"Completed in {time.perf_counter()-start:.1f}s; outputs: {OUTPUT_FOLDER}", flush=True)
    return results


def publish_pending_outputs():
    """Publish fully written results; retain pending files only while a viewer locks them."""
    for name in ["case1_training_results.csv"]:
        staged = OUTPUT_FOLDER/(name+".pending")
        target = OUTPUT_FOLDER/name
        if not staged.is_file():
            continue
        try:
            staged.replace(target)
            print(f"Updated {target}",flush=True)
        except PermissionError:
            print(f"File is open: {target}. Completed results are safe in {staged}. "
                  "Close the CSV viewer, then run this script with --publish-pending; "
                  "no experiment needs to be repeated.",flush=True)


def print_summary(results):
    selected=results[np.isclose(results.Alpha,.5)]
    columns=["RhoTarget","TrainingN","TargetZ","EDRLimit","TrainingEDR","EvaluationEDR",
             "EvaluationEDRMinusLimit","EvaluationEDRMinusTraining","EvaluationMeanProfit"]
    print("\nOriginal-calibration comparison at alpha=0.5:")
    print(selected[columns].round(4).to_string(index=False))
    print("All 11 alpha values remain in the CSV. Targets are calibrated separately for each N.")


def finish_page(plot_folder, fig, title, note=""):
    fig.suptitle(title, fontsize=16, y=.98)
    fig.tight_layout(rect=(.02, .07 if note else .03, .99, .94))
    if note:
        fig.text(.04, .025, note, fontsize=8.5, va="bottom")
    for ax in fig.axes:
        if ax.has_data(): ax.grid(alpha=.22)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") + ".png"
    fig.savefig(plot_folder / filename, dpi=200)
    plt.close(fig)


def sample_size_comparison_plot(plot_folder, results):
    data=results[np.isclose(results.Alpha,.5)].sort_values(["RhoTarget","TrainingN"])
    if data.empty:
        return
    fig=plt.figure(figsize=(11.7,8.3))
    grid=fig.add_gridspec(2,2,height_ratios=[1,1.25],left=.08,right=.97,top=.83,bottom=.15,
                         hspace=.35,wspace=.30)
    fig.suptitle("Training size and out-of-sample performance under original calibration",fontsize=15,y=.97)
    fig.text(.08,.91,f"Selected alpha = 0.5; training N = {TRAINING_SIZES}; evaluation M = {EVALUATION_N:,}.",fontsize=11)
    fig.text(.08,.87,"Each N uses its own training P20 profit target z and feasible-range EDR limit, as in the original model.",fontsize=9.5)
    gap_ax,profit_ax=fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])
    for rho,group in data.groupby("RhoTarget"):
        group=group.sort_values("TrainingN")
        gap_ax.plot(group.TrainingN,group.EvaluationEDRMinusLimit,"o-",label=f"rho={rho:+.1f}")
        profit_ax.plot(group.TrainingN,group.EvaluationMeanProfit,"o-",label=f"rho={rho:+.1f}")
    gap_ax.axhline(0,color="black",ls="--",linewidth=1)
    gap_ax.set(title="Signed risk deviation",ylabel="Evaluation EDR - corresponding limit",xlabel="Training N")
    profit_ax.set(title="Mean evaluation profit",ylabel="Mean profit",xlabel="Training N")
    for ax in [gap_ax,profit_ax]:
        ax.set_xticks(TRAINING_SIZES,[f"{n:,}" for n in TRAINING_SIZES]);ax.grid(alpha=.22);ax.legend(fontsize=8)
    table_ax=fig.add_subplot(grid[1,:]);table_ax.axis("off")
    cells=[]
    for row in data.itertuples():
        cells.append([f"{row.RhoTarget:+.1f}",f"{row.TrainingN:,}",f"{row.TargetZ:.2f}",f"{row.EDRLimit:.4f}",
                      f"{row.TrainingEDR:.4f}",f"{row.EvaluationEDR:.4f}",
                      f"{row.EvaluationEDRMinusLimit:+.4f}",f"{row.EvaluationEDRMinusTraining:+.4f}"])
    table=table_ax.table(cellText=cells,colLabels=["rho","Training N","Profit target z","EDR limit",
                          "Training EDR","Evaluation EDR","Eval - limit","Eval - train"],
                         cellLoc="center",bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(8.3)
    for (row,_),cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row==0:
            cell.set_facecolor("#eef2f6");cell.set_text_props(weight="bold")
    fig.text(.08,.09,"Positive deviation: evaluation EDR exceeds this solution's risk limit. Negative: it is below the limit.",fontsize=9)
    fig.text(.08,.055,"Single training seed per N; no repetition error bars. Because z and the limit change with N, this is sensitivity analysis,\n"
             "not convergence toward a common fixed risk target. All 11 alpha values and full product/service metrics are in the CSV.",fontsize=8.5)
    fig.savefig(plot_folder / "training_size_comparison.png", dpi=200)
    plt.close(fig)


def make_plots(results, plot_folder):
    """Save the existing chart panels as separate PNG files."""
    plot_folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    chapter = results[(results.Study == "Chapter") & (results.Status == "optimal")]
    for rho in RHO_VALUES:
        data = chapter[np.isclose(chapter.RhoTarget, rho)]
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        panels = [("EvaluationEDRReductionPct", "Evaluation EDR reduction (%)"),
                  ("EvaluationProfitLossPct", "Evaluation profit loss (%)"),
                  ("TotalStdProfit", "Total profit standard deviation"),
                  ("Q1", "Product 1 order"), ("Q2", "Product 2 order"),
                  ("CapacityUsed", "Shared capacity used")]
        for ax, (col, label) in zip(axes.flat, panels):
            for n, group in data.groupby("TrainingN"):
                ax.plot(group.Alpha, group[col], "o-", ms=3, label=f"N={n:,}")
            ax.set(xlabel="alpha", ylabel=label)
            ax.legend(fontsize=8)
        finish_page(plot_folder, fig, f"Chapter framework: trade-off and orders | rho={rho:+.1f}",
                    "Chapter targets are recalibrated per N; curves do not share a fixed z or epsilon.")

        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        cols = [("TotalVaR05Profit","System VaR 5%"),("TotalCVaR05Profit","System CVaR 5%"),
                ("P1CVaR05Profit","Product 1 CVaR 5%"),("P2CVaR05Profit","Product 2 CVaR 5%"),
                ("P1VaR05Profit","Product 1 VaR 5%"),("P2VaR05Profit","Product 2 VaR 5%")]
        for ax, (col, label) in zip(axes.flat, cols):
            for n, group in data.groupby("TrainingN"):
                ax.plot(group.Alpha, group[col], "o-", ms=3, label=f"N={n:,}")
            ax.set(xlabel="alpha", ylabel=label); ax.legend(fontsize=8)
        finish_page(plot_folder, fig, f"Chapter framework: lower-tail profit | rho={rho:+.1f}")

        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        cols = [("P1MeanProfit","Product 1 mean profit",1),("P2MeanProfit","Product 2 mean profit",1),
                ("P1FillRate","Product 1 fill rate (%)",100),("P2FillRate","Product 2 fill rate (%)",100),
                ("P1StockoutProbability","Product 1 stockout (%)",100),("P2StockoutProbability","Product 2 stockout (%)",100)]
        for ax, (col, label, scale) in zip(axes.flat, cols):
            for n, group in data.groupby("TrainingN"):
                ax.plot(group.Alpha, scale*group[col], "o-", ms=3, label=f"N={n:,}")
            ax.set(xlabel="alpha", ylabel=label); ax.legend(fontsize=8)
        finish_page(plot_folder, fig, f"Chapter framework: product profit and service | rho={rho:+.1f}")

        fig, axes = plt.subplots(2, len(TRAINING_SIZES), figsize=(11.7, 8.3), squeeze=False)
        eval_demand = demand_pool(EVALUATION_N, EVALUATION_SEED, rho)
        for j, n in enumerate(TRAINING_SIZES):
            group = data[data.TrainingN == n]
            ax = axes[0,j]
            training_line, = ax.plot(group.Alpha, group.TrainingEDR,
                                     color="C0", marker="o", ms=3,
                                     linewidth=1.8, label="Training EDR", zorder=3)
            evaluation_line, = ax.plot(group.Alpha, group.EvaluationEDR,
                                       color="C1", marker="s", ms=3,
                                       label="Evaluation EDR", zorder=3)
            # Draw the limit last with open rings, so coincident training points
            # remain visible inside them rather than hiding the risk limit.
            limit_line, = ax.plot(group.Alpha, group.EDRLimit,
                                  color="black", linestyle=(0, (4, 3)),
                                  linewidth=1.2, marker="o", markersize=7,
                                  markerfacecolor="none", markeredgewidth=1.2,
                                  label="EDR limit (open circles)", zorder=5)
            middle = group[np.isclose(group.Alpha, .5)]
            if not middle.empty:
                target = middle.iloc[0]
                ax.annotate(f"EDR limit = {target.EDRLimit:.4f}\n(alpha = 0.5)",
                            xy=(target.Alpha, target.EDRLimit),
                            xytext=(.06, .65 if rho < 0 else .30),
                            textcoords="axes fraction", fontsize=8.5,
                            fontweight="bold", ha="left", va="center",
                            bbox=dict(boxstyle="round,pad=.4", facecolor="white",
                                      edgecolor="black", alpha=.95),
                            arrowprops=dict(arrowstyle="->", color="black",
                                            linewidth=1.1), zorder=6)
            ax.set(title=f"N={n:,}", xlabel="alpha", ylabel="EDR")
            ax.legend(handles=[limit_line, training_line, evaluation_line],
                      fontsize=7, loc="center right" if rho < 0 else "upper right")
            selected = group[group.Alpha.isin([0,.5,1])]
            distributions = [(r.Alpha, product_profits(np.array([r.Q1,r.Q2]),eval_demand).sum(axis=1)) for r in selected.itertuples()]
            if distributions:
                edges = np.linspace(min(x.min() for _,x in distributions), max(x.max() for _,x in distributions)+1e-8, 46)
                for alpha, profits in distributions:
                    counts, _ = np.histogram(profits, bins=edges)
                    axes[1,j].plot((edges[:-1]+edges[1:])/2, counts/len(profits), label=f"alpha={alpha:g}")
                axes[1,j].axvline(group.TargetZ.iloc[0], color="k", ls="--", label="profit target z")
            axes[1,j].set(xlabel="Evaluation total profit", ylabel="Relative frequency"); axes[1,j].legend(fontsize=7)
        finish_page(plot_folder, fig, f"Chapter framework: EDR compliance and profit distributions | rho={rho:+.1f}",
                    "Upper panels: black open circles and dashes mark the EDR limit; blue centers show Training EDR.\nOverlapping limit and training values indicate a binding constraint. Lower panels: the dashed line marks the separate profit target z.")

        fig, axes = plt.subplots(1, 3, figsize=(11.7, 8.3))
        for ax, n in zip(axes, TRAINING_SIZES):
            group = data[data.TrainingN == n]
            ax.plot(group.EvaluationEDR, group.EvaluationMeanProfit, "o-")
            for row in group.itertuples():
                ax.annotate(f"{row.Alpha:g}", (row.EvaluationEDR,row.EvaluationMeanProfit), fontsize=7, xytext=(3,3), textcoords="offset points")
            ax.set(title=f"N={n:,}", xlabel="Evaluation EDR", ylabel="Evaluation mean profit")
        finish_page(plot_folder, fig, f"Chapter framework: risk-return frontier | rho={rho:+.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 8.3))
    setups = chapter.drop_duplicates(["TrainingN","RhoTarget"])
    for n, group in setups.groupby("TrainingN"):
        group = group.sort_values("RhoTarget")
        axes[0].plot(group.RhoTarget,group.BaselineTrainingEDR,"o-",label=f"baseline N={n:,}")
        axes[0].plot(group.RhoTarget,group.MinimumTrainingEDR,"s--",label=f"minimum N={n:,}")
        axes[1].plot(group.RhoTarget,group.MaximumFeasibleEDRReductionPct,"o-",label=f"N={n:,}")
        axes[2].plot(group.RhoTarget,group.TargetZ,"o-",label=f"N={n:,}")
    for ax,label in zip(axes,["Training EDR","Maximum EDR reduction (%)","Profit target z"]):
        ax.set(xlabel="rho",ylabel=label); ax.legend(fontsize=7)
    finish_page(plot_folder,fig,"Feasible risk ranges across correlations", "Targets differ by rho and N; this is descriptive, not an isolated causal effect of correlation.")

    sample_size_comparison_plot(plot_folder,chapter)


def verify_solver():
    """Check that native LP functions match the original, then solve small cases."""
    import ast
    import inspect
    original_tree=ast.parse(ORIGINAL_SCRIPT.read_text(encoding="utf-8"))
    functions={n.name:n for n in original_tree.body if isinstance(n,ast.FunctionDef)}
    for name in ["find_glpk","solve_unconstrained_saa","solve_minimum_edr","solve_risk_constrained_saa"]:
        current=ast.parse(inspect.getsource(globals()[name])).body[0]
        assert ast.dump(current,include_attributes=False)==ast.dump(functions[name],include_attributes=False),name
    solver=find_glpk()
    for rho in [-.5,0,.5]:
        demand=demand_pool(100,TRAIN_SEED,rho)
        problem,base,minimum,z=risk_setup(demand,capacity_value(),solver)
        for alpha in [.5,1.0]:
            limit=(1-alpha)*base["EDR"]+alpha*minimum["EDR"]+EDR_FEASIBILITY_TOL
            result=problem.solve(limit=limit)
            assert result["Status"]=="optimal"
            actual=float(np.maximum(z-product_profits(result["Q"],demand).sum(axis=1),0).mean())
            assert abs(actual-result["EDR"])<1e-9
            assert actual<=limit+1e-5
        print(f"Original Pyomo/GLPK checks passed: rho={rho:+.1f}, N=100",flush=True)
    print("All four copied solver functions structurally match the original; small-sample solves passed.",flush=True)



if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Original-calibration training-size sensitivity")
    parser.add_argument("--verify",action="store_true",help="Small solver checks; no output files")
    parser.add_argument("--quick",action="store_true",help="Small diagnostic run; no output files")
    parser.add_argument("--plots-only", "--report-only", dest="plots_only", action="store_true",
                        help="Redraw PNG figures from CSV; no solving or reports")
    parser.add_argument("--publish-pending",action="store_true",help="Publish saved results after closing CSV viewers")
    args,_unknown=parser.parse_known_args()
    if args.publish_pending:
        publish_pending_outputs()
    elif args.verify:
        verify_solver()
    elif args.plots_only:
        make_plots(pd.read_csv(OUTPUT_FOLDER/"case1_training_results.csv"), OUTPUT_FOLDER/"plots")
        publish_pending_outputs()
    else:
        if args.quick:
            TRAINING_SIZES=[50,100,200]
            ALPHA_VALUES=[0,.5,1]
            EVALUATION_N=2000
            SAVE_RESULTS=False
        run_experiment()
