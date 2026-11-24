# -*- coding: utf-8 -*-
"""
Chapter 1: SAA experiments for the single-product Newsvendor problem.

Usage:
    python chapter1_saa_publication_plots.py
    python chapter1_saa_publication_plots.py --quick
    python chapter1_saa_publication_plots.py --method quantile

Dependencies: numpy, pandas, scipy, matplotlib.
Optional: Pyomo with GLPK or HiGHS (highspy).

The default method is LP. Quantile and LP computation
times should be reported separately.

At each sample size N, the candidate is the SAA decision
with the highest evaluation mean profit among the replications.

The same evaluation sample is used for candidate selection
and evaluation. The nominal 95% gap intervals use a plug-in
standard error without adjustment for selection or covariance.
"""
from pathlib import Path
import argparse
import json
import math
import platform
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from scipy.stats import norm
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

SEED = 20
SELLING_PRICE = 20.0
UNIT_COST = 8.0
SALVAGE_VALUE = 2.0
SHORTAGE_PENALTY = 5.0
MU_TARGET = 100.0
SIGMA_TARGET = 20.0
SCENARIO_SIZES = [50, 500, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000]
REPLICATIONS = 20
EVAL_SIZE = 100_000
OUTPUT_DIR = Path("chapter1_saa_best_solution_results")
Z95 = 1.96  
CO = UNIT_COST - SALVAGE_VALUE
CU = SELLING_PRICE - UNIT_COST + SHORTAGE_PENALTY
CRIT = CU / (CU + CO)
COLORS = {"train": "#0072B2", "validation": "#009E73", "exact": "#D55E00"}
DIST_COLORS = {"normal": "#0072B2", "uniform": "#009E73", "exponential": "#D55E00"}
DIST_IDS = {"normal": 1, "uniform": 2, "exponential": 3}


def profit_samples(q, demand):
    return (SELLING_PRICE * np.minimum(q, demand)
            + SALVAGE_VALUE * np.maximum(q - demand, 0.0)
            - SHORTAGE_PENALTY * np.maximum(demand - q, 0.0)
            - UNIT_COST * q)


def expected_profit(q, dist):
    shortage = dist["shortage"](q)
    return (SELLING_PRICE * (dist["mean"] - shortage)
            + SALVAGE_VALUE * (shortage + q - dist["mean"])
            - SHORTAGE_PENALTY * shortage - UNIT_COST * q)


def build_distributions():
    mu, sigma = MU_TARGET, SIGMA_TARGET
    # D=max(0,Z), Z~Normal(mu,sigma)
    mean_censored = mu * norm.cdf(mu / sigma) + sigma * norm.pdf(mu / sigma)
    def shortage_normal(q):
        z = (np.asarray(q) - mu) / sigma
        return sigma * norm.pdf(z) - (np.asarray(q) - mu) * norm.sf(z)
    half = sigma * np.sqrt(3.0)
    a, b = mu - half, mu + half
    def shortage_uniform(q):
        q = np.asarray(q, dtype=float)
        return np.where(q <= a, mu - q,
                        np.where(q >= b, 0.0, (b - q)**2 / (2 * (b - a))))
    def shortage_exponential(q):
        q = np.asarray(q, dtype=float)
        return np.where(q <= 0, mu - q, mu * np.exp(-q / mu))
    return {
        "normal": dict(label="Normal (100, 20), censored at zero",
                       sampler=lambda rng, n: np.maximum(0.0, rng.normal(mu, sigma, n)),
                       mean=mean_censored, shortage=shortage_normal,
                       q_star=mu + sigma * norm.ppf(CRIT), nominal_sd=sigma),
        "uniform": dict(label=f"Uniform ({a:.2f}, {b:.2f})",
                        sampler=lambda rng, n: rng.uniform(a, b, n),
                        mean=mu, shortage=shortage_uniform,
                        q_star=a + CRIT * (b - a), nominal_sd=sigma),
        "exponential": dict(label="Exponential (mean=100, SD=100)",
                            sampler=lambda rng, n: rng.exponential(mu, n),
                            mean=mu, shortage=shortage_exponential,
                            q_star=-mu * np.log1p(-CRIT), nominal_sd=mu),
    }


def empirical_saa_order(demand):
    """An actual maximizer of the empirical newsvendor objective.

    Linear interpolation (np.quantile's default) need not be SAA-optimal.
    Use the ceil(N*CRIT)-th order statistic. If N*CRIT is an integer,
    either adjacent endpoint is optimal; this chooses the lower endpoint.
    """
    k = min(len(demand) - 1, max(0, math.ceil(len(demand) * CRIT) - 1))
    return float(np.partition(demand, k)[k])


class SAASolver:
    def __init__(self, method="lp", solver="auto"):
        self.method = method
        self.opt = None
        if method == "quantile":
            if solver != "auto":
                raise ValueError("--solver is only applicable to --method lp")
            self.name = "empirical_quantile"
            return
        if solver in ("auto", "glpk", "appsi_highs"):
            try:
                import pyomo.environ as pyo
            except ImportError:
                if solver != "auto":
                    raise RuntimeError("Install pyomo and the requested solver.")
            else:
                for name in (["glpk", "appsi_highs"] if solver == "auto" else [solver]):
                    candidate = pyo.SolverFactory(name)
                    if candidate.available(exception_flag=False):
                        self.opt, self.name, self.pyo = candidate, name, pyo
                        return
                if solver != "auto":
                    raise RuntimeError(f"Solver {solver} is unavailable.")
        self.name = "scipy_highs"  # still the same two-stage LP

    def solve(self, demand):
        # Includes LP construction + solve; excludes scenario generation/evaluation.
        started = time.perf_counter()
        if self.method == "quantile":
            q = empirical_saa_order(demand)
            obj = float(profit_samples(q, demand).mean())
        elif self.opt is not None:
            pyo, n = self.pyo, len(demand)
            model = pyo.ConcreteModel()
            model.q = pyo.Var(domain=pyo.NonNegativeReals)
            model.sold = pyo.Var(range(n), domain=pyo.NonNegativeReals)
            model.left = pyo.Var(range(n), domain=pyo.NonNegativeReals)
            model.short = pyo.Var(range(n), domain=pyo.NonNegativeReals)
            model.balance_q = pyo.Constraint(range(n), rule=lambda m, s: m.q == m.sold[s] + m.left[s])
            model.balance_d = pyo.Constraint(range(n), rule=lambda m, s: demand[s] == m.sold[s] + m.short[s])
            model.obj = pyo.Objective(expr=pyo.quicksum(
                SELLING_PRICE * model.sold[s] + SALVAGE_VALUE * model.left[s]
                - SHORTAGE_PENALTY * model.short[s] for s in range(n)) / n
                - UNIT_COST * model.q, sense=pyo.maximize)
            result = self.opt.solve(model, tee=False)
            if not pyo.check_optimal_termination(result):
                raise RuntimeError(f"LP failed: {result.solver.termination_condition}")
            q, obj = float(pyo.value(model.q)), float(pyo.value(model.obj))
        else:
            # Variables: [q, sold[0:N], left[0:N], short[0:N]].
            n = len(demand)
            i = np.arange(n)
            rows = np.concatenate([i, i, i, n+i, n+i])
            cols = np.concatenate([np.zeros(n, dtype=int), 1+i, 1+n+i, 1+i, 1+2*n+i])
            values = np.concatenate([np.ones(n), -np.ones(n), -np.ones(n), np.ones(n), np.ones(n)])
            aeq = coo_matrix((values, (rows, cols)), shape=(2*n, 1+3*n)).tocsr()
            c = np.concatenate([[UNIT_COST], np.full(n, -SELLING_PRICE/n),
                                np.full(n, -SALVAGE_VALUE/n), np.full(n, SHORTAGE_PENALTY/n)])
            result = linprog(c, A_eq=aeq, b_eq=np.concatenate([np.zeros(n), demand]),
                             bounds=(0, None), method="highs")
            if not result.success:
                raise RuntimeError(f"LP failed: {result.message}")
            q, obj = float(result.x[0]), float(-result.fun)
        elapsed = time.perf_counter() - started
        
        direct = float(profit_samples(q, demand).mean())
        exact_sample = float(profit_samples(empirical_saa_order(demand), demand).mean())
        if not np.isclose(obj, direct, rtol=1e-7, atol=1e-5):
            raise RuntimeError("LP objective and scenario profit disagree.")
        if not np.isclose(direct, exact_sample, rtol=1e-7, atol=1e-5):
            raise RuntimeError("SAA objective failed the empirical-quantile check.")
        return obj, q, elapsed


def rng_for(name, stream, n=0, r=0):
    # Streams: 0=evaluation; 1=training. Candidate is selected from training runs.
    
    return np.random.default_rng(np.random.SeedSequence([SEED, DIST_IDS[name], stream, n, r]))


def configure_plotting():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 8,
                         "lines.linewidth": 1.6, "lines.markersize": 4,
                         "savefig.dpi": 300, "axes.spines.top": False,
                         "axes.spines.right": False})


def format_n(ax, ns):
    ax.set_xscale("log")
    
    ticks = np.asarray(ns)
    if len(ticks) > 8:
        chosen = [ticks[0]]
        for n in ticks[1:-1]:
            if n / chosen[-1] >= 1.8 and ticks[-1] / n >= 1.8:
                chosen.append(n)
        chosen.append(ticks[-1])
        ticks = np.asarray(chosen)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([
        str(int(n)) if n < 1000 else f"{n/1000:g}k" for n in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Training sample size N (log scale)")
    ax.grid(axis="y", alpha=.25)
    ax.margins(x=.04)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def save_figure(fig, output, filename):
    fig.savefig(output / f"{filename}.png", bbox_inches="tight")
    fig.savefig(output / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_distribution(name, dist, summary, bounds, m, output, solver):
    ns = summary.index.to_numpy(dtype=float)
    opt = float(summary["AnalyticalOptimalProfit"].iloc[0])
    denom = abs(opt)
    tag = f"{dist['label']}; M={m}"
    # Fig 1: absolute means + signed relative difference from the true optimum.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for key, label, marker in [("Train", "Mean training optimum", "o"),
                                ("Validation", "Mean evaluated profit", "s"),
                                ("Exact", "Mean exact profit of SAA orders", "^")]:
        y = summary[f"{key}_mean"].to_numpy()
        axes[0].plot(ns, y, marker+"-", color=COLORS[key.lower()], label=label)
        axes[1].plot(ns, 100*(y-opt)/denom, marker+"-", color=COLORS[key.lower()], label=label)
    axes[0].axhline(opt, color="black", ls="--", label=r"Analytical optimum $f(Q^*)$")
    axes[1].axhline(0, color="black", ls="--")
    axes[0].set(title="A. Mean profit", ylabel="Mean profit (monetary units)")
    axes[1].set(title="B. Difference from analytical optimum", ylabel="Signed difference (% of optimal profit)")
    axes[0].legend()
    for ax in axes: format_n(ax, ns)
    fig.suptitle(f"SAA objective convergence | {tag}")
    save_figure(fig, output, f"fig01_objective_convergence_{name}")

    # Fig 2: SD describes order variability, not a 95% confidence interval.
    fig, ax = plt.subplots(figsize=(8.7, 5), layout="constrained")
    mean, sd = summary["Q_mean"].to_numpy(), summary["Q_sd"].to_numpy()
    ax.plot(ns, mean, "o-", color=COLORS["train"], label="Mean SAA order quantity")
    ax.fill_between(ns, mean-sd, mean+sd, alpha=.18, color=COLORS["train"],
                    label=f"Mean +/- 1 SD across {m} training runs")
    ax.axhline(dist["q_star"], color="black", ls="--",
               label=f"Analytical Q* = {dist['q_star']:.3f}")
    ax.set(title=f"Convergence and stability of SAA order quantity\n{tag}",
           ylabel="Order quantity Q (units)")
    format_n(ax, ns); ax.legend()
    save_figure(fig, output, f"fig02_order_stability_{name}")

    # Fig 3: two distinct errors, both normalized by the same known benchmark.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, col, title, color in [
        (axes[0], "TrainingOptimismPct_mean", "A. In-sample optimism", COLORS["train"]),
        (axes[1], "EvaluationErrorPct_mean", "B. Monte Carlo evaluation error", COLORS["validation"])]:
        ax.plot(ns, summary[col], "o-", color=color)
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set(title=title, ylabel="Signed difference (% of optimal profit)")
        format_n(ax, ns)
    fig.suptitle(f"Training and evaluation diagnostics | {tag}")
    save_figure(fig, output, f"fig03_optimism_evaluation_error_{name}")

    # Fig 4: paper-style plug-in interval, analytical regret, and nominal width.
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.5), layout="constrained")
    axes[0].errorbar(ns, bounds["EstimatedGapPct"], yerr=bounds["GapCI95HalfWidthPct"],
                     fmt="o-", capsize=3, color=COLORS["train"],
                     label="Estimated gap +/- 1.96 plug-in SE")
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(title="A. Estimated statistical gap and nominal 95% plug-in interval",
                ylabel="Estimated gap (% of optimal profit)")
    axes[0].legend()
    axes[1].plot(ns, bounds["CandidateTrueRegretPct"], "s-", color=COLORS["exact"],
                 label="Best evaluated SAA decision: analytical true regret")
    axes[1].axhline(0, color="black", ls="--", lw=1)
    axes[1].set(title="B. Analytical true regret of the same candidate",
                ylabel="True regret (% of optimal profit)")
    axes[1].legend()
    axes[2].plot(ns, bounds["GapCI95FullWidthPct"], "o-", color="#CC79A7")
    axes[2].set(title="C. Full width of the nominal plug-in interval",
                ylabel="Full interval width (% of optimal profit)")
    for ax in axes: format_n(ax, ns)
    fig.suptitle(f"Statistical gap assessment | {tag}\nBest of {m} evaluated SAA decisions; interval does not adjust for selection")
    save_figure(fig, output, f"fig04_statistical_gap_{name}")

    # Fig 5: clock time, not process CPU time; per distribution and N.
    fig, ax = plt.subplots(figsize=(8.7, 5), layout="constrained")
    ax.plot(ns, bounds["TotalOptimizationWallSeconds"], "o-", color=COLORS["exact"],
            label=f"Optimization: {m} SAA runs (candidate selected from these)")
    ax.plot(ns, bounds["TotalExperimentWallSeconds"], "s--", color=COLORS["train"],
            label="Complete experiment (excluding plots / file writing)")
    ax.set(title=f"Computational cost | {solver.name}\n{tag}",
           ylabel="Total elapsed wall time (seconds)")
    format_n(ax, ns); ax.legend()
    save_figure(fig, output, f"fig05_total_time_{name}")


def plot_comparison(summaries, m, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name, s in summaries.items():
        ns = s.index.to_numpy(dtype=float)
        for ax, col in [(axes[0], "TrainingOptimismPct_mean"), (axes[1], "Q_sd")]:
            ax.plot(ns, s[col], "o-", color=DIST_COLORS[name], label=name.title())
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set(title="A. In-sample optimism", ylabel="Mean training - exact profit (% of optimal profit)")
    axes[1].set(title="B. Order variability across training runs", ylabel="SD of SAA order quantity (units)")
    for ax in axes: format_n(ax, ns); ax.legend()
    fig.suptitle(f"SAA convergence across demand distributions | M={m}")
    save_figure(fig, output, "fig06_cross_distribution_comparison")


def run_experiment(name, dist, args, solver, output):
    qstar = dist["q_star"]
    fstar = float(expected_profit(qstar, dist))
    denom = abs(fstar)
    if denom <= 1e-12:
        raise ValueError("Percent metrics require a nonzero analytical optimal profit.")
    eval_d = dist["sampler"](rng_for(name, 0), args.eval_size)
    rows, bound_rows = [], []
    print(f"\n{name}: Q*={qstar:.4f}, f*={fstar:.4f}, solver={solver.name}", flush=True)
    for n in args.sizes:
        n_started = time.perf_counter()
        local = []
        for r in range(args.replications):
            demand = dist["sampler"](rng_for(name, 1, n, r), n)
            train, q, elapsed = solver.solve(demand)
            evaluation = profit_samples(q, eval_d)
            validation = float(evaluation.mean())
            exact = float(expected_profit(q, dist))
            row = dict(ScenarioSize=n, Replication=r+1, TrainObj=train,
                       ValidationObj=validation, ExactObj=exact, OptimalQ=q,
                       EmpiricalSAAQ=empirical_saa_order(demand),
                       OptimizationWallSeconds=elapsed,
                       TrainingOptimism=train-exact, EvaluationError=validation-exact,
                       TrueRegret=fstar-exact, SignedOrderError=q-qstar,
                       AbsoluteOrderError=abs(q-qstar),
                       TrainingOptimismPct=100*(train-exact)/denom,
                       EvaluationErrorPct=100*(validation-exact)/denom,
                       TrueRegretPct=100*(fstar-exact)/denom,
                       ValidationScenarioSD=float(evaluation.std(ddof=1)))
            rows.append(row); local.append(row)
        # Operational definition of "best solution": highest common-pool
        # evaluated profit among the M SAA decisions. ExactObj and TrainObj
      
        selected = max(local, key=lambda row: row["ValidationObj"])
        for row in local:
            row["IsSelectedCandidate"] = row is selected
        cand_train, cand_q = selected["TrainObj"], selected["OptimalQ"]
        cand_profits = profit_samples(cand_q, eval_d)
        cand_val, cand_exact = selected["ValidationObj"], selected["ExactObj"]
        trains = np.array([r["TrainObj"] for r in local])
        train_var = float(trains.var(ddof=1) / args.replications)
        evaluation_var = float(cand_profits.var(ddof=1) / args.eval_size)
      
        se = np.sqrt(train_var + evaluation_var)
        gap = float(trains.mean() - cand_val)
        pct = 100 / denom
        elapsed_total = time.perf_counter() - n_started
        bound_rows.append(dict(
            ScenarioSize=n, M=args.replications, EvaluationSize=args.eval_size,
            Solver=solver.name, Method=args.method, AnalyticalOptimalProfit=fstar,
            MeanTrainObj=float(trains.mean()), VarMeanTrainObj=train_var,
            CandidateRule="highest evaluated profit among training SAA decisions; first replication breaks ties",
            CandidateReplication=selected["Replication"], CandidateCount=args.replications,
            CandidateUsesEvaluationPoolForSelection=True,
            IntervalMethod="nominal normal plug-in; no selection or covariance adjustment",
            CandidateQ=cand_q, CandidateTrainObj=cand_train,
            CandidateValidationObj=cand_val, CandidateExactObj=cand_exact,
            VarCandidateValidationMean=evaluation_var,
            EstimatedGap=gap, SEEstimatedGap=float(se), EstimatedGapPct=pct*gap,
            GapCI95Lower=gap-Z95*se, GapCI95Upper=gap+Z95*se,
            GapCI95LowerPct=pct*(gap-Z95*se), GapCI95UpperPct=pct*(gap+Z95*se),
            GapCI95HalfWidthPct=pct*Z95*se, GapCI95FullWidthPct=pct*2*Z95*se,
            # Nominal one-sided reference endpoint, not a certified regret bound.
            GapUpperReferenceOneSided95=gap+norm.ppf(.95)*se,
            CandidateTrueRegret=fstar-cand_exact,
            CandidateTrueRegretPct=pct*(fstar-cand_exact),
            CandidateEvaluationError=cand_val-cand_exact,
            TrainingOptimizationWallSeconds=sum(r["OptimizationWallSeconds"] for r in local),
            # Selected run time is already part of the training total.
            CandidateOptimizationWallSeconds=selected["OptimizationWallSeconds"],
            TotalOptimizationWallSeconds=sum(r["OptimizationWallSeconds"] for r in local),
            TotalExperimentWallSeconds=elapsed_total))
        print(f"  N={n:6d}: best run={selected['Replication']:2d}, gap={pct*gap: .5f}%, plug-in SE={pct*se:.5f}%, time={elapsed_total:.2f}s", flush=True)
    df = pd.DataFrame(rows)
    bounds = pd.DataFrame(bound_rows)
    summary = df.groupby("ScenarioSize").agg(
        Train_mean=("TrainObj", "mean"), Train_sd=("TrainObj", "std"),
        Validation_mean=("ValidationObj", "mean"), Exact_mean=("ExactObj", "mean"),
        Q_mean=("OptimalQ", "mean"), Q_sd=("OptimalQ", "std"),
        OptimizationWallSeconds_mean=("OptimizationWallSeconds", "mean"),
        TrainingOptimismPct_mean=("TrainingOptimismPct", "mean"),
        EvaluationErrorPct_mean=("EvaluationErrorPct", "mean"),
        TrueRegretPct_mean=("TrueRegretPct", "mean"),
        AbsoluteOrderError_mean=("AbsoluteOrderError", "mean"))
    summary["AnalyticalOptimalProfit"] = fstar
    summary["AnalyticalOptimalQ"] = qstar
    summary["M"] = args.replications
    summary["EvaluationSize"] = args.eval_size
    summary["Solver"] = solver.name
    df.to_csv(output / f"saa_replications_{name}.csv", index=False)
    bounds.to_csv(output / f"saa_statistical_bounds_{name}.csv", index=False)
    summary.to_csv(output / f"saa_summary_{name}.csv")
    plot_distribution(name, dist, summary, bounds, args.replications, output, solver)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="N=[50,500,2000]; M=20; evaluation=100000")
    parser.add_argument("--method", choices=["lp", "quantile"], default="lp")
    parser.add_argument("--solver", choices=["auto", "glpk", "appsi_highs", "scipy_highs"], default="auto")
    parser.add_argument("--sizes", nargs="+", type=int, help="override training sample sizes")
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--eval-size", type=int, default=EVAL_SIZE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.sizes = sorted(set(args.sizes or ([50, 500, 2000] if args.quick else SCENARIO_SIZES)))
    if args.replications < 2 or args.eval_size < 2 or min(args.sizes) < 1:
        parser.error("M and evaluation size must be >=2; all N must be >=1")
    return args


def main():
    args = parse_args()
    solver = SAASolver(args.method, args.solver)
    # Distinct defaults prevent quick/quantile results overwriting full LP results.
    output = args.output_dir or Path(str(OUTPUT_DIR)
                                    + ("_quick" if args.quick else "")
                                    + ("_quantile" if args.method == "quantile" else ""))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}. Use a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    dists = build_distributions()
    metadata = dict(seed=SEED, sizes=args.sizes, replications=args.replications,
                    evaluation_size=args.eval_size, method=args.method, solver=solver.name,
                    python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    cost_parameters=dict(p=SELLING_PRICE, c=UNIT_COST, s=SALVAGE_VALUE, shortage_penalty=SHORTAGE_PENALTY),
                    critical_ratio=CRIT, percentages="100 * metric / abs(analytical optimal profit)",
                    candidate_rule="highest evaluation mean among training SAA decisions at each N; ties choose first replication",
                    candidate_independent_of_training=False,
                    interval="nominal pointwise normal plug-in interval, z=1.96; selection/covariance not adjusted; coverage not guaranteed",
                    gap_target="mean training optimum minus best evaluated SAA decision; distinct from analytical true regret",
                    timing="elapsed wall seconds; optimization includes model build and solve",
                    evaluation_pool="drawn independently of training samples; reused for selecting and reporting candidate, and across N within each distribution",
                    random_streams="SeedSequence([seed, distribution_id, stream, N, replication]); 0=evaluation, 1=training")
    with (output / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    summaries = {name: run_experiment(name, dist, args, solver, output) for name, dist in dists.items()}
    plot_comparison(summaries, args.replications, output)
    print(f"\nDone: {output.resolve()}\n16 figures, each saved as PNG and PDF; 9 CSV tables + metadata.")


if __name__ == "__main__":
    main()
