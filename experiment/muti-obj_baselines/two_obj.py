# baseline models
from model.baselines.moead import moead
from model.baselines.nsga2 import nsga2
from model.baselines.nsga3 import nsga3
from model.baselines.rvea import rvea

# metrics
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator

# other imports
import json
import os
import numpy as np
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed


# evaluate function
def evaluate_multiobjective(true_pf, approximate_pf):
    gd = GD(true_pf)
    igd = IGD(true_pf)
    sp = SpacingIndicator()

    return {
        'GD': gd(approximate_pf),
        'IGD': igd(approximate_pf),
        'SP': sp.do(approximate_pf)
    }


# true pareto front
def get_true_pf(problem, n_points=1000):
    try:
        return problem.pareto_front(n_pareto_points=n_points)
    except TypeError:
        return problem.pareto_front(n_points=n_points)


# test seeds (optional env cap: TWO_OBJ_SEEDS=N, useful for smoke tests / sharding)
n_seeds = int(os.environ.get('TWO_OBJ_SEEDS', '30'))
seeds = list(range(0, n_seeds))


# problems to test: ZDT1 - ZDT6
PROBLEMS = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt5', 'zdt6']


# algorithms to test: (runner, n_gen)
ALGORITHM_RUNNERS = {
    'NSGA-II': (nsga2, 400),
    'NSGA-III': (nsga3, 400),
    'MOEA/D': (moead, 400),
    'RVEA': (rvea, 400)
}


# ---------------------------------------------------------------------------
# Parallel training
# ---------------------------------------------------------------------------
# 6 problems x 30 seeds x 4 algorithms = 720 fully independent trials, run
# concurrently across a process pool. Results are identical to the sequential
# loop (every trial keeps its own seed) -- only wall-clock shrinks.
#
# Implementation notes (same as three_obj.py):
#  * run_one is a top-level function so it is picklable for ProcessPoolExecutor.
#  * The `if __name__ == '__main__':` guard is required (Windows workers
#    re-import this module).
#  * Problems / true PFs are cached per process (built once per worker).
#  * Set N_WORKERS to control concurrency (default: all cores).
# ---------------------------------------------------------------------------

_problem_cache = {}
_true_pf_cache = {}


def _get_problem(pname):
    if pname not in _problem_cache:
        _problem_cache[pname] = get_problem(pname)
    return _problem_cache[pname]


def _get_true_pf(pname):
    if pname not in _true_pf_cache:
        _true_pf_cache[pname] = get_true_pf(_get_problem(pname))
    return _true_pf_cache[pname]


def run_one(job):
    """Run a single (problem, algorithm, seed) trial and return its metrics."""
    pname, name, seed = job
    runner, n_gen = ALGORITHM_RUNNERS[name]
    try:
        problem = _get_problem(pname)
        algorithm = runner(pop_size=100, n_obj=problem.n_obj)
        res = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            seed=seed,
            verbose=False
        )
        result = evaluate_multiobjective(_get_true_pf(pname), res.F)
        return pname, name, seed, result
    except Exception as e:
        print(f"Error in {name} with seed {seed}: {e}", flush=True)
        return pname, name, seed, None


def main():
    n_workers = int(os.environ.get('N_WORKERS', os.cpu_count() or 1))
    jobs = [
        (pname, name, seed)
        for pname in PROBLEMS
        for name in ALGORITHM_RUNNERS
        for seed in seeds
    ]
    print(f"Total trials: {len(jobs)} | workers: {n_workers}", flush=True)

    # dispatch all trials across the process pool
    collected = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            pname, name, seed, result = fut.result()
            collected[(pname, name, seed)] = result
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] completed", flush=True)

    # assemble results in deterministic seed order (same as the sequential loop)
    all_results = {
        p: {name: {metric: [] for metric in ['GD', 'IGD', 'SP']} for name in ALGORITHM_RUNNERS}
        for p in PROBLEMS
    }
    for pname in PROBLEMS:
        for name in ALGORITHM_RUNNERS:
            for seed in seeds:
                result = collected.get((pname, name, seed))
                if result is None:
                    for metric in all_results[pname][name]:
                        all_results[pname][name][metric].append(np.nan)
                else:
                    for metric in all_results[pname][name]:
                        all_results[pname][name][metric].append(result[metric])

    print("\n" + "=" * 60)
    print("Final Results over 30 runs")

    for pname in PROBLEMS:
        print(f"\n########## {pname.upper()} ##########")
        for name in ALGORITHM_RUNNERS:
            print(f"\n【{name}】")
            for metric in ['GD', 'IGD', 'SP']:
                values = np.array(all_results[pname][name][metric])
                median_val = np.nanmedian(values)
                q1 = np.nanpercentile(values, 25)
                q3 = np.nanpercentile(values, 75)
                print(f"{metric:3s}: {median_val:.6f} (IQR: {q3 - q1:.6f})")

    summary = {}
    for pname in PROBLEMS:
        summary[pname] = {}
        for name in ALGORITHM_RUNNERS:
            summary[pname][name] = {}
            for metric in ['GD', 'IGD', 'SP']:
                values = np.array(all_results[pname][name][metric])
                summary[pname][name][metric] = {
                    'median': float(np.nanmedian(values)),
                    'iqr': float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
                }

    with open('zdt_compare.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    main()
