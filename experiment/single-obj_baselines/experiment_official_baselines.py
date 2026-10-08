"""Run the six official single-objective baselines on the 23 VAE tasks.

The script uses pymoo's RandomSearch/GA/DE/PSO/CMAES and EvoTorch's PGPE
through ``model.baselines.optimize_single_objective``.  Every completed run is
immediately appended to ``runs.csv`` so interrupted experiments can resume.
``summary.csv`` reports the median, quartiles and IQR over seeds 0--29.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines import optimize_single_objective
from probfun.fitness import (
    ackley,
    beale,
    griewank,
    himmelblau,
    levy,
    rastrigin,
    rosenbrock,
    weierstrass,
)
from scripts.eval import (
    distance_scale,
    energy_wrapper,
    fitness_target,
    point_entropy,
    set_seed,
)


EXPERIMENT_VERSION = "official_single_objective_baselines_v1"
ALGORITHMS = ("random", "ga", "de", "pso", "cma_es", "pgpe")
METRIC_NAMES = (
    "entropy_all",
    "entropy_top_k",
    "fitness_reward_all",
    "fitness_reward_top_k",
    "average_fitness_all",
    "average_fitness_top_k",
)

# Only the two initialization choices selected by the preliminary sensitivity
# check differ from the libraries' standard settings. All settings are fixed
# across every benchmark and dimension.
ALGORITHM_KWARGS = {
    "random": {},
    "ga": {},
    "de": {},
    "pso": {},
    "cma_es": {"sigma": 0.3},
    "pgpe": {
        "center_learning_rate": 0.1,
        "stdev_learning_rate": 0.1,
        "stdev_init": 0.5,
    },
}
ALGORITHM_LIBRARY = {
    "random": "pymoo",
    "ga": "pymoo",
    "de": "pymoo",
    "pso": "pymoo",
    "cma_es": "pymoo",
    "pgpe": "evotorch",
}


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    function_name: str
    function: Callable[[torch.Tensor], torch.Tensor]
    dimension: int
    objective: str
    eval_name: str

    @property
    def maximize(self) -> bool:
        return self.objective == "maximize"


# This is deliberately identical to experiment/VAE/experiment_single_obj.py.
PROBLEMS = (
    ProblemSpec("ackley_2d", "ackley", ackley, 2, "maximize", "ackley"),
    ProblemSpec("ackley_4d", "ackley", ackley, 4, "maximize", "ackley"),
    ProblemSpec("ackley_32d", "ackley", ackley, 32, "maximize", "ackley"),
    ProblemSpec("ackley_256d", "ackley", ackley, 256, "maximize", "ackley"),
    ProblemSpec("rastrigin_2d", "rastrigin", rastrigin, 2, "maximize", "rastrigin"),
    ProblemSpec("rastrigin_4d", "rastrigin", rastrigin, 4, "maximize", "rastrigin_4d"),
    ProblemSpec("rastrigin_32d", "rastrigin", rastrigin, 32, "maximize", "rastrigin_32d"),
    ProblemSpec("rastrigin_256d", "rastrigin", rastrigin, 256, "maximize", "rastrigin_256d"),
    ProblemSpec("rosenbrock_2d", "rosenbrock", rosenbrock, 2, "minimize", "rosenbrock"),
    ProblemSpec("beale_2d", "beale", beale, 2, "minimize", "beale"),
    ProblemSpec("himmelblau_2d", "himmelblau", himmelblau, 2, "minimize", "himmelblau"),
    ProblemSpec("griewank_10d", "griewank", griewank, 10, "minimize", "griewank_10d"),
    ProblemSpec("griewank_30d", "griewank", griewank, 30, "minimize", "griewank_30d"),
    ProblemSpec("griewank_100d", "griewank", griewank, 100, "minimize", "griewank_100d"),
    ProblemSpec("weierstrass_10d", "weierstrass", weierstrass, 10, "minimize", "weierstrass_10d"),
    ProblemSpec("weierstrass_30d", "weierstrass", weierstrass, 30, "minimize", "weierstrass_30d"),
    ProblemSpec("weierstrass_100d", "weierstrass", weierstrass, 100, "minimize", "weierstrass_100d"),
    ProblemSpec("levy_10d", "levy", levy, 10, "minimize", "levy_10d"),
    ProblemSpec("levy_30d", "levy", levy, 30, "minimize", "levy_30d"),
    ProblemSpec("levy_100d", "levy", levy, 100, "minimize", "levy_100d"),
    ProblemSpec("rosenbrock_10d", "rosenbrock", rosenbrock, 10, "minimize", "rosenbrock_10d"),
    ProblemSpec("rosenbrock_30d", "rosenbrock", rosenbrock, 30, "minimize", "rosenbrock_30d"),
    ProblemSpec("rosenbrock_100d", "rosenbrock", rosenbrock, 100, "minimize", "rosenbrock_100d"),
)
PROBLEM_BY_NAME = {problem.name: problem for problem in PROBLEMS}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def config_identifier(algorithm: str, problem: ProblemSpec, args) -> str:
    payload = {
        "version": EXPERIMENT_VERSION,
        "algorithm": algorithm,
        "algorithm_kwargs": ALGORITHM_KWARGS[algorithm],
        "problem": problem.name,
        "bounds": [-4.0, 4.0],
        "budget": args.budget,
        "population_size": args.population_size,
        "top_k": args.top_k,
        "objective_device": str(args.resolved_device),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def evaluation_population(result, population_size: int):
    """Return exactly one final population for metric computation.

    Most pymoo algorithms expose a complete final population. RandomSearch
    can expose one extra incumbent, while converged CMA-ES can expose only its
    terminal best point. In those two interface-specific cases, trim to the
    requested size or use the last requested number of evaluated solutions.
    """
    population = np.asarray(result.population, dtype=np.float64)
    if len(population) == population_size:
        return population, "official_final_population"
    if len(population) > population_size:
        return population[-population_size:], "trimmed_official_population"

    archive = np.asarray(result.archive_solutions, dtype=np.float64)
    if len(archive) < population_size:
        raise RuntimeError(
            f"only {len(archive)} evaluated solutions are available; "
            f"cannot construct a population of {population_size}"
        )
    return archive[-population_size:], "last_evaluated_population"


@torch.no_grad()
def evaluate_population(
    population: np.ndarray,
    problem: ProblemSpec,
    *,
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    """Compute exactly the six scalar metrics used by scripts.eval.evaluate."""
    pop = torch.as_tensor(population, dtype=torch.float32, device=device)
    fitness = problem.function(pop)
    top_count = min(int(top_k), len(fitness))
    if top_count < 1:
        raise ValueError("top_k must be positive")
    top_fitness, indices = torch.topk(
        fitness, top_count, largest=problem.maximize
    )
    top_population = pop[indices]

    reward_function = energy_wrapper(
        problem.function,
        target=fitness_target[problem.eval_name],
        scale=distance_scale[problem.eval_name],
    )
    return {
        "entropy_all": float(point_entropy(pop, fitness)),
        "entropy_top_k": float(point_entropy(top_population, top_fitness)),
        "fitness_reward_all": float(reward_function(pop).mean().item()),
        "fitness_reward_top_k": float(
            reward_function(top_population).mean().item()
        ),
        "average_fitness_all": float(fitness.mean().item()),
        "average_fitness_top_k": float(top_fitness.mean().item()),
    }


RUN_FIELDS = (
    "experiment_version",
    "config_id",
    "timestamp_utc",
    "algorithm",
    "library",
    "library_version",
    "task",
    "function",
    "dimension",
    "objective",
    "seed",
    "bounds",
    "requested_budget",
    "function_evaluations",
    "population_size",
    "evaluation_population_size",
    "evaluation_population_source",
    "top_k",
    "objective_device",
    "optimization_seconds",
    "algorithm_config_json",
    *METRIC_NAMES,
)


def append_csv(path: Path, fieldnames, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile_summary(values) -> dict[str, float]:
    array = np.fromiter(values, dtype=np.float64)
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    return {"median": median, "q1": q1, "q3": q3, "iqr": q3 - q1}


def write_summary(runs_path: Path, summary_path: Path) -> None:
    # If a CSV was manually concatenated, keep only its latest duplicate row.
    deduplicated = {}
    for row in read_rows(runs_path):
        key = (row["config_id"], row["algorithm"], row["task"], row["seed"])
        deduplicated[key] = row

    groups: dict[tuple, list[dict]] = {}
    for row in deduplicated.values():
        key = (
            row["config_id"],
            row["algorithm"],
            row["library"],
            row["library_version"],
            row["task"],
            row["function"],
            row["dimension"],
            row["objective"],
            row["requested_budget"],
            row["population_size"],
            row["top_k"],
            row["algorithm_config_json"],
        )
        groups.setdefault(key, []).append(row)

    summary_fields = [
        "experiment_version",
        "config_id",
        "algorithm",
        "library",
        "library_version",
        "task",
        "function",
        "dimension",
        "objective",
        "requested_budget",
        "population_size",
        "top_k",
        "algorithm_config_json",
        "completed_runs",
    ]
    summarized_fields = (
        *METRIC_NAMES,
        "function_evaluations",
        "optimization_seconds",
    )
    for field in summarized_fields:
        summary_fields.extend(
            (f"{field}_median", f"{field}_q1", f"{field}_q3", f"{field}_iqr")
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key in sorted(groups):
            (
                config_id,
                algorithm,
                library,
                library_version,
                task,
                function_name,
                dimension,
                objective,
                requested_budget,
                population_size,
                top_k,
                algorithm_config_json,
            ) = key
            rows = groups[key]
            output = {
                "experiment_version": EXPERIMENT_VERSION,
                "config_id": config_id,
                "algorithm": algorithm,
                "library": library,
                "library_version": library_version,
                "task": task,
                "function": function_name,
                "dimension": dimension,
                "objective": objective,
                "requested_budget": requested_budget,
                "population_size": population_size,
                "top_k": top_k,
                "algorithm_config_json": algorithm_config_json,
                "completed_runs": len(rows),
            }
            for field in summarized_fields:
                stats = percentile_summary(float(row[field]) for row in rows)
                for statistic, value in stats.items():
                    output[f"{field}_{statistic}"] = value
            writer.writerow(output)
    temporary.replace(summary_path)


def run_one(algorithm: str, problem: ProblemSpec, seed: int, args) -> dict:
    seed_everything(seed)
    algorithm_kwargs = dict(ALGORITHM_KWARGS[algorithm])
    config_id = config_identifier(algorithm, problem, args)
    started = time.perf_counter()
    result = optimize_single_objective(
        algorithm,
        problem.function,
        dimension=problem.dimension,
        bounds=(-4.0, 4.0),
        budget=args.budget,
        seed=seed,
        sense=problem.objective,
        population_size=args.population_size,
        objective_input="torch",
        torch_device=args.resolved_device,
        algorithm_kwargs=algorithm_kwargs,
        verbose=False,
    )
    optimization_seconds = time.perf_counter() - started
    population, population_source = evaluation_population(
        result, args.population_size
    )
    metrics = evaluate_population(
        population,
        problem,
        device=args.resolved_device,
        top_k=args.top_k,
    )
    library = ALGORITHM_LIBRARY[algorithm]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "config_id": config_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "library": library,
        "library_version": package_version(library),
        "task": problem.name,
        "function": problem.function_name,
        "dimension": problem.dimension,
        "objective": problem.objective,
        "seed": seed,
        "bounds": "[-4.0,4.0]",
        "requested_budget": args.budget,
        "function_evaluations": result.function_evaluations,
        "population_size": args.population_size,
        "evaluation_population_size": len(population),
        "evaluation_population_source": population_source,
        "top_k": args.top_k,
        "objective_device": str(args.resolved_device),
        "optimization_seconds": optimization_seconds,
        "algorithm_config_json": json.dumps(algorithm_kwargs, sort_keys=True),
        **metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, auto, cuda, or cuda:N")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(PROBLEM_BY_NAME),
        default=list(PROBLEM_BY_NAME),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "results" / "single_objective_baselines",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.budget < 1 or args.population_size < 2 or args.top_k < 1:
        raise ValueError("--budget, --population-size and --top-k are invalid")
    if args.budget < args.population_size:
        raise ValueError("--budget must be at least --population-size")
    if args.budget % args.population_size:
        raise ValueError("--budget must be divisible by --population-size")
    if "pgpe" in args.algorithms and args.population_size % 2:
        raise ValueError("PGPE requires an even --population-size")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.resolved_device = resolve_device(args.device)
    # EvoTorch is intentionally quiet in a 690-run PGPE matrix.
    if "pgpe" in args.algorithms:
        # EvoTorch configures its named loggers during import, so set their
        # levels only after the package has initialized.
        import evotorch  # noqa: F401

        logging.getLogger("evotorch").setLevel(logging.WARNING)
        logging.getLogger("evotorch.core").setLevel(logging.WARNING)
        logging.getLogger("evotorch.optimizers").setLevel(logging.WARNING)

    output_dir = args.output_dir.resolve()
    runs_path = output_dir / "runs.csv"
    summary_path = output_dir / "summary.csv"
    failures_path = output_dir / "failures.csv"
    if args.overwrite:
        for path in (runs_path, summary_path, failures_path):
            path.unlink(missing_ok=True)

    problems = [PROBLEM_BY_NAME[name] for name in args.tasks]
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    total = len(args.algorithms) * len(problems) * args.seeds
    print(
        f"runs={total}, algorithms={args.algorithms}, "
        f"seeds={args.seed_start}..{args.seed_start + args.seeds - 1}"
    )
    print(
        f"budget={args.budget}, population={args.population_size}, "
        f"top_k={args.top_k}, objective_device={args.resolved_device}"
    )
    print(f"output={output_dir}")
    for problem in problems:
        print(
            f"  {problem.name}: dimension={problem.dimension}, "
            f"objective={problem.objective}, eval_name={problem.eval_name}"
        )
    if args.dry_run:
        return

    completed = {
        (row["config_id"], row["algorithm"], row["task"], int(row["seed"]))
        for row in read_rows(runs_path)
    }
    failure_fields = (
        "timestamp_utc",
        "algorithm",
        "task",
        "seed",
        "error",
        "traceback",
    )

    position = 0
    completed_since_summary = 0
    try:
        for algorithm in args.algorithms:
            for problem in problems:
                config_id = config_identifier(algorithm, problem, args)
                for seed in seeds:
                    position += 1
                    key = (config_id, algorithm, problem.name, seed)
                    if key in completed:
                        print(
                            f"[{position}/{total}] skip: "
                            f"{algorithm}/{problem.name}/seed={seed}"
                        )
                        continue

                    print(
                        f"[{position}/{total}] run: "
                        f"{algorithm}/{problem.name}/seed={seed}"
                    )
                    try:
                        row = run_one(algorithm, problem, seed, args)
                        append_csv(runs_path, RUN_FIELDS, row)
                        completed.add(key)
                        completed_since_summary += 1
                        print(
                            "  done: "
                            f"average={row['average_fitness_all']:.6g}, "
                            f"top_k={row['average_fitness_top_k']:.6g}, "
                            f"fes={row['function_evaluations']}, "
                            f"seconds={row['optimization_seconds']:.2f}"
                        )
                        if completed_since_summary >= 10:
                            write_summary(runs_path, summary_path)
                            completed_since_summary = 0
                    except Exception as exc:
                        append_csv(
                            failures_path,
                            failure_fields,
                            {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "algorithm": algorithm,
                                "task": problem.name,
                                "seed": seed,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        print(f"  FAILED: {exc!r}", file=sys.stderr)
                        if args.fail_fast:
                            raise
    finally:
        if runs_path.exists():
            write_summary(runs_path, summary_path)

    print(f"raw results: {runs_path}")
    print(f"median/IQR summary: {summary_path}")
    if failures_path.exists():
        print(f"failures: {failures_path}")


if __name__ == "__main__":
    main()
