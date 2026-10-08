"""Run the PVAE benchmark on ZDT and DTLZ multi-objective problems.

"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.VAEs import PVAE
from probfun.mutifitness import (
    dtlz2,
    dtlz4,
    dtlz7,
    zdt1,
    zdt2,
    zdt3,
    zdt4,
    zdt5,
    zdt6,
)
from scripts.eval import evaluate_multi_objective, set_seed
from scripts.train import train_multiobjective
from scripts.train_multiobjective import simplex_reference_directions


EXPERIMENT_VERSION = "pvae_zdt_dtlz_decoder_only_v1"
METRICS = ("GD", "IGD", "SP")

# Add shorter schedules only after a per-problem timing/quality validation.
VALIDATED_EPOCHS = {
    "zdt1": 100,
    "zdt2": 100,
    "zdt3": 100,
    "zdt4": 100,
    "zdt5": 100,
    "zdt6": 100,
    "dtlz2_3obj": 100,
    "dtlz4_3obj": 100,
    "dtlz7_3obj": 100,
    "dtlz2_5obj": 100,
    "dtlz7_5obj": 100,
}


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    function_name: str
    objective_function: Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
    decision_dim: int
    objective_count: int
    front_kind: str
    binary_decisions: bool = False


PROBLEMS = (
    ProblemSpec("zdt1", "zdt1", zdt1, 30, 2, "zdt1"),
    ProblemSpec("zdt2", "zdt2", zdt2, 30, 2, "zdt2"),
    ProblemSpec("zdt3", "zdt3", zdt3, 30, 2, "zdt3"),
    ProblemSpec("zdt4", "zdt4", zdt4, 10, 2, "zdt1"),
    ProblemSpec("zdt5", "zdt5", zdt5, 80, 2, "zdt5", True),
    ProblemSpec("zdt6", "zdt6", zdt6, 10, 2, "zdt6"),
    ProblemSpec(
        "dtlz2_3obj", "dtlz2", partial(dtlz2, num_objectives=3), 12, 3, "dtlz2"
    ),
    ProblemSpec(
        "dtlz4_3obj", "dtlz4", partial(dtlz4, num_objectives=3), 12, 3, "dtlz2"
    ),
    ProblemSpec(
        "dtlz7_3obj", "dtlz7", partial(dtlz7, num_objectives=3), 22, 3, "dtlz7"
    ),
    ProblemSpec(
        "dtlz2_5obj", "dtlz2", partial(dtlz2, num_objectives=5), 14, 5, "dtlz2"
    ),
    ProblemSpec(
        "dtlz7_5obj", "dtlz7", partial(dtlz7, num_objectives=5), 24, 5, "dtlz7"
    ),
)
PROBLEM_BY_NAME = {problem.name: problem for problem in PROBLEMS}


def previous_result_preset(problem: ProblemSpec) -> dict:
    """Return the final parameter family used by the earlier PVAE tests."""
    config = {
        "hidden_dim": 256,
        "latent_dim": 16,
        "epochs": 2000,
        "batch_size": 1024,
        "learning_rate": 3e-4,
        "elite_fraction": 0.2,
        "rank_temperature": 0.2,
        "exploration_fraction": 0.2,
        "train_steps": 5,
        "div_weight": 0.002,
        "beta": 1e-3,
        "set_loss_weight": 1.0,
        "reconstruction_weight": 0.25,
        "archive_size": 512,
        "anchor_count": 512,
        "initial_polish_steps": 32,
        "dominance_polish_steps": 64,
        "dominance_step_start": 1.0,
        "dominance_step_end": 1e-4,
        "calibration_steps": 1000,
        "calibration_learning_rate": 1e-3,
        "calibration_latent_repeats": 2,
        "calibration_endpoint_weight": 10.0,
        "calibration_batch_size": 256,
        "calibration_beta": 1e-4,
        "calibration_prior_weight": 0.25,
        "calibration_latent_weight": 0.01,
        "calibration_objective_weight": 0.1,
        "calibration_objective_step": 1e-3,
        "calibration_objective_batch_size": 32,
        "discontinuity_gap_factor": 16.0,
        "axis_coverage_points": 512,
        "axis_coverage_max_dimensions": 2,
        "binary_decisions": problem.binary_decisions,
        "binary_group_size": 1,
        "preference_polish_steps": 64,
        "preference_pbi_penalty": 10.0,
    }
    if problem.objective_count == 3:
        config.update(
            batch_size=2048,
            archive_size=2048,
            anchor_count=2048,
            axis_coverage_points=2048,
        )
    elif problem.objective_count == 5:
        config.update(
            hidden_dim=512,
            latent_dim=32,
            batch_size=4096,
            archive_size=4096,
            anchor_count=4096,
            axis_coverage_points=4096,
        )
    if problem.binary_decisions:
        config.update(hidden_dim=512, latent_dim=32)
    config["epochs"] = VALIDATED_EPOCHS.get(problem.name, config["epochs"])
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def _zdt_continuous_front(kind: str, count: int) -> torch.Tensor:
    if kind == "zdt3":
        intervals = (
            (0.0, 0.0830015349),
            (0.1822287280, 0.2577623634),
            (0.4093136748, 0.4538821041),
            (0.6183967944, 0.6525117038),
            (0.8233317983, 0.8518328654),
        )
        lengths = np.asarray([right - left for left, right in intervals])
        allocation = np.maximum(2, np.floor(count * lengths / lengths.sum()).astype(int))
        while allocation.sum() < count:
            allocation[np.argmax(lengths / allocation)] += 1
        while allocation.sum() > count:
            index = int(np.argmax(allocation))
            if allocation[index] <= 2:
                break
            allocation[index] -= 1
        f1 = torch.cat(
            [
                torch.linspace(left, right, int(points), dtype=torch.float64)
                for (left, right), points in zip(intervals, allocation)
            ]
        )
        f2 = 1.0 - torch.sqrt(f1) - f1 * torch.sin(10.0 * torch.pi * f1)
        return torch.stack((f1, f2), dim=1)

    lower = 0.2807753191 if kind == "zdt6" else 0.0
    f1 = torch.linspace(lower, 1.0, count, dtype=torch.float64)
    if kind in {"zdt2", "zdt6"}:
        f2 = 1.0 - f1.square()
    else:
        f2 = 1.0 - torch.sqrt(f1)
    return torch.stack((f1, f2), dim=1)


def _zdt5_front() -> torch.Tensor:
    # Construct all 31 canonical Pareto points through the public objective.
    decisions = torch.zeros(31, 80, dtype=torch.float64)
    for ones in range(31):
        decisions[ones, :ones] = 1.0
    decisions[:, 30:] = 1.0
    return torch.stack(zdt5(decisions), dim=1)


def _dtlz_spherical_front(count: int, objective_count: int) -> torch.Tensor:
    directions = simplex_reference_directions(
        count, objective_count, torch.device("cpu"), torch.float64
    )
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _dtlz7_front(count: int, objective_count: int) -> torch.Tensor:
    # On the DTLZ7 front, each free objective occupies one of two intervals.
    # Sobol points cover all 2**(M-1) disconnected components deterministically.
    engine = torch.quasirandom.SobolEngine(objective_count - 1, scramble=False)
    unit = engine.draw(count).to(torch.float64)
    choose_upper = unit >= 0.5
    local = torch.remainder(unit * 2.0, 1.0)
    lower = local * 0.251412
    upper = 0.631627 + local * (0.859401 - 0.631627)
    first = torch.where(choose_upper, upper, lower)
    final = 2.0 * objective_count - (
        first * (1.0 + torch.sin(3.0 * torch.pi * first))
    ).sum(dim=1)
    return torch.cat((first, final[:, None]), dim=1)


def make_true_front(problem: ProblemSpec, count: int) -> torch.Tensor:
    if problem.front_kind.startswith("zdt") and problem.front_kind != "zdt5":
        return _zdt_continuous_front(problem.front_kind, count)
    if problem.front_kind == "zdt5":
        return _zdt5_front()
    if problem.front_kind == "dtlz2":
        return _dtlz_spherical_front(count, problem.objective_count)
    if problem.front_kind == "dtlz7":
        return _dtlz7_front(count, problem.objective_count)
    raise ValueError(f"Unknown true-front kind: {problem.front_kind}")


def build_model(problem: ProblemSpec, config: dict, device: torch.device) -> PVAE:
    model = PVAE(
        input_dim=config["latent_dim"],
        latent_dim=problem.decision_dim,
        hidden_dim=config["hidden_dim"],
        activation="tanh",
        preference_dim=problem.objective_count,
        vae_latent_dim=config["latent_dim"],
    )
    return model.to(device)


def train_model(model, problem: ProblemSpec, config: dict, device: torch.device):
    return train_multiobjective(
        model=model,
        device=device,
        input_dim=problem.decision_dim,
        latent_dim=config["latent_dim"],
        batch_size=config["batch_size"],
        fitness_function=problem.objective_function,
        scale=True,
        optimization="Adam",
        log_interval=config["epochs"],
        **{
            key: value
            for key, value in config.items()
            if key not in {"hidden_dim", "latent_dim", "batch_size"}
        },
    )


def config_identifier(problem: ProblemSpec, config: dict, args) -> str:
    payload = {
        "version": EXPERIMENT_VERSION,
        "task": problem.name,
        "decision_dim": problem.decision_dim,
        "objective_count": problem.objective_count,
        "sample_count": args.sample_count,
        "reference_count": args.reference_count,
        "training": config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


RUN_FIELDS = (
    "experiment_version",
    "config_id",
    "timestamp_utc",
    "model",
    "task",
    "function",
    "decision_dim",
    "objective_count",
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "latent_dim",
    "archive_size",
    "anchor_count",
    "sample_count",
    "reference_count",
    "training_seconds",
    "training_config_json",
    *METRICS,
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


def write_summary(runs_path: Path, summary_path: Path) -> None:
    deduplicated = {}
    for row in read_rows(runs_path):
        deduplicated[(row["config_id"], row["task"], row["seed"])] = row

    groups: dict[tuple, list[dict]] = {}
    for row in deduplicated.values():
        key = (
            row["config_id"],
            row["task"],
            row["function"],
            row["decision_dim"],
            row["objective_count"],
        )
        groups.setdefault(key, []).append(row)

    fields = [
        "experiment_version",
        "config_id",
        "model",
        "task",
        "function",
        "decision_dim",
        "objective_count",
        "completed_runs",
    ]
    for metric in METRICS:
        fields.extend(
            (f"{metric}_median", f"{metric}_q1", f"{metric}_q3", f"{metric}_iqr")
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(groups):
            config_id, task, function_name, decision_dim, objective_count = key
            rows = groups[key]
            output = {
                "experiment_version": EXPERIMENT_VERSION,
                "config_id": config_id,
                "model": "PVAE",
                "task": task,
                "function": function_name,
                "decision_dim": decision_dim,
                "objective_count": objective_count,
                "completed_runs": len(rows),
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
                output[f"{metric}_median"] = median
                output[f"{metric}_q1"] = q1
                output[f"{metric}_q3"] = q3
                output[f"{metric}_iqr"] = q3 - q1
            writer.writerow(output)
    temporary.replace(summary_path)


def run_one(
    problem: ProblemSpec,
    true_pf: torch.Tensor,
    seed: int,
    device: torch.device,
    args,
) -> dict:
    seed_everything(seed)
    config = previous_result_preset(problem)
    config_id = config_identifier(problem, config, args)
    model = build_model(problem, config, device)
    started = time.perf_counter()
    model, loss_history = train_model(model, problem, config, device)
    training_seconds = time.perf_counter() - started
    result = evaluate_multi_objective(
        model=model,
        true_pf=true_pf,
        objective_function=problem.objective_function,
        device=device,
        sample_count=args.sample_count,
        scale=True,
    )
    row = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_id": config_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "PVAE",
        "task": problem.name,
        "function": problem.function_name,
        "decision_dim": problem.decision_dim,
        "objective_count": problem.objective_count,
        "seed": seed,
        "device": str(device),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "hidden_dim": config["hidden_dim"],
        "latent_dim": config["latent_dim"],
        "archive_size": config["archive_size"],
        "anchor_count": config["anchor_count"],
        "sample_count": args.sample_count,
        "reference_count": len(true_pf),
        "training_seconds": training_seconds,
        "training_config_json": json.dumps(config, sort_keys=True),
        **{metric: float(result[metric]) for metric in METRICS},
    }
    del loss_history, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--reference-count", type=int, default=10000)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(PROBLEM_BY_NAME),
        default=list(PROBLEM_BY_NAME),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "results" / "pvae_multi_objective",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.sample_count < 1 or args.reference_count < 2:
        raise ValueError("sample/reference counts must be positive")

    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    runs_path = output_dir / "runs.csv"
    summary_path = output_dir / "summary.csv"
    failures_path = output_dir / "failures.csv"
    if args.overwrite:
        for path in (runs_path, summary_path, failures_path):
            path.unlink(missing_ok=True)

    selected = [PROBLEM_BY_NAME[name] for name in args.tasks]
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    print(f"device={device}")
    print(f"runs={len(selected) * args.seeds}, seeds={args.seed_start}..{args.seed_start + args.seeds - 1}")
    print(f"output={output_dir}")
    true_fronts = {}
    for problem in selected:
        config = previous_result_preset(problem)
        true_fronts[problem.name] = make_true_front(problem, args.reference_count)
        print(
            f"  {problem.name}: decisions={problem.decision_dim}, "
            f"objectives={problem.objective_count}, true_pf={len(true_fronts[problem.name])}, "
            f"epochs={config['epochs']}, "
            f"archive/anchors={config['archive_size']}/{config['anchor_count']}, "
            f"binary={problem.binary_decisions}"
        )
    if args.dry_run:
        return

    existing = read_rows(runs_path)
    completed = {
        (row["config_id"], row["task"], int(row["seed"])) for row in existing
    }
    failure_fields = (
        "timestamp_utc",
        "task",
        "seed",
        "device",
        "error",
        "traceback",
    )
    total = len(selected) * args.seeds
    position = 0
    for problem in selected:
        config = previous_result_preset(problem)
        config_id = config_identifier(problem, config, args)
        for seed in seeds:
            position += 1
            key = (config_id, problem.name, seed)
            if key in completed:
                print(f"[{position}/{total}] skip completed: {problem.name}/seed={seed}")
                continue
            print(f"[{position}/{total}] run: {problem.name}/seed={seed}")
            try:
                row = run_one(problem, true_fronts[problem.name], seed, device, args)
                append_csv(runs_path, RUN_FIELDS, row)
                completed.add(key)
                write_summary(runs_path, summary_path)
                print(
                    f"  done: GD={row['GD']:.6g}, IGD={row['IGD']:.6g}, "
                    f"SP={row['SP']:.6g}, seconds={row['training_seconds']:.1f}"
                )
            except Exception as exc:
                append_csv(
                    failures_path,
                    failure_fields,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "task": problem.name,
                        "seed": seed,
                        "device": str(device),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"  FAILED: {exc!r}", file=sys.stderr)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if args.fail_fast:
                    raise

    write_summary(runs_path, summary_path)
    print(f"raw results: {runs_path}")
    print(f"median/IQR summary: {summary_path}")
    if failures_path.exists():
        print(f"failures: {failures_path}")


if __name__ == "__main__":
    main()

