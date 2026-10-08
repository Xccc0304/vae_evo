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
from pathlib import Path
from typing import Callable

import numpy as np
import torch


# This script lives in `experiment/VAE`; two parent levels reach the
# repository root that contains `model`, `probfun` and `scripts`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.VAEs import IVAE, MHVQVAE, VQVAE, adaptive_vae_capacity
from probfun.fitness import ackley, beale, griewank, himmelblau, levy, rastrigin, rosenbrock, weierstrass
from scripts.eval import evaluate, evaluate_muti_peak, set_seed
from scripts.train import train_IVAE, train_MHVQVAE, train_VQVAE


EXPERIMENT_VERSION = "single_objective_decoder_only_v1"
MODEL_NAMES = ("vae", "vqvae", "mhvqvae")
METRIC_NAMES = (
    "entropy_all",
    "entropy_top_k",
    "fitness_reward_all",
    "fitness_reward_top_k",
    "average_fitness_all",
    "average_fitness_top_k",
)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    function_name: str
    function: Callable[[torch.Tensor], torch.Tensor]
    dimension: int
    objective: str
    eval_name: str
    mhvqvae_heads: int

    @property
    def maximize(self) -> bool:
        return self.objective == "maximize"


PROBLEMS = (
    # --- 最大化: 2 / 4 / 32 / 256 维, target 为 (-4,4)^D 上的最大值 ---
    ProblemSpec("ackley_2d", "ackley", ackley, 2, "maximize", "ackley", 4),
    ProblemSpec("ackley_4d", "ackley", ackley, 4, "maximize", "ackley", 4),
    ProblemSpec("ackley_32d", "ackley", ackley, 32, "maximize", "ackley", 4),
    ProblemSpec("ackley_256d", "ackley", ackley, 256, "maximize", "ackley", 4),
    ProblemSpec("rastrigin_2d", "rastrigin", rastrigin, 2, "maximize", "rastrigin", 4),
    ProblemSpec("rastrigin_4d", "rastrigin", rastrigin, 4, "maximize", "rastrigin_4d", 4),
    ProblemSpec("rastrigin_32d", "rastrigin", rastrigin, 32, "maximize", "rastrigin_32d", 4),
    ProblemSpec("rastrigin_256d", "rastrigin", rastrigin, 256, "maximize", "rastrigin_256d", 4),
    # --- 最小化: 保留的 2D 基准 ---
    ProblemSpec("rosenbrock_2d", "rosenbrock", rosenbrock, 2, "minimize", "rosenbrock", 1),
    ProblemSpec("beale_2d", "beale", beale, 2, "minimize", "beale", 1),
    ProblemSpec("himmelblau_2d", "himmelblau", himmelblau, 2, "minimize", "himmelblau", 4),
    # --- 最小化: 高维 10 / 30 / 100 维 ---
    ProblemSpec("griewank_10d", "griewank", griewank, 10, "minimize", "griewank_10d", 1),
    ProblemSpec("griewank_30d", "griewank", griewank, 30, "minimize", "griewank_30d", 1),
    ProblemSpec("griewank_100d", "griewank", griewank, 100, "minimize", "griewank_100d", 1),
    ProblemSpec("weierstrass_10d", "weierstrass", weierstrass, 10, "minimize", "weierstrass_10d", 1),
    ProblemSpec("weierstrass_30d", "weierstrass", weierstrass, 30, "minimize", "weierstrass_30d", 1),
    ProblemSpec("weierstrass_100d", "weierstrass", weierstrass, 100, "minimize", "weierstrass_100d", 1),
    ProblemSpec("levy_10d", "levy", levy, 10, "minimize", "levy_10d", 1),
    ProblemSpec("levy_30d", "levy", levy, 30, "minimize", "levy_30d", 1),
    ProblemSpec("levy_100d", "levy", levy, 100, "minimize", "levy_100d", 1),
    ProblemSpec("rosenbrock_10d", "rosenbrock", rosenbrock, 10, "minimize", "rosenbrock_10d", 1),
    ProblemSpec("rosenbrock_30d", "rosenbrock", rosenbrock, 30, "minimize", "rosenbrock_30d", 1),
    ProblemSpec("rosenbrock_100d", "rosenbrock", rosenbrock, 100, "minimize", "rosenbrock_100d", 1),
)
PROBLEM_BY_NAME = {problem.name: problem for problem in PROBLEMS}


# Validated with full-length seed-0 runs; only non-default values are listed.
TANH_ALPHA_OVERRIDES = {
    # Rosenbrock keeps model-specific mapping scales within a 2x range.
    ("vae", "rosenbrock", 10): 4.0,
    ("vae", "rosenbrock", 30): 4.0,
    ("vae", "rosenbrock", 100): 1.0,
    ("vqvae", "rosenbrock", 30): 2.0,
    ("vqvae", "rosenbrock", 100): 0.25,
    ("mhvqvae", "rosenbrock", 30): 2.0,
    ("mhvqvae", "rosenbrock", 100): 0.25,
    # Griewank benefits consistently from alpha=3 across models and dimensions.
    ("vae", "griewank", 10): 3.0,
    ("vqvae", "griewank", 10): 3.0,
    ("mhvqvae", "griewank", 10): 3.0,
    ("vae", "griewank", 30): 3.0,
    ("vqvae", "griewank", 30): 3.0,
    ("mhvqvae", "griewank", 30): 3.0,
    ("vae", "griewank", 100): 3.0,
    ("vqvae", "griewank", 100): 3.0,
    ("mhvqvae", "griewank", 100): 3.0,
    # Levy uses nearby mapping scales; the largest same-dimension ratio is 2x.
    ("vae", "levy", 10): 1.5,
    ("vqvae", "levy", 10): 1.5,
    ("mhvqvae", "levy", 10): 1.5,
    ("vae", "levy", 30): 0.75,
    ("vqvae", "levy", 30): 0.75,
    ("mhvqvae", "levy", 30): 1.5,
    ("vae", "levy", 100): 3.0,
    ("vqvae", "levy", 100): 4.0,
    ("mhvqvae", "levy", 100): 2.0,
    # Weierstrass values are intentionally unchanged.
    ("vae", "weierstrass", 10): 0.0125,
    ("vqvae", "weierstrass", 10): 0.05,
    ("mhvqvae", "weierstrass", 10): 0.0125,
    ("vae", "weierstrass", 30): 0.0125,
    ("vqvae", "weierstrass", 30): 0.0125,
    ("mhvqvae", "weierstrass", 30): 0.0125,
    ("vae", "weierstrass", 100): 0.075,
    ("vqvae", "weierstrass", 100): 0.025,
    ("mhvqvae", "weierstrass", 100): 0.10,
    # Ackley-256 MHVQVAE remains unchanged.
    ("mhvqvae", "ackley", 256): 0.25,
}


VAE_TRAINING_OVERRIDES = {
    # Full-length seed-0 validation for the 100-D curved Rosenbrock valley.
    ("rosenbrock", 100): {
        "hidden_dim": 96,
        "latent_dim": 32,
        "epochs": 1000,
        "blackbox_max_directions": 100,
    },
}

# Full-direction settings retained only where seed-0 validation improved the
# decoder-only evaluation. Keys are (model, function, dimension).
MODEL_TRAINING_OVERRIDES = {
    ("vae", "levy", 10): {"epochs": 1000, "blackbox_max_directions": 10},
    ("vqvae", "levy", 10): {"epochs": 1000, "blackbox_max_directions": 10},
    ("mhvqvae", "levy", 10): {"epochs": 1000, "blackbox_max_directions": 10},
    ("vae", "levy", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("vqvae", "levy", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("mhvqvae", "levy", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("vae", "levy", 100): {
        "epochs": 1000, "blackbox_max_directions": 100,
        "hidden_dim": 96, "latent_dim": 32,
    },
    ("vqvae", "levy", 100): {
        "epochs": 1000, "blackbox_max_directions": 100,
        "blackbox_weight": 1.0,
    },
    ("mhvqvae", "levy", 100): {
        "epochs": 1000, "blackbox_max_directions": 100,
        "blackbox_weight": 1.0,
    },
    ("vqvae", "weierstrass", 10): {
        "epochs": 1000, "blackbox_max_directions": 10,
    },
    ("vae", "weierstrass", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("vqvae", "weierstrass", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("mhvqvae", "weierstrass", 30): {"epochs": 1000, "blackbox_max_directions": 30},
    ("vae", "weierstrass", 100): {
        "epochs": 1000, "blackbox_max_directions": 100,
        "hidden_dim": 96, "latent_dim": 32,
        "blackbox_weight": 1.0, "blackbox_start_fraction": 0.0,
    },
    ("vqvae", "weierstrass", 100): {
        "epochs": 1000, "blackbox_max_directions": 100,
    },
}
ROSENBROCK_BLACKBOX_OVERRIDES = {
    10: {
        "blackbox_weight": 0.5,
        "blackbox_start_fraction": 0.1,
        "blackbox_max_directions": 10,
    },
    30: {
        "blackbox_weight": 1.0,
        "blackbox_start_fraction": 0.0,
        "blackbox_max_directions": 30,
    },
    100: {
        "blackbox_weight": 1.0,
        "blackbox_start_fraction": 0.0,
        "blackbox_max_directions": 100,
    },
}


def _apply_alpha_override(config, model_name, function_name, dimension):
    alpha_override = TANH_ALPHA_OVERRIDES.get((model_name, function_name, dimension))
    if alpha_override is not None:
        config["tanh_alpha"] = alpha_override
    if function_name == "rosenbrock":
        blackbox_override = ROSENBROCK_BLACKBOX_OVERRIDES.get(dimension)
        if blackbox_override is not None:
            config.update(blackbox_override)
    # Apply validated model-specific settings after the shared defaults.
    if model_name == "vae":
        training_override = VAE_TRAINING_OVERRIDES.get((function_name, dimension))
        if training_override is not None:
            config.update(training_override)
    model_override = MODEL_TRAINING_OVERRIDES.get(
        (model_name, function_name, dimension)
    )
    if model_override is not None:
        config.update(model_override)
    # Respect each validated task setting; only cap impossible values at d.
    config["blackbox_max_directions"] = min(
        int(dimension), int(config["blackbox_max_directions"])
    )
    return config


def dimension_preset(
    model_name: str,
    dimension: int,
    objective=None,
    function_name=None,
) -> dict:
    """Return dimension- and direction-aware training parameters."""
    common = {
        "global_radius": 4.0,
        "elite_fraction": 0.2,
        "elite_fraction_end": 0.05,
        "rank_temperature": 0.2,
        "rank_temperature_end": 0.05,
        "exploration_fraction": 0.2,
        "mutation_sigma_start": 0.5,
        "mutation_sigma_end": 0.02,
        "train_steps": 3,
        "blackbox_weight": 0.05,
        "blackbox_step": 1e-3,
    }
    if model_name == "vae":
        common.update(adaptive_vae_capacity(dimension))



    if dimension >= 256:
        common.update(
            tanh_alpha=0.10,
            blackbox_max_directions=256,
            blackbox_start_fraction=0.20,
        )
        if model_name == "vae":
            common.update(
                epochs=500,
                batch_size=2048,
                learning_rate=1e-3,
                elite_fraction=0.10,
                vae_weight=0.25,
            )
        elif model_name == "vqvae":
            common.update(
                epochs=1000,
                batch_size=512,
                learning_rate=3e-4,
                vae_weight=0.1,
                archive_size=512,
                routing_codes=32,
                code_calibration_steps=1500,
                routing_calibration_steps=0,
            )
        else:
            common.update(
                tanh_alpha=0.15,
                epochs=1000,
                batch_size=512,
                learning_rate=3e-4,
                vae_weight=0.1,
                archive_size=256,
                routing_codes=32,
                code_calibration_steps=1500,
                routing_calibration_steps=0,
            )
        if model_name == "vae" and objective == "minimize":
            common["tanh_alpha"] = 2.0
        return _apply_alpha_override(common, model_name, function_name, dimension)

    if dimension >= 32:
        common.update(
            tanh_alpha=0.15,
            blackbox_max_directions=32,
            blackbox_start_fraction=0.40,
        )
        if model_name == "vae":
            common.update(
                epochs=500,
                batch_size=1024,
                learning_rate=1e-3,
                vae_weight=0.25,
            )
        else:
            common.update(
                epochs=1000,
                batch_size=1024,
                learning_rate=3e-4,
                vae_weight=0.1,
                archive_size=512,
                routing_codes=8,
                code_calibration_steps=1500,
                routing_calibration_steps=0,
            )
        if model_name == "vae" and objective == "minimize":
            common["tanh_alpha"] = 2.0
        return _apply_alpha_override(common, model_name, function_name, dimension)

    common.update(
        epochs=500,
        batch_size=1024,
        learning_rate=1e-3 if model_name == "vae" else 3e-4,
        vae_weight=0.25 if model_name == "vae" else 0.1,
        tanh_alpha=0.10 if model_name == "vae" else 2.0,
        blackbox_max_directions=16,
        blackbox_start_fraction=0.80,
    )
    if model_name != "vae":
        common.update(
            archive_size=512,
            routing_codes=6,
            code_calibration_steps=500,
            routing_calibration_steps=0,
        )
    if model_name == "vae" and objective == "minimize":
        common["tanh_alpha"] = 2.0
    return _apply_alpha_override(common, model_name, function_name, dimension)


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


def build_model(model_name: str, problem: ProblemSpec, config: dict, device: torch.device):
    model_kwargs = {
        "input_dim": problem.dimension,
        "use_tanh": True,
        "tanh_scale": 4.0,
        "tanh_alpha": config["tanh_alpha"],
    }
    if model_name == "vae":
        model = IVAE(
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            **model_kwargs,
        )
    elif model_name == "vqvae":
        model = VQVAE(**model_kwargs)
    elif model_name == "mhvqvae":
        model = MHVQVAE(n_heads=problem.mhvqvae_heads, **model_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def common_train_kwargs(model, problem: ProblemSpec, config: dict, device: torch.device) -> dict:
    kwargs = {
        "model": model,
        "device": device,
        "input_dim": problem.dimension,
        "batch_size": config["batch_size"],
        "fitness_function": problem.function,
        "objective": problem.objective,
        "global_radius": config["global_radius"],
        "epochs": config["epochs"],
        "log_interval": config["epochs"],
        "learning_rate": config["learning_rate"],
        "elite_fraction": config["elite_fraction"],
        "elite_fraction_end": config["elite_fraction_end"],
        "rank_temperature": config["rank_temperature"],
        "rank_temperature_end": config["rank_temperature_end"],
        "exploration_fraction": config["exploration_fraction"],
        "mutation_sigma_start": config["mutation_sigma_start"],
        "mutation_sigma_end": config["mutation_sigma_end"],
        "train_steps": config["train_steps"],
        "blackbox_weight": config["blackbox_weight"],
        "blackbox_step": config["blackbox_step"],
        "blackbox_max_directions": config["blackbox_max_directions"],
        "blackbox_start_fraction": config["blackbox_start_fraction"],
    }
    if "elite_guidance" in config:
        kwargs["elite_guidance"] = bool(config["elite_guidance"])
    return kwargs


def train_model(model_name: str, model, problem: ProblemSpec, config: dict, device: torch.device):
    kwargs = common_train_kwargs(model, problem, config, device)
    kwargs["vae_weight"] = config["vae_weight"]

    if model_name == "vae":
        kwargs["archive_size"] = config.get("archive_size", 512)
        return train_IVAE(**kwargs)

    kwargs.update(
        archive_size=config["archive_size"],
        routing_codes=config["routing_codes"],
        code_calibration_steps=config["code_calibration_steps"],
        routing_calibration_steps=config["routing_calibration_steps"],
    )
    kwargs["single_peak"] = problem.mhvqvae_heads == 1
    if model_name == "vqvae":
        return train_VQVAE(**kwargs)

    kwargs["head_min_distance"] = 0.0 if problem.mhvqvae_heads == 1 else 2.0
    return train_MHVQVAE(**kwargs)


def scalar(value) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return float(value)


def flatten_evaluation(result: dict) -> dict[str, float]:
    return {
        "entropy_all": scalar(result["entropy"]["all"]),
        "entropy_top_k": scalar(result["entropy"]["top_k"]),
        "fitness_reward_all": scalar(result["fitness_reward"]["all"]),
        "fitness_reward_top_k": scalar(result["fitness_reward"]["top_k"]),
        "average_fitness_all": scalar(result["average_fitness"]["all"]),
        "average_fitness_top_k": scalar(result["average_fitness"]["top_k"]),
    }


def evaluate_model(
    model_name: str,
    model,
    problem: ProblemSpec,
    device: torch.device,
    sample_count: int,
    top_k: int,
) -> dict[str, float]:
    if model_name == "mhvqvae":
        result = evaluate_muti_peak(
            model=model,
            input_dim=problem.dimension,
            obj=problem.function,
            obj_name=problem.eval_name,
            k=top_k,
            maximize=problem.maximize,
            device=device,
            sample_count=sample_count,
        )
    else:
        result = evaluate(
            model=model,
            device=device,
            input_dim=problem.dimension,
            obj=problem.function,
            obj_name=problem.eval_name,
            k=top_k,
            maximize=problem.maximize,
            sample_count=sample_count,
        )
    return flatten_evaluation(result)


def config_identifier(model_name: str, problem: ProblemSpec, config: dict, args) -> str:
    payload = {
        "version": EXPERIMENT_VERSION,
        "model": model_name,
        "problem": problem.name,
        "heads": problem.mhvqvae_heads if model_name == "mhvqvae" else 1,
        "sample_count": args.sample_count,
        "top_k": args.top_k,
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
    "dimension",
    "objective",
    "n_heads",
    "seed",
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "hidden_dim",
    "latent_dim",
    "num_embeddings",
    "embedding_dim",
    "routing_codes",
    "sample_count",
    "top_k",
    "training_seconds",
    "training_config_json",
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


def write_summary(runs_path: Path, summary_path: Path) -> None:
    # Keep only the latest copy if a user manually appended a duplicate run.
    deduplicated = {}
    for row in read_rows(runs_path):
        key = (row["config_id"], row["model"], row["task"], row["seed"])
        deduplicated[key] = row

    groups: dict[tuple, list[dict]] = {}
    for row in deduplicated.values():
        key = (
            row["config_id"],
            row["model"],
            row["task"],
            row["function"],
            row["dimension"],
            row["objective"],
            row["n_heads"],
        )
        groups.setdefault(key, []).append(row)

    summary_fields = [
        "experiment_version",
        "config_id",
        "model",
        "task",
        "function",
        "dimension",
        "objective",
        "n_heads",
        "completed_runs",
    ]
    for metric in METRIC_NAMES:
        summary_fields.extend(
            (f"{metric}_median", f"{metric}_q1", f"{metric}_q3", f"{metric}_iqr")
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key in sorted(groups):
            config_id, model_name, task, function_name, dimension, objective, n_heads = key
            rows = groups[key]
            output = {
                "experiment_version": EXPERIMENT_VERSION,
                "config_id": config_id,
                "model": model_name,
                "task": task,
                "function": function_name,
                "dimension": dimension,
                "objective": objective,
                "n_heads": n_heads,
                "completed_runs": len(rows),
            }
            for metric in METRIC_NAMES:
                values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
                output[f"{metric}_median"] = median
                output[f"{metric}_q1"] = q1
                output[f"{metric}_q3"] = q3
                output[f"{metric}_iqr"] = q3 - q1
            writer.writerow(output)
    temporary.replace(summary_path)


def run_one(model_name: str, problem: ProblemSpec, seed: int, device, args) -> dict:
    seed_everything(seed)
    config = dimension_preset(
        model_name, problem.dimension, problem.objective, problem.function_name
    )
    config_id = config_identifier(model_name, problem, config, args)
    model = build_model(model_name, problem, config, device)
    started = time.perf_counter()
    model, loss_history = train_model(model_name, model, problem, config, device)
    training_seconds = time.perf_counter() - started
    metrics = evaluate_model(
        model_name, model, problem, device, args.sample_count, args.top_k
    )

    row = {
        "experiment_version": EXPERIMENT_VERSION,
        "config_id": config_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "task": problem.name,
        "function": problem.function_name,
        "dimension": problem.dimension,
        "objective": problem.objective,
        "n_heads": problem.mhvqvae_heads if model_name == "mhvqvae" else 1,
        "seed": seed,
        "device": str(device),
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "hidden_dim": getattr(model, "hidden_dim", ""),
        "latent_dim": getattr(model, "latent_dim", ""),
        "num_embeddings": getattr(model, "num_embeddings", ""),
        "embedding_dim": getattr(model, "embedding_dim", ""),
        "routing_codes": config.get("routing_codes", ""),
        "sample_count": args.sample_count,
        "top_k": args.top_k,
        "training_seconds": training_seconds,
        "training_config_json": json.dumps(config, sort_keys=True),
        **metrics,
    }

    del loss_history, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", type=int, default=30, help="number of independent seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed; defaults to 0")
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(PROBLEM_BY_NAME),
        default=list(PROBLEM_BY_NAME),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "results" / "single_objective",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing CSV outputs")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failed run")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.sample_count < 1 or args.top_k < 1:
        raise ValueError("--sample-count and --top-k must be positive")

    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    runs_path = output_dir / "runs.csv"
    summary_path = output_dir / "summary.csv"
    failures_path = output_dir / "failures.csv"

    if args.overwrite:
        for path in (runs_path, summary_path, failures_path):
            path.unlink(missing_ok=True)

    selected_problems = [PROBLEM_BY_NAME[name] for name in args.tasks]
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    total = len(args.models) * len(selected_problems) * args.seeds

    print(f"device={device}")
    print(f"runs={total}, models={args.models}, seeds={args.seed_start}..{args.seed_start + args.seeds - 1}")
    print(f"output={output_dir}")
    for problem in selected_problems:
        print(
            f"  {problem.name}: dim={problem.dimension}, objective={problem.objective}, "
            f"mhvqvae_heads={problem.mhvqvae_heads}"
        )
    if args.dry_run:
        return

    existing_rows = read_rows(runs_path)
    completed = {
        (row["config_id"], row["model"], row["task"], int(row["seed"]))
        for row in existing_rows
    }
    failure_fields = (
        "timestamp_utc",
        "model",
        "task",
        "seed",
        "device",
        "error",
        "traceback",
    )

    position = 0
    for model_name in args.models:
        for problem in selected_problems:
            config = dimension_preset(
        model_name, problem.dimension, problem.objective, problem.function_name
    )
            config_id = config_identifier(model_name, problem, config, args)
            for seed in seeds:
                position += 1
                key = (config_id, model_name, problem.name, seed)
                if key in completed:
                    print(f"[{position}/{total}] skip completed: {model_name}/{problem.name}/seed={seed}")
                    continue

                print(f"[{position}/{total}] run: {model_name}/{problem.name}/seed={seed}")
                try:
                    row = run_one(model_name, problem, seed, device, args)
                    append_csv(runs_path, RUN_FIELDS, row)
                    completed.add(key)
                    write_summary(runs_path, summary_path)
                    print(
                        "  done: "
                        f"average={row['average_fitness_all']:.6g}, "
                        f"top_k={row['average_fitness_top_k']:.6g}, "
                        f"seconds={row['training_seconds']:.1f}"
                    )
                except Exception as exc:
                    append_csv(
                        failures_path,
                        failure_fields,
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model": model_name,
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






