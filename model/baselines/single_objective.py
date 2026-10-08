"""Official single-objective optimization baselines.

Random Search, GA, DE, PSO and CMA-ES are provided by :mod:`pymoo`.
PGPE is provided by :mod:`evotorch`. This module only adapts their public
interfaces to tensor objectives and gives every algorithm the same result
format and function-evaluation budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


ArrayLike = Union[float, Sequence[float], np.ndarray]
Objective = Callable[[Any], Any]

PYMOO_ALGORITHMS = ("random", "ga", "de", "pso", "cma_es")
SINGLE_OBJECTIVE_ALGORITHMS = PYMOO_ALGORITHMS + ("pgpe",)


@dataclass(frozen=True)
class OptimizationResult:
    """Common output of every official single-objective baseline.

    ``population`` is the last population produced by the optimizer.
    ``archive_*`` contains every solution actually passed to the objective,
    which is the appropriate source for best/top-k comparisons.
    """

    algorithm: str
    sense: str
    best_solution: np.ndarray
    best_fitness: float
    population: np.ndarray
    fitness: np.ndarray
    archive_solutions: np.ndarray
    archive_fitness: np.ndarray
    function_evaluations: int
    requested_budget: int
    seed: int

    def top(self, k: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Return the best ``k`` evaluated solutions and raw fitness values."""
        if k <= 0:
            raise ValueError("k must be positive")
        count = min(int(k), self.function_evaluations)
        if self.sense == "minimize":
            indices = np.argsort(self.archive_fitness)[:count]
        else:
            indices = np.argsort(self.archive_fitness)[-count:][::-1]
        return self.archive_solutions[indices], self.archive_fitness[indices]

    def as_dict(self) -> Dict[str, Any]:
        """Return scalar metadata suitable for CSV/JSON experiment records."""
        return {
            "algorithm": self.algorithm,
            "sense": self.sense,
            "best_fitness": self.best_fitness,
            "function_evaluations": self.function_evaluations,
            "requested_budget": self.requested_budget,
            "seed": self.seed,
        }


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "random_search": "random",
        "randomsearch": "random",
        "cmaes": "cma_es",
        "cma": "cma_es",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SINGLE_OBJECTIVE_ALGORITHMS:
        choices = ", ".join(SINGLE_OBJECTIVE_ALGORITHMS)
        raise ValueError(f"unknown algorithm {name!r}; choose one of: {choices}")
    return normalized


def _normalize_sense(sense: str) -> str:
    normalized = sense.strip().lower()
    if normalized in {"min", "minimum", "minimize", "minimization"}:
        return "minimize"
    if normalized in {"max", "maximum", "maximize", "maximization"}:
        return "maximize"
    raise ValueError("sense must be 'minimize' or 'maximize'")


def _bounds_array(value: ArrayLike, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(dimension, float(array), dtype=np.float64)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must be a scalar or have shape ({dimension},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _normalize_bounds(
    bounds: Tuple[ArrayLike, ArrayLike], dimension: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair")
    lower = _bounds_array(bounds[0], dimension, "lower bound")
    upper = _bounds_array(bounds[1], dimension, "upper bound")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be smaller than its upper bound")
    return lower, upper


class _ObjectiveAdapter:
    """Evaluate NumPy or PyTorch objectives and retain the raw archive."""

    def __init__(
        self,
        objective: Objective,
        *,
        input_type: str,
        torch_device: Union[str, torch.device],
    ) -> None:
        if input_type not in {"auto", "torch", "numpy"}:
            raise ValueError("objective_input must be 'auto', 'torch', or 'numpy'")
        self.objective = objective
        self.input_type = input_type
        self.torch_device = torch.device(torch_device)
        self.solutions = []
        self.fitness = []

    @staticmethod
    def _values_array(values: Any, batch_size: int) -> np.ndarray:
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = array.reshape(1)
        array = array.reshape(-1)
        if array.size != batch_size:
            raise ValueError(
                "objective must return one scalar per solution; "
                f"received {array.size} values for a batch of {batch_size}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("objective returned NaN or infinity")
        return array

    def _call_torch(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.torch_device)
        with torch.no_grad():
            values = self.objective(tensor)
        return self._values_array(values, len(x))

    def _call_numpy(self, x: np.ndarray) -> np.ndarray:
        return self._values_array(self.objective(x), len(x))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]

        if self.input_type == "torch":
            values = self._call_torch(x)
        elif self.input_type == "numpy":
            values = self._call_numpy(x)
        else:
            try:
                values = self._call_torch(x)
                self.input_type = "torch"
            except (TypeError, ValueError, AttributeError):
                try:
                    values = self._call_numpy(x)
                    self.input_type = "numpy"
                except Exception as numpy_error:
                    raise RuntimeError(
                        "objective failed for both torch and numpy inputs; "
                        "set objective_input explicitly for a clearer error"
                    ) from numpy_error

        self.solutions.append(x.copy())
        self.fitness.append(values.copy())
        return values

    def archive(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.solutions:
            raise RuntimeError("the objective was never evaluated")
        return np.concatenate(self.solutions), np.concatenate(self.fitness)


class _PymooProblem(Problem):
    def __init__(
        self,
        dimension: int,
        lower: np.ndarray,
        upper: np.ndarray,
        evaluator: _ObjectiveAdapter,
        sense: str,
    ) -> None:
        super().__init__(n_var=dimension, n_obj=1, xl=lower, xu=upper)
        self.evaluator = evaluator
        self.sign = 1.0 if sense == "minimize" else -1.0

    def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray], *args, **kwargs) -> None:
        out["F"] = (self.sign * self.evaluator.evaluate(x))[:, None]


def make_pymoo_algorithm(
    name: str,
    *,
    dimension: int,
    lower: ArrayLike = -4.0,
    upper: ArrayLike = 4.0,
    population_size: Optional[int] = None,
    algorithm_kwargs: Optional[Dict[str, Any]] = None,
):
    """Create an unmodified official pymoo algorithm instance.

    PGPE is not accepted here because it is supplied by EvoTorch.
    """
    name = _normalize_name(name)
    if name == "pgpe":
        raise ValueError("PGPE is an EvoTorch algorithm, not a pymoo algorithm")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    lower_array, upper_array = _normalize_bounds((lower, upper), dimension)
    kwargs = dict(algorithm_kwargs or {})

    if name == "random":
        if population_size is not None:
            kwargs.setdefault("n_points_per_iteration", int(population_size))
        return RandomSearch(**kwargs)
    if name == "ga":
        if population_size is not None:
            kwargs.setdefault("pop_size", int(population_size))
        return GA(**kwargs)
    if name == "de":
        if population_size is not None:
            kwargs.setdefault("pop_size", int(population_size))
        return DE(**kwargs)
    if name == "pso":
        if population_size is not None:
            kwargs.setdefault("pop_size", int(population_size))
        return PSO(**kwargs)

    kwargs.setdefault("x0", (lower_array + upper_array) / 2.0)
    kwargs.setdefault("sigma", 0.3)
    if population_size is not None:
        kwargs.setdefault("pop_size", int(population_size))
    return CMAES(**kwargs)


def _build_result(
    *,
    name: str,
    sense: str,
    evaluator: _ObjectiveAdapter,
    population: np.ndarray,
    population_fitness: np.ndarray,
    budget: int,
    seed: int,
) -> OptimizationResult:
    archive_x, archive_f = evaluator.archive()
    best_index = int(np.argmin(archive_f) if sense == "minimize" else np.argmax(archive_f))
    return OptimizationResult(
        algorithm=name,
        sense=sense,
        best_solution=archive_x[best_index].copy(),
        best_fitness=float(archive_f[best_index]),
        population=np.asarray(population, dtype=np.float64).copy(),
        fitness=np.asarray(population_fitness, dtype=np.float64).reshape(-1).copy(),
        archive_solutions=archive_x,
        archive_fitness=archive_f,
        function_evaluations=len(archive_f),
        requested_budget=budget,
        seed=seed,
    )


def _run_pymoo(
    name: str,
    objective: Objective,
    *,
    dimension: int,
    bounds: Tuple[ArrayLike, ArrayLike],
    budget: int,
    seed: int,
    sense: str,
    population_size: Optional[int],
    objective_input: str,
    torch_device: Union[str, torch.device],
    algorithm_kwargs: Optional[Dict[str, Any]],
    verbose: bool,
) -> OptimizationResult:
    lower, upper = _normalize_bounds(bounds, dimension)
    evaluator = _ObjectiveAdapter(
        objective, input_type=objective_input, torch_device=torch_device
    )
    problem = _PymooProblem(dimension, lower, upper, evaluator, sense)
    algorithm = make_pymoo_algorithm(
        name,
        dimension=dimension,
        lower=lower,
        upper=upper,
        population_size=population_size,
        algorithm_kwargs=algorithm_kwargs,
    )
    result = minimize(
        problem,
        algorithm,
        termination=("n_eval", budget),
        seed=seed,
        verbose=verbose,
    )
    if result.pop is None:
        archive_x, archive_f = evaluator.archive()
        population = archive_x[-1:]
        population_fitness = archive_f[-1:]
    else:
        population = np.asarray(result.pop.get("X"), dtype=np.float64)
        minimized_fitness = np.asarray(result.pop.get("F"), dtype=np.float64).reshape(-1)
        population_fitness = minimized_fitness if sense == "minimize" else -minimized_fitness
    return _build_result(
        name=name,
        sense=sense,
        evaluator=evaluator,
        population=population,
        population_fitness=population_fitness,
        budget=budget,
        seed=seed,
    )


def _run_pgpe(
    objective: Objective,
    *,
    dimension: int,
    bounds: Tuple[ArrayLike, ArrayLike],
    budget: int,
    seed: int,
    sense: str,
    population_size: Optional[int],
    objective_input: str,
    torch_device: Union[str, torch.device],
    algorithm_kwargs: Optional[Dict[str, Any]],
) -> OptimizationResult:
    try:
        from evotorch import Problem as EvoTorchProblem
        from evotorch.algorithms import PGPE as EvoTorchPGPE
    except ImportError as error:
        raise ImportError(
            "PGPE requires EvoTorch; run `pip install -r requirements.txt`"
        ) from error

    lower, upper = _normalize_bounds(bounds, dimension)
    evaluator = _ObjectiveAdapter(
        objective, input_type=objective_input, torch_device=torch_device
    )
    midpoint = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0

    def bounded_objective(unbounded_solution: torch.Tensor) -> float:
        y = unbounded_solution.detach().cpu().numpy().astype(np.float64, copy=False)
        x = midpoint + half_range * np.tanh(y)
        return float(evaluator.evaluate(x)[0])

    popsize = 100 if population_size is None else int(population_size)
    if popsize <= 1 or popsize % 2:
        raise ValueError("PGPE population_size must be an even integer greater than 1")
    iterations = budget // popsize
    if iterations < 1:
        raise ValueError("budget must be at least population_size for PGPE")

    kwargs = dict(algorithm_kwargs or {})
    kwargs.setdefault("center_learning_rate", 0.1)
    kwargs.setdefault("stdev_learning_rate", 0.1)
    kwargs.setdefault("stdev_init", 0.5)
    kwargs.setdefault("center_init", np.zeros(dimension, dtype=np.float32))
    problem = EvoTorchProblem(
        "min" if sense == "minimize" else "max",
        bounded_objective,
        solution_length=dimension,
        initial_bounds=(-1.0, 1.0),
        dtype=torch.float32,
        device="cpu",
        seed=seed,
    )
    searcher = EvoTorchPGPE(problem, popsize=popsize, **kwargs)
    searcher.run(iterations)

    archive_x, archive_f = evaluator.archive()
    population = archive_x[-popsize:]
    population_fitness = archive_f[-popsize:]
    return _build_result(
        name="pgpe",
        sense=sense,
        evaluator=evaluator,
        population=population,
        population_fitness=population_fitness,
        budget=budget,
        seed=seed,
    )


def optimize_single_objective(
    algorithm: str,
    objective: Objective,
    *,
    dimension: int,
    bounds: Tuple[ArrayLike, ArrayLike] = (-4.0, 4.0),
    budget: int = 10_000,
    seed: int = 0,
    sense: str = "minimize",
    population_size: Optional[int] = None,
    objective_input: str = "auto",
    torch_device: Union[str, torch.device] = "cpu",
    algorithm_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Run an official baseline under a common function-evaluation budget.

    Pymoo stops after the first completed generation reaching ``budget`` and
    can exceed it by less than one generation. PGPE uses the largest whole
    population not exceeding ``budget``. Constructor-specific options are
    forwarded unchanged via ``algorithm_kwargs``.
    """
    name = _normalize_name(algorithm)
    sense = _normalize_sense(sense)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if population_size is not None and population_size <= 1:
        raise ValueError("population_size must be greater than 1")

    common = dict(
        dimension=int(dimension),
        bounds=bounds,
        budget=int(budget),
        seed=int(seed),
        sense=sense,
        population_size=population_size,
        objective_input=objective_input,
        torch_device=torch_device,
        algorithm_kwargs=algorithm_kwargs,
    )
    if name == "pgpe":
        return _run_pgpe(objective, **common)
    return _run_pymoo(name, objective, verbose=verbose, **common)


def random_search(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("random", objective, **kwargs)


def ga(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("ga", objective, **kwargs)


def de(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("de", objective, **kwargs)


def pso(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("pso", objective, **kwargs)


def cma_es(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("cma_es", objective, **kwargs)


def pgpe(objective: Objective, **kwargs) -> OptimizationResult:
    return optimize_single_objective("pgpe", objective, **kwargs)


__all__ = [
    "OptimizationResult",
    "PYMOO_ALGORITHMS",
    "SINGLE_OBJECTIVE_ALGORITHMS",
    "make_pymoo_algorithm",
    "optimize_single_objective",
    "random_search",
    "ga",
    "de",
    "pso",
    "cma_es",
    "pgpe",
]
