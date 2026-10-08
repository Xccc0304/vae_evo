"""Official single- and multi-objective optimization baselines."""

from . import single_objective, utils
from .cma_es import CMAES, CMA_ES
from .de import DE
from .ga import GA
from .moead import moead
from .nsga2 import nsga2
from .nsga3 import nsga3
from .pso import PSO
from .random import Random, RandomSearch
from .rvea import rvea
from .single_objective import (
    OptimizationResult,
    PYMOO_ALGORITHMS,
    SINGLE_OBJECTIVE_ALGORITHMS,
    make_pymoo_algorithm,
    optimize_single_objective,
)


def __getattr__(name):
    """Load EvoTorch only when the official PGPE class is requested."""
    if name == "PGPE":
        from .pgpe import PGPE

        globals()[name] = PGPE
        return PGPE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Official single-objective classes.
    "Random",
    "RandomSearch",
    "GA",
    "DE",
    "PSO",
    "CMAES",
    "CMA_ES",
    "PGPE",
    # Common runner and result type.
    "single_objective",
    "OptimizationResult",
    "PYMOO_ALGORITHMS",
    "SINGLE_OBJECTIVE_ALGORITHMS",
    "make_pymoo_algorithm",
    "optimize_single_objective",
    # Existing multi-objective factories.
    "moead",
    "nsga2",
    "nsga3",
    "rvea",
    "utils",
]
