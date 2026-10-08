"""Official pymoo CMA-ES baseline."""

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

# ``CMA_ES`` is provided as an optional readable alias; ``CMAES`` is the
# official pymoo class name and the preferred import.
CMA_ES = CMAES

__all__ = ["CMAES", "CMA_ES"]
