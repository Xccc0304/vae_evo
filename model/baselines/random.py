"""Official pymoo random-search baseline."""

from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch

# Keep the concise project-level name requested by the experiment code.
Random = RandomSearch

__all__ = ["Random", "RandomSearch"]
