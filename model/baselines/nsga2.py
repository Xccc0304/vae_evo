"""NSGA-II baseline model."""

from pymoo.algorithms.moo.nsga2 import NSGA2


def nsga2(pop_size=100, n_obj=2):
    """Create an NSGA-II instance.

    NSGA-II needs no reference directions, so it natively supports any
    number of objectives. ``n_obj`` is accepted only to keep the signature
    uniform across the MOO baselines.
    """
    return NSGA2(pop_size=pop_size)


__all__ = ["nsga2"]
