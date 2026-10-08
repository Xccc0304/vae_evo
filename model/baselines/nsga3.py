"""NSGA-III baseline model."""

from pymoo.algorithms.moo.nsga3 import NSGA3

from .utils import get_ref_dirs


def nsga3(pop_size=100, n_obj=2):
    """Create an NSGA-III instance.

    Reference directions must match the number of objectives ``n_obj``
    (2 objectives use das-dennis, 3+ use the energy method).
    """
    ref_dirs = get_ref_dirs(pop_size, n_obj)
    return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)


__all__ = ["nsga3"]
