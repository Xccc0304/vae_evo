"""RVEA baseline model."""

from pymoo.algorithms.moo.rvea import RVEA

from .utils import get_ref_dirs


def rvea(pop_size=100, n_obj=2):
    """Create an RVEA instance.

    Reference directions must match the number of objectives ``n_obj``
    (2 objectives use das-dennis, 3+ use the energy method).
    """
    ref_dirs = get_ref_dirs(pop_size, n_obj)
    return RVEA(ref_dirs=ref_dirs)


__all__ = ["rvea"]
