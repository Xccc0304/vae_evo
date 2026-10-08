"""MOEA/D baseline model."""

from pymoo.algorithms.moo.moead import MOEAD

from .utils import get_ref_dirs


def moead(pop_size=100, n_obj=2):
    """Create a MOEA/D instance.

    Reference directions must match the number of objectives ``n_obj``
    (2 objectives use das-dennis, 3+ use the energy method).
    """
    ref_dirs = get_ref_dirs(pop_size, n_obj)
    return MOEAD(ref_dirs=ref_dirs)


__all__ = ["moead"]
