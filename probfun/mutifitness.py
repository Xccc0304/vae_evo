import torch
import math



def zdt1(x):
    x = x.clamp(1e-6, 1 - 1e-6)
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1:].mean(dim=1)
    ratio = (f1 / (g + 1e-6)).clamp(0.0, 1.0)
    f2 = g * (1.0 - torch.sqrt(ratio))
    return f1, f2

def zdt2(x):

    x = torch.clamp(x, 0.0, 1.0)
    f1 = x[:, 0]
    g = 1.0 + 9.0 * x[:, 1:].mean(dim=1)
    f1_over_g = f1 / (g + 1e-6)
    f2 = g * (1.0 - f1_over_g ** 2)

    return f1, f2


def zdt3(x):
    x = x.clamp(1e-8, 1.0)
    f1 = x[:, 0]
    g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
    h = 1 - torch.sqrt(f1 / (g + 1e-8)) - f1 / (g + 1e-8) * torch.sin(10 * torch.pi * f1)
    f2 = g * h
    return f1, f2




def zdt4(x):
    """Normalized-domain ZDT4.

    The public input stays in [0, 1]^n for compatibility with the generative
    training pipeline. Coordinates 1..n-1 are internally mapped to [-5, 5].
    """
    x = x.clamp(0.0, 1.0)
    x1 = x[:, 0]
    tail = -5.0 + 10.0 * x[:, 1:]
    g = 1.0 + 10.0 * tail.shape[1]
    g = g + (tail.pow(2) - 10.0 * torch.cos(4.0 * torch.pi * tail)).sum(dim=1)
    f2 = g * (1.0 - torch.sqrt((x1 / g.clamp_min(1e-12)).clamp_min(0.0)))
    return x1, f2


def zdt5(x, m=11, n=5, normalize=True):
    """Binary ZDT5 evaluated from relaxed [0, 1] decisions.

    Values are thresholded at 0.5 into the canonical 30-bit first block and
    ``m - 1`` blocks of ``n`` bits. Normalization matches pymoo's ZDT5.
    """
    expected = 30 + n * (m - 1)
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"ZDT5 expects {expected} decision variables")
    bits = (x >= 0.5).to(dtype=x.dtype)
    first_count = bits[:, :30].sum(dim=1)
    tail = bits[:, 30:].reshape(len(bits), m - 1, n)
    counts = tail.sum(dim=2)
    values = torch.where(counts < n, 2.0 + counts, torch.ones_like(counts))
    raw_f1 = 1.0 + first_count
    raw_f2 = values.sum(dim=1) / raw_f1
    if not normalize:
        return raw_f1, raw_f2
    f1 = (raw_f1 - 1.0) / 30.0
    lower_f2 = (m - 1) / 31.0
    upper_f2 = float(m - 1)
    f2 = (raw_f2 - lower_f2) / (upper_f2 - lower_f2)
    return f1, f2


def zdt6(x):
    """Canonical continuous ZDT6 on [0, 1]^n."""
    x = x.clamp(0.0, 1.0)
    x1 = x[:, 0]
    f1 = 1.0 - torch.exp(-4.0 * x1) * torch.sin(6.0 * torch.pi * x1).pow(6)
    mean_tail = x[:, 1:].mean(dim=1).clamp_min(0.0)
    g = 1.0 + 9.0 * mean_tail.pow(0.25)
    f2 = g * (1.0 - (f1 / g.clamp_min(1e-12)).pow(2))
    return f1, f2



def _validate_dtlz_input(x, num_objectives):
    """Validate the batched tensor interface shared by the DTLZ functions."""
    if not isinstance(num_objectives, int) or num_objectives < 2:
        raise ValueError("num_objectives must be an integer greater than or equal to 2")
    if x.ndim != 2:
        raise ValueError("x must have shape [batch_size, input_dim]")
    if x.shape[1] < num_objectives:
        raise ValueError(
            "DTLZ requires input_dim >= num_objectives; "
            f"got input_dim={x.shape[1]} and num_objectives={num_objectives}"
        )


def _dtlz_spherical_objectives(x, num_objectives, alpha):
    """Shared implementation of DTLZ2 (alpha=1) and DTLZ4."""
    _validate_dtlz_input(x, num_objectives)
    x = x.clamp(0.0, 1.0)
    position = x[:, :num_objectives - 1].pow(alpha)
    distance = x[:, num_objectives - 1:]
    g = (distance - 0.5).pow(2).sum(dim=1)
    radius = 1.0 + g
    half_pi = torch.pi / 2.0

    objectives = []
    for objective_index in range(num_objectives):
        num_cosines = num_objectives - objective_index - 1
        value = radius
        if num_cosines > 0:
            value = value * torch.cos(
                position[:, :num_cosines] * half_pi
            ).prod(dim=1)
        if objective_index > 0:
            value = value * torch.sin(
                position[:, num_cosines] * half_pi
            )
        # DTLZ2/4 objectives are non-negative analytically. At boundary
        # angles, float32 cos(pi/2) can be a tiny negative number and can
        # corrupt Pareto-dominance comparisons, so remove only that roundoff.
        objectives.append(value.clamp_min(0.0))
    return tuple(objectives)


def dtlz2(x, num_objectives=3):
    """Canonical DTLZ2 with a configurable number of objectives.

    x follows the same batched PyTorch interface as the ZDT functions.
    The returned tuple contains one [batch_size] tensor per objective.
    """
    return _dtlz_spherical_objectives(x, num_objectives, alpha=1.0)


def dtlz4(x, num_objectives=3, alpha=100.0):
    """Canonical DTLZ4, using the standard bias exponent alpha=100."""
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    return _dtlz_spherical_objectives(x, num_objectives, alpha=alpha)


def dtlz7(x, num_objectives=3):
    """Canonical DTLZ7 with a configurable number of objectives."""
    _validate_dtlz_input(x, num_objectives)
    x = x.clamp(0.0, 1.0)
    first_objectives = x[:, :num_objectives - 1]
    distance = x[:, num_objectives - 1:]
    g = 1.0 + 9.0 * distance.mean(dim=1)
    h = num_objectives - (
        (first_objectives / (1.0 + g).unsqueeze(1))
        * (1.0 + torch.sin(3.0 * torch.pi * first_objectives))
    ).sum(dim=1)
    final_objective = (1.0 + g) * h
    return tuple(first_objectives.unbind(dim=1)) + (final_objective,)











