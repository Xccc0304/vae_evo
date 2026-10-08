import torch


def _validate_decisions(x, *, min_dimensions=1, exact_dimensions=None):
    """Validate a floating-point decision tensor whose last axis is features."""
    if not torch.is_tensor(x) or x.ndim < 1:
        raise ValueError("x must be a tensor with at least one dimension")
    dimensions = x.shape[-1]
    if exact_dimensions is not None and dimensions != exact_dimensions:
        raise ValueError(f"The last dimension must be {exact_dimensions}.")
    if dimensions < min_dimensions:
        raise ValueError(
            f"The last dimension must contain at least {min_dimensions} decision variables."
        )
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")
    return x



def rosenbrock(x, a=1, b=100):
    """
        Generalized n-dimensional Rosenbrock
        Parameters:
                   x (Tensor): Input tensor, the last dimension is the dimension D [..., x0, ..., x_{D-1}]
                   a (float): Default  1
                   b (float): Default  100
        Return:
        Tensor: Function value
        Global minimum f(x) = 0 at x = ones(D), for any dimension D.
    """
    x = _validate_decisions(x, min_dimensions=2)
    return ((a - x[..., :-1]).pow(2)
            + b * (x[..., 1:] - x[..., :-1].pow(2)).pow(2)).sum(dim=-1)


def beale(x):
    """
    Beale 
    Parameters:
               x (Tensor): Input tensor, the last dimension must be 2
    Return:
            Tensor: Function value 
    """
    x = _validate_decisions(x, exact_dimensions=2)
    x0, x1 = x[..., 0], x[..., 1]
    term1 = (1.5 - x0 + x0*x1)**2
    term2 = (2.25 - x0 + x0*x1**2)**2
    term3 = (2.625 - x0 + x0*x1**3)**2
    return term1 + term2 + term3

def himmelblau(x):
    """
     Himmelblau 
     Parameters:
               x (Tensor): Input tensor, the last dimension must be 2
     Return:
               Tensor: Function value
    """
    x = _validate_decisions(x, exact_dimensions=2)
    x0, x1 = x[..., 0], x[..., 1]
    return (x0**2 + x1 - 11)**2 + (x0 + x1**2 - 7)**2



def _project_to_bounds(x, bounds):
    """Project decision tensors onto a finite box along their last axis."""
    x = _validate_decisions(x)
    lower, upper = map(float, bounds)
    if not lower < upper:
        raise ValueError("bounds must satisfy lower < upper")
    return x.clamp(min=lower, max=upper)


def ackley(x, a=20.0, b=0.2, c=2 * torch.pi, bounds=(-4.0, 4.0)):
    """Evaluate the bounded Ackley function for a batch of decisions.

    Inputs outside ``bounds`` are projected to the nearest boundary before
    evaluation. Projection is safe for both minimization and maximization and
    cannot create an artificial optimum outside the feasible domain.
    """
    bounded = _project_to_bounds(x, bounds)
    # ``vector_norm`` uses the zero subgradient at the origin, unlike the
    # algebraically equivalent sqrt(mean(x**2)), whose backward pass is NaN.
    radius = torch.linalg.vector_norm(bounded, dim=-1) / bounded.shape[-1] ** 0.5
    mean_cosine = torch.cos(c * bounded).mean(dim=-1)
    return (
        -float(a) * torch.exp(-float(b) * radius)
        - torch.exp(mean_cosine)
        + float(a)
        + torch.e
    )


def rastrigin(x, A=10.0, bounds=(-4.0, 4.0)):
    """Evaluate the bounded Rastrigin function for a batch of decisions.

    Inputs outside ``bounds`` are projected to the nearest boundary before
    evaluation, avoiding the previous ``1e-6`` artificial optimum.
    """
    bounded = _project_to_bounds(x, bounds)
    dimensions = bounded.shape[-1]
    return float(A) * dimensions + (
        bounded.pow(2) - float(A) * torch.cos(2 * torch.pi * bounded)
    ).sum(dim=-1)


def griewank(x, bounds=(-4.0, 4.0)):
    """Evaluate the bounded Griewank function for a batch of decisions.

    Inputs outside ``bounds`` are projected to the nearest boundary before
    evaluation. Global minimum ``f(x) = 0`` at ``x = 0`` (inside ``bounds``).
    """
    bounded = _project_to_bounds(x, bounds)
    dimensions = bounded.shape[-1]
    indices = torch.arange(1, dimensions + 1, dtype=x.dtype, device=x.device)
    sum_term = (bounded.pow(2) / 4000.0).sum(dim=-1)
    prod_term = torch.cos(bounded / torch.sqrt(indices)).prod(dim=-1)
    return 1.0 + sum_term - prod_term


def weierstrass(x, a=0.5, b=3.0, k_max=20, bounds=(-4.0, 4.0)):
    """Evaluate the bounded Weierstrass function for a batch of decisions.

    This finite ``k_max`` approximation is differentiable but highly
    oscillatory. With integer ``b=3`` on ``[-4, 4]``, every integer decision
    vector is a global minimizer with ``f(x) = 0``; ``x = 0`` is one of them.
    """
    bounded = _project_to_bounds(x, bounds)
    dimensions = bounded.shape[-1]
    k = torch.arange(k_max + 1, dtype=x.dtype, device=x.device)
    a_k = a ** k
    # For odd integer b, cos(2*pi*b^k*(x + 0.5)) == -cos(2*pi*b^k*x), so the
    # standard definition reduces exactly to sum_k a^k * (d - sum_i cos(...)).
    # This avoids huge-phase cos() calls (e.g. 2*pi*3^20) that lose precision.
    oscillators = torch.cos(2 * torch.pi * (b ** k) * bounded[..., None])
    inner = oscillators.sum(dim=-2)
    return (a_k * (dimensions - inner)).sum(dim=-1)


def levy(x, bounds=(-4.0, 4.0)):
    """Evaluate the bounded Levy function for a batch of decisions.

    Multimodal with many local optima. Global minimum ``f(x) = 0`` at
    ``x = 1`` (in the original, unprojected coordinate frame, inside
    ``bounds``).
    """
    bounded = _project_to_bounds(x, bounds)
    w = 1.0 + (bounded - 1.0) / 4.0
    first = torch.sin(torch.pi * w[..., 0]).pow(2)
    interior = (
        (w[..., :-1] - 1.0).pow(2)
        * (1.0 + 10.0 * torch.sin(torch.pi * w[..., :-1] + 1.0).pow(2))
    ).sum(dim=-1)
    last = (w[..., -1] - 1.0).pow(2) * (1.0 + torch.sin(2 * torch.pi * w[..., -1]).pow(2))
    return first + interior + last
