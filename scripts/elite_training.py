"""Shared utilities for objective-gradient-free elite training.

The objective functions are evaluated on detached candidates under
``torch.no_grad()``.  Model gradients are computed only from likelihood/
reconstruction terms on the evaluated elite set.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _as_vector(values, reference: torch.Tensor, name: str) -> torch.Tensor:
    values = torch.as_tensor(values, device=reference.device, dtype=reference.dtype)
    values = values.reshape(-1)
    if values.numel() != reference.shape[0]:
        raise ValueError(
            f"{name} must return one value per candidate; got {values.numel()} "
            f"values for {reference.shape[0]} candidates."
        )
    return values.detach()


@torch.no_grad()
def evaluate_objective(objective, candidates: torch.Tensor) -> torch.Tensor:
    """Evaluate a scalar black-box objective without building an autograd graph."""
    detached = candidates.detach()
    return _as_vector(objective(detached), detached, "objective")


def _elite_count(population_size: int, elite_fraction: float) -> int:
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    return max(1, min(population_size, math.ceil(population_size * elite_fraction)))


def rank_weights(count: int, temperature: float, device, dtype) -> torch.Tensor:
    """Return normalized weights for samples already sorted best-to-worst."""
    if temperature <= 0:
        raise ValueError("rank_temperature must be positive")
    if count == 1:
        return torch.ones(1, device=device, dtype=dtype)
    normalized_rank = torch.arange(count, device=device, dtype=dtype) / (count - 1)
    return torch.softmax(-normalized_rank / temperature, dim=0).detach()


def select_ranked_elites(
    candidates: torch.Tensor,
    values: torch.Tensor,
    elite_fraction: float,
    rank_temperature: float,
    maximize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select scalar-objective elites and return detached rank weights."""
    count = _elite_count(len(candidates), elite_fraction)
    order = torch.argsort(values, descending=maximize)
    indices = order[:count]
    weights = rank_weights(count, rank_temperature, candidates.device, candidates.dtype)
    return candidates[indices].detach(), weights, values[indices].detach(), indices


def _per_sample_mean(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value
    return value.reshape(value.shape[0], -1).mean(dim=1)


def weighted_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Gaussian decoder negative log-likelihood up to an additive constant."""
    per_sample = _per_sample_mean((reconstruction - target.detach()).pow(2))
    normalized = weights.detach() / weights.detach().sum().clamp_min(1e-12)
    return torch.sum(normalized * per_sample)


def weighted_elbo_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    weights: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative rank-weighted ELBO for an evaluated elite population."""
    reconstruction_loss = weighted_reconstruction_loss(reconstruction, target, weights)
    kl_per_sample = -0.5 * (
        1.0 + logvar - mu.pow(2) - logvar.exp()
    ).reshape(logvar.shape[0], -1).sum(dim=1)
    normalized = weights.detach() / weights.detach().sum().clamp_min(1e-12)
    kl_loss = torch.sum(normalized * kl_per_sample)
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def pareto_front_ranks(objectives: torch.Tensor) -> torch.Tensor:
    """Compute nondominated front numbers for minimization objectives."""
    if objectives.ndim != 2:
        raise ValueError("objectives must have shape [population, objectives]")
    # The dense formulation is fastest for ordinary training populations, but
    # it retains several [N, N] boolean tensors. Large Pareto archives need the
    # equivalent blocked formulation below to keep peak memory bounded.
    if len(objectives) > 2048:
        count = len(objectives)
        block_size = 256
        domination_count = torch.zeros(
            count, device=objectives.device, dtype=torch.long
        )
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            query = objectives[start:stop]
            dominators = (
                (objectives.unsqueeze(0) <= query.unsqueeze(1)).all(dim=2)
                & (objectives.unsqueeze(0) < query.unsqueeze(1)).any(dim=2)
            )
            domination_count[start:stop] = dominators.sum(dim=1)

        ranks = torch.full(
            (count,), -1, device=objectives.device, dtype=torch.long
        )
        remaining = domination_count
        front = remaining == 0
        rank = 0
        while bool(front.any()):
            ranks[front] = rank
            front_objectives = objectives[front]
            next_remaining = remaining.clone()
            unresolved = torch.nonzero(ranks < 0, as_tuple=False).flatten()
            for start in range(0, len(unresolved), block_size):
                indices = unresolved[start:start + block_size]
                query = objectives[indices]
                dominated_by_front = (
                    (front_objectives.unsqueeze(0) <= query.unsqueeze(1)).all(dim=2)
                    & (front_objectives.unsqueeze(0) < query.unsqueeze(1)).any(dim=2)
                ).sum(dim=1)
                next_remaining[indices] -= dominated_by_front
            next_remaining[ranks >= 0] = -1
            remaining = next_remaining
            front = remaining == 0
            rank += 1
        return ranks

    no_worse = (objectives[:, None, :] <= objectives[None, :, :]).all(dim=2)
    strictly_better = (objectives[:, None, :] < objectives[None, :, :]).any(dim=2)
    dominates = no_worse & strictly_better
    domination_count = dominates.transpose(0, 1).sum(dim=1).to(torch.long)

    ranks = torch.full(
        (len(objectives),), -1, device=objectives.device, dtype=torch.long
    )
    remaining = domination_count.clone()
    front = remaining == 0
    rank = 0
    while front.any():
        ranks[front] = rank
        dominated_by_front = dominates[front].sum(dim=0).to(torch.long)
        remaining = remaining - dominated_by_front
        remaining[ranks >= 0] = -1
        front = remaining == 0
        rank += 1
    return ranks


def crowding_distance(objectives: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    """Compute NSGA-II style crowding distances within each Pareto front."""
    distances = torch.zeros(len(objectives), device=objectives.device, dtype=objectives.dtype)
    for rank in torch.unique(ranks):
        indices = torch.nonzero(ranks == rank, as_tuple=False).flatten()
        if indices.numel() <= 2:
            distances[indices] = torch.inf
            continue
        front = objectives[indices]
        for objective_index in range(objectives.shape[1]):
            order = torch.argsort(front[:, objective_index])
            ordered_indices = indices[order]
            distances[ordered_indices[0]] = torch.inf
            distances[ordered_indices[-1]] = torch.inf
            span = (
                front[order[-1], objective_index] - front[order[0], objective_index]
            ).clamp_min(1e-12)
            interior = ordered_indices[1:-1]
            increment = (
                front[order[2:], objective_index] - front[order[:-2], objective_index]
            ) / span
            finite = torch.isfinite(distances[interior])
            distances[interior[finite]] += increment[finite]
    return distances


def select_pareto_elites(
    candidates: torch.Tensor,
    objectives: torch.Tensor,
    elite_fraction: float,
    rank_temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select Pareto elites by front rank and descending crowding distance."""
    count = _elite_count(len(candidates), elite_fraction)
    objectives = objectives.detach()
    ranks = pareto_front_ranks(objectives)
    crowding = crowding_distance(objectives, ranks)
    # Python sorting makes the lexicographic front/crowding rule explicit and
    # keeps all objective-derived quantities outside the model graph.
    order_list = sorted(
        range(len(candidates)),
        key=lambda index: (int(ranks[index]), -float(crowding[index])),
    )
    indices = torch.as_tensor(order_list[:count], device=candidates.device)
    weights = rank_weights(count, rank_temperature, candidates.device, candidates.dtype)
    return candidates[indices].detach(), weights, objectives[indices].detach(), indices
