"""Preference-conditioned VAE training for black-box Pareto sets."""

from __future__ import annotations

import math

import torch
from tqdm import tqdm

from probfun.mutifitness import zdt1 as fitness_function
from scripts.elite_training import (
    crowding_distance,
    pareto_front_ranks,
    select_pareto_elites,
    weighted_reconstruction_loss,
)
from scripts.train import diversity_loss, _optimizer, _population_split


@torch.no_grad()
def sample_preferences(count, objective_count, device, dtype):
    """Sample approximately uniform points on the objective-weight simplex."""
    exponential = -torch.log(
        torch.rand(count, objective_count, device=device, dtype=dtype).clamp_min(1e-7)
    )
    return exponential / exponential.sum(dim=1, keepdim=True)


@torch.no_grad()
def simplex_reference_directions(count, objective_count, device, dtype):
    """Create deterministic Das-Dennis directions on an M-objective simplex."""
    count = max(int(count), 1)
    objective_count = int(objective_count)
    if objective_count < 2:
        raise ValueError("objective_count must be at least two")
    if count < objective_count:
        return torch.eye(objective_count, device=device, dtype=dtype)[:count]

    partitions = 1
    while math.comb(partitions + objective_count, objective_count - 1) <= count:
        partitions += 1

    compositions = []

    def enumerate_compositions(remaining, dimensions, prefix):
        if dimensions == 1:
            compositions.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            enumerate_compositions(
                remaining - value, dimensions - 1, prefix + [value]
            )

    enumerate_compositions(partitions, objective_count, [])
    directions = torch.tensor(compositions, device=device, dtype=dtype)
    return directions / float(partitions)


@torch.no_grad()
def _reference_direction_anchors(
    archive_x,
    archive_objectives,
    count,
    augmentation=0.05,
):
    """Associate every simplex direction with an augmented-Tchebycheff anchor."""
    directions = simplex_reference_directions(
        int(count),
        archive_objectives.shape[1],
        archive_objectives.device,
        archive_objectives.dtype,
    )
    ideal = archive_objectives.amin(dim=0, keepdim=True)
    span = (archive_objectives.amax(dim=0, keepdim=True) - ideal).clamp_min(1e-8)
    normalized = (archive_objectives - ideal) / span
    # Directions describe desired objective-space rays. Augmented
    # Tchebycheff uses the reciprocal coefficients: a zero ray coordinate
    # must be penalized strongly, while a large coordinate may remain large.
    weights = 1.0 / directions.clamp_min(1e-4)
    weights = weights / weights.sum(dim=1, keepdim=True)

    selected = []
    chunk_size = 256
    for start in range(0, len(weights), chunk_size):
        chunk = weights[start:start + chunk_size]
        weighted = chunk[:, None, :] * normalized[None, :, :]
        score = weighted.amax(dim=2) + float(augmentation) * weighted.sum(dim=2)
        selected.append(score.argmin(dim=1))
    indices = torch.cat(selected)
    return (
        archive_x[indices].detach(),
        archive_objectives[indices].detach(),
        directions.detach(),
    )


@torch.no_grad()
def _many_objective_survival_indices(objectives, count):
    """Select a well-spread many-objective front using directions and FPS."""
    count = min(int(count), len(objectives))
    if count == len(objectives):
        return torch.arange(len(objectives), device=objectives.device)
    ideal = objectives.amin(dim=0, keepdim=True)
    span = (objectives.amax(dim=0, keepdim=True) - ideal).clamp_min(1e-8)
    normalized = (objectives - ideal) / span
    directions = simplex_reference_directions(
        count, objectives.shape[1], objectives.device, objectives.dtype
    )
    weights = 1.0 / directions.clamp_min(1e-4)
    weights = weights / weights.sum(dim=1, keepdim=True)

    representatives = []
    for start in range(0, len(weights), 256):
        chunk = weights[start:start + 256]
        weighted = chunk[:, None, :] * normalized[None, :, :]
        score = weighted.amax(dim=2) + 0.05 * weighted.sum(dim=2)
        representatives.append(score.argmin(dim=1))
    selected = torch.unique(torch.cat(representatives), sorted=False)

    if len(selected) < count:
        available = torch.ones(
            len(objectives), device=objectives.device, dtype=torch.bool
        )
        available[selected] = False
        if len(selected):
            minimum_distance = torch.cdist(
                normalized, normalized[selected]
            ).amin(dim=1)
        else:
            minimum_distance = normalized.norm(dim=1)
        minimum_distance[~available] = -torch.inf
        additions = []
        for _ in range(count - len(selected)):
            index = minimum_distance.argmax()
            if not torch.isfinite(minimum_distance[index]):
                break
            additions.append(index)
            available[index] = False
            distance = torch.norm(
                normalized - normalized[index:index + 1], dim=1
            )
            minimum_distance = torch.minimum(minimum_distance, distance)
            minimum_distance[~available] = -torch.inf
        if additions:
            selected = torch.cat((selected, torch.stack(additions)))
    return selected[:count]


@torch.no_grad()
def preferences_from_objectives(objectives):
    """Assign each Pareto point a trade-off condition from normalized costs."""
    lower = objectives.amin(dim=0, keepdim=True)
    span = (objectives.amax(dim=0, keepdim=True) - lower).clamp_min(1e-8)
    normalized = (objectives - lower) / span
    preference = 1.0 / (normalized + 0.05)
    return preference / preference.sum(dim=1, keepdim=True)


def pareto_set_loss(generated, elite):
    """Symmetric Chamfer loss between generated and Pareto-elite decision sets."""
    squared_distance = torch.cdist(generated, elite.detach()).pow(2)
    generated_to_elite = squared_distance.amin(dim=1).mean()
    elite_to_generated = squared_distance.amin(dim=0).mean()
    return generated_to_elite + elite_to_generated


@torch.no_grad()
def _update_pareto_archive(
    archive_x, archive_objectives, candidates, objectives, archive_size
):
    """Keep a persistent, crowding-balanced nondominated archive."""
    if archive_x is None:
        combined_x, combined_objectives = candidates, objectives
    else:
        combined_x = torch.cat((archive_x, candidates), dim=0)
        combined_objectives = torch.cat((archive_objectives, objectives), dim=0)
    ranks = pareto_front_ranks(combined_objectives)
    indices = torch.nonzero(ranks == 0, as_tuple=False).flatten()
    front_x = combined_x[indices]
    front_objectives = combined_objectives[indices]
    unique_objectives, inverse = torch.unique(
        front_objectives, dim=0, return_inverse=True
    )
    if len(unique_objectives) < len(front_objectives):
        first = torch.stack(
            [
                torch.nonzero(inverse == index, as_tuple=False)[0, 0]
                for index in range(len(unique_objectives))
            ]
        )
        front_x = front_x[first]
        front_objectives = front_objectives[first]
    if len(front_x) > archive_size:
        if front_objectives.shape[1] == 2:
            front_ranks = torch.zeros(
                len(front_x), device=front_x.device, dtype=torch.long
            )
            crowding = crowding_distance(front_objectives, front_ranks)
            keep = torch.argsort(crowding, descending=True)[:archive_size]
        else:
            keep = _many_objective_survival_indices(
                front_objectives, int(archive_size)
            )
        front_x = front_x[keep]
        front_objectives = front_objectives[keep]
    return front_x.detach(), front_objectives.detach()


@torch.no_grad()
def _objective_discontinuities(ordered_objectives, gap_factor=8.0):
    """Detect large adjacent gaps in an ordered two-objective Pareto front."""
    if len(ordered_objectives) < 3:
        return torch.zeros(
            max(len(ordered_objectives) - 1, 0),
            device=ordered_objectives.device,
            dtype=torch.bool,
        )
    distance = (
        ordered_objectives[1:] - ordered_objectives[:-1]
    ).abs().sum(dim=1)
    positive = distance[distance > 1e-12]
    if not len(positive):
        return torch.zeros_like(distance, dtype=torch.bool)
    typical = positive.median().clamp_min(1e-12)
    return distance > typical * float(gap_factor)


@torch.no_grad()
def _validated_objective_discontinuities(
    ordered_x,
    ordered_objectives,
    fitness,
    scale,
    gap_factor,
):
    """Confirm a geometric gap only when its decision midpoint is dominated."""
    candidates = _objective_discontinuities(ordered_objectives, gap_factor)
    if fitness is None or not bool(candidates.any()):
        return candidates
    indices = torch.nonzero(candidates, as_tuple=False).flatten()
    midpoints = 0.5 * (ordered_x[indices] + ordered_x[indices + 1])
    midpoint_objectives = _evaluate_objectives(fitness, midpoints, scale)
    combined = torch.cat((ordered_objectives, midpoint_objectives), dim=0)
    midpoint_ranks = pareto_front_ranks(combined)[len(ordered_objectives):]
    validated = torch.zeros_like(candidates)
    validated[indices] = midpoint_ranks > 0
    return validated


@torch.no_grad()
def _uniform_pareto_anchors(
    archive_x,
    archive_objectives,
    count,
    discontinuity_gap_factor=16.0,
    fitness=None,
    scale=True,
):
    """Sample a two-objective front uniformly without bridging discontinuities."""
    requested_count = int(count)
    if archive_objectives.shape[1] != 2:
        return _reference_direction_anchors(
            archive_x, archive_objectives, requested_count
        )
    count = min(requested_count, len(archive_x))

    order = torch.argsort(archive_objectives[:, 0])
    ordered_x = archive_x[order]
    ordered_objectives = archive_objectives[order]
    # pymoo's SpacingIndicator uses city-block (L1) nearest-neighbour
    # distance, so use the same objective-space geometry for uniform anchors.
    segment = (ordered_objectives[1:] - ordered_objectives[:-1]).abs().sum(dim=1)
    discontinuities = _validated_objective_discontinuities(
        ordered_x,
        ordered_objectives,
        fitness,
        scale,
        discontinuity_gap_factor,
    )
    continuous_segment = segment.masked_fill(discontinuities, 0.0)
    arc = torch.cat(
        (continuous_segment.new_zeros(1), torch.cumsum(continuous_segment, dim=0))
    )
    arc = arc / arc[-1].clamp_min(1e-8)
    desired = torch.linspace(0.0, 1.0, count, device=arc.device, dtype=arc.dtype)
    right = torch.searchsorted(arc, desired).clamp(max=len(arc) - 1)
    left = (right - 1).clamp(min=0)
    denominator = (arc[right] - arc[left]).clamp_min(1e-8)
    interpolation = ((desired - arc[left]) / denominator).unsqueeze(1)
    interpolation[right == left] = 0.0
    targets = torch.lerp(ordered_x[left], ordered_x[right], interpolation)
    objectives = torch.lerp(
        ordered_objectives[left], ordered_objectives[right], interpolation
    )
    # A uniform preference grid is now explicitly aligned to objective arc length.
    preferences = torch.stack((1.0 - desired, desired), dim=1)
    return targets.detach(), objectives.detach(), preferences.detach()


@torch.no_grad()
def _discrete_pareto_anchors(archive_x, archive_objectives):
    """Preserve every discrete Pareto point and map preferences by L1 arc."""
    order = torch.argsort(archive_objectives[:, 0])
    targets = archive_x[order]
    objectives = archive_objectives[order]
    if len(targets) == 1:
        coordinate = objectives.new_zeros(1)
    else:
        segment = (objectives[1:] - objectives[:-1]).abs().sum(dim=1)
        coordinate = torch.cat((segment.new_zeros(1), torch.cumsum(segment, dim=0)))
        coordinate = coordinate / coordinate[-1].clamp_min(1e-12)
    preferences = torch.stack((1.0 - coordinate, coordinate), dim=1)
    return targets.detach(), objectives.detach(), preferences.detach()


@torch.no_grad()
def _dominance_polish(
    targets,
    fitness,
    scale,
    steps,
    step_start,
    step_end,
):
    """Apply black-box Pareto-dominance coordinate search to archive anchors."""
    if steps <= 0:
        return targets.detach(), _evaluate_objectives(fitness, targets, scale)
    polished = targets.detach().clone()
    objectives = _evaluate_objectives(fitness, polished, scale)
    dimensions = polished.shape[1]
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    for iteration in range(steps):
        progress = iteration / max(steps - 1, 1)
        step = step_start * (step_end / step_start) ** progress
        proposals = []
        for dimension in range(dimensions):
            plus = polished.clone()
            minus = polished.clone()
            plus[:, dimension].add_(step).clamp_(lower, upper)
            minus[:, dimension].sub_(step).clamp_(lower, upper)
            proposals.extend((plus, minus))
        query = torch.stack(proposals, dim=0)
        query_objectives = _evaluate_objectives(
            fitness, query.reshape(-1, dimensions), scale
        ).reshape(2 * dimensions, len(polished), -1)
        dominance_tolerance = max(
            1e-10, 10.0 * torch.finfo(objectives.dtype).eps
        )
        no_worse = (
            query_objectives
            <= objectives.unsqueeze(0) + dominance_tolerance
        ).all(dim=2)
        strictly_better = (
            query_objectives
            < objectives.unsqueeze(0) - dominance_tolerance
        ).any(dim=2)
        dominates = no_worse & strictly_better
        improvement = (
            objectives.unsqueeze(0) - query_objectives
        ).clamp_min(0.0).sum(dim=2)
        improvement = improvement.masked_fill(~dominates, -torch.inf)
        best_move = improvement.argmax(dim=0)
        has_move = dominates.any(dim=0)
        column = torch.arange(len(polished), device=polished.device)
        polished[has_move] = query[best_move[has_move], column[has_move]]
        objectives[has_move] = query_objectives[
            best_move[has_move], column[has_move]
        ]
    return polished.detach(), objectives.detach()


@torch.no_grad()
def _preference_objective_polish(
    targets,
    target_objectives,
    fitness,
    scale,
    steps,
    step_start,
    step_end,
):
    """Align each decision anchor to its desired objective-space location."""
    if steps <= 0:
        return targets.detach(), _evaluate_objectives(fitness, targets, scale)
    polished = targets.detach().clone()
    objectives = _evaluate_objectives(fitness, polished, scale)
    dimensions = polished.shape[1]
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    span = (
        target_objectives.amax(dim=0, keepdim=True)
        - target_objectives.amin(dim=0, keepdim=True)
    ).clamp_min(1e-8)

    def error(values):
        return ((values - target_objectives) / span).pow(2).sum(dim=1)

    current_error = error(objectives)
    for iteration in range(int(steps)):
        progress = iteration / max(int(steps) - 1, 1)
        step = step_start * (step_end / step_start) ** progress
        proposals = []
        for dimension in range(dimensions):
            plus = polished.clone()
            minus = polished.clone()
            plus[:, dimension].add_(step).clamp_(lower, upper)
            minus[:, dimension].sub_(step).clamp_(lower, upper)
            proposals.extend((plus, minus))
        query = torch.stack(proposals, dim=0)
        query_objectives = _evaluate_objectives(
            fitness, query.reshape(-1, dimensions), scale
        ).reshape(2 * dimensions, len(polished), -1)
        query_error = (
            (query_objectives - target_objectives.unsqueeze(0)) / span
        ).pow(2).sum(dim=2)
        best_move = query_error.argmin(dim=0)
        column = torch.arange(len(polished), device=polished.device)
        best_error = query_error[best_move, column]
        improved = best_error < current_error
        polished[improved] = query[best_move[improved], column[improved]]
        objectives[improved] = query_objectives[
            best_move[improved], column[improved]
        ]
        current_error[improved] = best_error[improved]
    return polished.detach(), objectives.detach()


@torch.no_grad()
def _preference_scalarized_polish(
    targets,
    preferences,
    fitness,
    scale,
    ideal,
    nadir,
    steps,
    step_start,
    step_end,
    pbi_penalty=10.0,
):
    """Optimize one PBI black-box subproblem per objective-space direction."""
    if steps <= 0:
        return targets.detach(), _evaluate_objectives(fitness, targets, scale)
    polished = targets.detach().clone()
    objectives = _evaluate_objectives(fitness, polished, scale)
    dimensions = polished.shape[1]
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    span = (nadir - ideal).clamp_min(1e-8)
    directions = preferences / preferences.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-8)

    def scalarize(values):
        normalized = ((values - ideal) / span).clamp_min(0.0)
        along = (normalized * directions).sum(dim=1)
        perpendicular = (
            normalized - along.unsqueeze(1) * directions
        ).norm(dim=1)
        return along + float(pbi_penalty) * perpendicular

    current_score = scalarize(objectives)
    for iteration in range(int(steps)):
        progress = iteration / max(int(steps) - 1, 1)
        step = step_start * (step_end / step_start) ** progress
        proposals = []
        for dimension in range(dimensions):
            plus = polished.clone()
            minus = polished.clone()
            plus[:, dimension].add_(step).clamp_(lower, upper)
            minus[:, dimension].sub_(step).clamp_(lower, upper)
            proposals.extend((plus, minus))
        query = torch.stack(proposals, dim=0)
        query_objectives = _evaluate_objectives(
            fitness, query.reshape(-1, dimensions), scale
        ).reshape(2 * dimensions, len(polished), -1)
        normalized = (
            (query_objectives - ideal.unsqueeze(0)) / span.unsqueeze(0)
        ).clamp_min(0.0)
        along = (
            normalized * directions.unsqueeze(0)
        ).sum(dim=2)
        perpendicular = (
            normalized
            - along.unsqueeze(2) * directions.unsqueeze(0)
        ).norm(dim=2)
        query_score = along + float(pbi_penalty) * perpendicular
        best_move = query_score.argmin(dim=0)
        column = torch.arange(len(polished), device=polished.device)
        best_score = query_score[best_move, column]
        improved = best_score < current_score - 1e-12
        polished[improved] = query[best_move[improved], column[improved]]
        objectives[improved] = query_objectives[
            best_move[improved], column[improved]
        ]
        current_score[improved] = best_score[improved]
    return polished.detach(), objectives.detach()


@torch.no_grad()
def _many_objective_components(
    targets,
    objectives,
    preferences,
    fitness,
    scale,
    neighbor_count=8,
):
    """Detect disconnected Pareto regions using a normalized objective kNN graph."""
    del targets, preferences, fitness, scale
    count = len(objectives)
    if count < 2:
        return torch.zeros(count, device=objectives.device, dtype=torch.long)

    lower = objectives.amin(dim=0, keepdim=True)
    span = (objectives.amax(dim=0, keepdim=True) - lower).clamp_min(1e-8)
    normalized = (objectives - lower) / span
    pairwise = torch.cdist(normalized, normalized)
    k = min(int(neighbor_count) + 1, count)
    neighbor_distance, neighbors = torch.topk(
        pairwise, k, largest=False, dim=1
    )
    neighbors = neighbors[:, 1:]
    neighbor_distance = neighbor_distance[:, 1:]
    source = torch.arange(
        count, device=objectives.device
    ).unsqueeze(1).expand_as(neighbors)
    edges = torch.stack((source.reshape(-1), neighbors.reshape(-1)), dim=1)
    edge_distance = neighbor_distance.reshape(-1)
    # Directed kNN edges may occur in both directions; duplicate union
    # operations are harmless and avoid GPU-synchronizing Python reductions.
    nonzero = edge_distance[edge_distance > 1e-8]
    if not len(nonzero):
        return torch.zeros(count, device=objectives.device, dtype=torch.long)
    typical = nonzero.median().clamp_min(1e-8)
    valid_edges = edges[edge_distance <= typical * 64.0]

    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in valid_edges.detach().cpu().tolist():
        union(left, right)
    roots = [find(index) for index in range(count)]
    mapping = {}
    labels = []
    for root in roots:
        if root not in mapping:
            mapping[root] = len(mapping)
        labels.append(mapping[root])
    labels = torch.tensor(labels, device=objectives.device, dtype=torch.long)

    # Sparse boundary outliers must not create neural experts of their own.
    # A genuine disconnected branch needs at least one local neighbourhood and
    # 0.5% of the anchor support; smaller fragments join the nearest supported
    # component in normalized objective space.
    component_ids, component_sizes = torch.unique(
        labels, return_counts=True
    )
    minimum_support = max(int(neighbor_count), math.ceil(0.005 * count))
    supported = component_ids[component_sizes >= minimum_support]
    if len(supported) == 0:
        supported = component_ids[component_sizes.argmax():component_sizes.argmax() + 1]
    if len(supported) < len(component_ids):
        centroids = torch.stack(
            [normalized[labels == component].mean(dim=0) for component in supported]
        )
        unsupported_mask = ~torch.isin(labels, supported)
        nearest = torch.cdist(
            normalized[unsupported_mask], centroids
        ).argmin(dim=1)
        labels[unsupported_mask] = supported[nearest]
    _, labels = torch.unique(labels, sorted=True, return_inverse=True)
    return labels


@torch.no_grad()
def _pareto_output_statistics(
    targets,
    fitness,
    scale,
    binary_decisions=False,
    max_samples=256,
    perturbation=1e-2,
):
    """Learn fixed coordinates and black-box objective sensitivity per dimension."""
    dimensions = targets.shape[1]
    device, dtype = targets.device, targets.dtype
    if binary_decisions:
        return (
            torch.zeros(dimensions, device=device, dtype=torch.bool),
            targets.median(dim=0).values,
            torch.ones(dimensions, device=device, dtype=dtype),
        )

    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    lower_quantile = torch.quantile(targets, 0.05, dim=0)
    upper_quantile = torch.quantile(targets, 0.95, dim=0)
    robust_spread = upper_quantile - lower_quantile
    domain_span = float(upper - lower)
    fixed_threshold = robust_spread.new_tensor(1e-3 * domain_span)
    positive_spread = torch.sort(
        robust_spread[robust_spread > 1e-10]
    ).values
    if len(positive_spread) > 1:
        adjacent_ratio = (
            positive_spread[1:]
            / positive_spread[:-1].clamp_min(1e-10)
        )
        gap_index = int(adjacent_ratio.argmax())
        gap_ratio = float(adjacent_ratio[gap_index])
        small_group_max = positive_spread[gap_index]
        if (
            gap_ratio >= 32.0
            and float(small_group_max) <= 0.01 * domain_span
        ):
            fixed_threshold = torch.maximum(
                fixed_threshold, 1.05 * small_group_max
            )
    fixed_mask = robust_spread <= fixed_threshold
    fixed_values = targets.median(dim=0).values.clamp(lower, upper)

    sample_count = min(int(max_samples), len(targets))
    indices = torch.linspace(
        0, len(targets) - 1, sample_count, device=device
    ).round().long()
    sample = targets[indices]
    baseline = _evaluate_objectives(fitness, sample, scale)
    objective_lower = baseline.amin(dim=0, keepdim=True)
    objective_span = (
        baseline.amax(dim=0, keepdim=True) - objective_lower
    ).clamp_min(1e-6)
    responses = []
    step = float(perturbation) * (upper - lower)
    for dimension in range(dimensions):
        plus = sample.clone()
        minus = sample.clone()
        plus[:, dimension].add_(step).clamp_(lower, upper)
        minus[:, dimension].sub_(step).clamp_(lower, upper)
        plus_objectives = _evaluate_objectives(fitness, plus, scale)
        minus_objectives = _evaluate_objectives(fitness, minus, scale)
        response = torch.maximum(
            (plus_objectives - baseline).abs(),
            (minus_objectives - baseline).abs(),
        ) / objective_span
        responses.append(response.amax(dim=1).median())
    sensitivity = torch.stack(responses)
    positive = sensitivity[sensitivity > 1e-10]
    typical = positive.median() if len(positive) else sensitivity.new_tensor(1.0)
    decision_weights = torch.sqrt(
        sensitivity / typical.clamp_min(1e-10)
    ).clamp(0.25, 64.0)
    decision_weights[fixed_mask] = 1.0
    decision_weights = decision_weights / decision_weights.mean().clamp_min(1e-8)
    return fixed_mask, fixed_values, decision_weights


@torch.no_grad()
def _blackbox_objective_gradient(
    generated,
    target_objectives,
    fitness,
    scale,
    objective_lower,
    objective_span,
    step=1e-3,
):
    """Central-difference gradient of normalized objective matching loss."""
    dimensions = generated.shape[1]
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    baseline = _evaluate_objectives(fitness, generated, scale)
    normalized_baseline = (baseline - objective_lower) / objective_span
    normalized_target = (target_objectives - objective_lower) / objective_span
    residual = normalized_baseline - normalized_target

    proposals = []
    for dimension in range(dimensions):
        plus = generated.detach().clone()
        minus = generated.detach().clone()
        plus[:, dimension].add_(step).clamp_(lower, upper)
        minus[:, dimension].sub_(step).clamp_(lower, upper)
        proposals.extend((plus, minus))
    query = torch.stack(proposals, dim=0)
    query_objectives = _evaluate_objectives(
        fitness, query.reshape(-1, dimensions), scale
    ).reshape(2 * dimensions, len(generated), -1)
    plus = query_objectives[0::2]
    minus = query_objectives[1::2]
    normalized_jacobian = (
        (plus - minus)
        / (2.0 * float(step))
        / objective_span.unsqueeze(0)
    )
    gradient = 2.0 * (
        normalized_jacobian * residual.unsqueeze(0)
    ).mean(dim=2).transpose(0, 1)
    gradient_norm = gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gradient = gradient * torch.clamp(10.0 / gradient_norm, max=1.0)
    objective_error = residual.pow(2).mean()
    return gradient.detach(), objective_error.detach()


def _calibrate_preference_decoder(
    model,
    targets,
    preferences,
    components,
    steps,
    learning_rate,
    latent_repeats,
    endpoint_weight,
    batch_size=256,
    beta=1e-4,
    prior_weight=0.25,
    latent_weight=0.01,
    decision_weights=None,
    fitness=None,
    scale=True,
    target_objectives=None,
    objective_weight=0.1,
    objective_step=1e-3,
    objective_batch_size=32,
    set_loss_weight=1.0,
):
    """Distil the polished Pareto set into the neural conditional VAE.

    Archive anchors are supervision only.  Both posterior reconstructions and
    prior samples pass through trainable neural decoders; no anchor lookup is
    used in this stage or in normal inference.
    """
    if steps <= 0:
        return None
    model.set_generation_mode("decoder")
    routed = components is not None and int(torch.unique(components).numel()) > 1
    if routed:
        component_count = int(components.max().item()) + 1
        model.configure_component_decoder(component_count)

    parameters = list(model.encoder.parameters())
    if routed:
        parameters += list(model.pareto_router.parameters())
        parameters += list(model.pareto_component_embedding.parameters())
        parameters += list(model.pareto_component_decoder.parameters())
    else:
        parameters += list(model.decoder.parameters())
        model.pareto_router_enabled = False
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    latent_dim = model.vae_latent_dim
    batch_size = min(max(int(batch_size), 2), len(targets))
    repeats = max(int(latent_repeats), 1)
    last_metrics = None
    use_objective_distillation = (
        fitness is not None
        and target_objectives is not None
        and float(objective_weight) > 0.0
        # A sub-threshold continuous perturbation cannot change a hard binary
        # decision, so central differences are identically zero for ZDT5.
        and not bool(getattr(model, "binary_decisions", False))
    )
    if use_objective_distillation:
        objective_lower = target_objectives.amin(dim=0, keepdim=True)
        objective_span = (
            target_objectives.amax(dim=0, keepdim=True) - objective_lower
        ).clamp_min(1e-6)
    if decision_weights is None:
        decision_weights = torch.ones(
            targets.shape[1], device=targets.device, dtype=targets.dtype
        )
    decision_weights = decision_weights / decision_weights.mean().clamp_min(1e-8)

    def weighted_error(prediction, target):
        return (
            (prediction - target).pow(2)
            * decision_weights.unsqueeze(0)
        ).mean(dim=1)

    def neural_decode(z, condition, labels=None):
        if not getattr(model, "preference_conditioning_enabled", True):
            condition = torch.full_like(condition, 1.0 / model.preference_dim)
        if routed:
            return model.decode_with_components(z, condition, labels)
        decisions = model.decoder(torch.cat((z, condition), dim=1))
        return model._apply_output_constraints(decisions)

    component_indices = None
    if routed:
        component_indices = [
            torch.nonzero(components == component, as_tuple=False).flatten()
            for component in range(model.pareto_component_count)
        ]

    model.train()
    for _ in range(int(steps)):
        if routed:
            # Sample experts uniformly, then sample anchors within each expert.
            # This prevents small disconnected Pareto regions from being
            # overwhelmed by large regions during neural distillation.
            sampled_components = torch.randint(
                model.pareto_component_count,
                (batch_size,),
                device=targets.device,
            )
            sampled_indices = []
            for component in range(model.pareto_component_count):
                positions = torch.nonzero(
                    sampled_components == component, as_tuple=False
                ).flatten()
                if len(positions):
                    pool = component_indices[component]
                    chosen = pool[
                        torch.randint(
                            len(pool), (len(positions),), device=targets.device
                        )
                    ]
                    sampled_indices.append((positions, chosen))
            indices = torch.empty(
                batch_size, device=targets.device, dtype=torch.long
            )
            for positions, chosen in sampled_indices:
                indices[positions] = chosen
        else:
            indices = torch.randint(
                len(targets), (batch_size,), device=targets.device
            )
        target = targets[indices]
        condition = preferences[indices]
        labels = components[indices] if routed else None

        mu, logvar = model.encode(target, condition)
        posterior_z = model.reparameterize(mu, logvar)
        reconstruction = neural_decode(posterior_z, condition, labels)
        reconstruction_loss = weighted_error(reconstruction, target).mean()
        kl_loss = -0.5 * (
            1.0 + logvar - mu.pow(2) - logvar.exp()
        ).sum(dim=1).mean()

        prior_condition = condition.repeat(repeats, 1)
        prior_target = target.repeat(repeats, 1)
        prior_labels = labels.repeat(repeats) if routed else None
        prior_z = torch.randn(
            len(prior_condition),
            latent_dim,
            device=targets.device,
            dtype=targets.dtype,
        )
        prior_generated = neural_decode(
            prior_z, prior_condition, prior_labels
        )
        # Conditional matching keeps every preference close to its polished
        # target; the symmetric set term also preserves global front coverage.
        conditional_prior_loss = weighted_error(
            prior_generated, prior_target
        ).mean()
        prior_set_loss = (
            pareto_set_loss(prior_generated, prior_target)
            if float(set_loss_weight) > 0.0
            else target.new_zeros(())
        )

        cycle_mu, _ = model.encode(prior_generated, prior_condition)
        latent_cycle_loss = (cycle_mu - prior_z).pow(2).mean()

        objective_proxy = target.new_zeros(())
        objective_error = target.new_zeros(())
        if use_objective_distillation:
            objective_count = min(
                int(objective_batch_size), len(prior_generated)
            )
            objective_indices = torch.randperm(
                len(prior_generated), device=targets.device
            )[:objective_count]
            target_objective = target_objectives[
                indices.repeat(repeats)[objective_indices]
            ]
            blackbox_gradient, objective_error = _blackbox_objective_gradient(
                prior_generated[objective_indices],
                target_objective,
                fitness,
                scale,
                objective_lower,
                objective_span,
                float(objective_step),
            )
            objective_proxy = (
                prior_generated[objective_indices]
                * blackbox_gradient
            ).sum(dim=1).mean()

        router_loss = target.new_zeros(())
        if routed:
            router_condition = condition
            if not getattr(model, "preference_conditioning_enabled", True):
                router_condition = torch.full_like(
                    condition, 1.0 / model.preference_dim
                )
            router_loss = torch.nn.functional.cross_entropy(
                model.pareto_router(router_condition), labels
            )

        endpoint = condition.amax(dim=1) >= 1.0 - 1e-6
        sample_weights = torch.ones_like(endpoint, dtype=target.dtype)
        sample_weights[endpoint] = float(endpoint_weight)
        endpoint_loss = (
            weighted_error(reconstruction, target) * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-12)

        loss = (
            0.5 * reconstruction_loss
            + 0.5 * endpoint_loss
            + float(beta) * kl_loss
            + float(prior_weight)
            * (conditional_prior_loss + float(set_loss_weight) * prior_set_loss)
            + float(latent_weight) * latent_cycle_loss
            + float(objective_weight) * objective_proxy
            + router_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        last_metrics = {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(reconstruction_loss.detach().cpu()),
            "kl": float(kl_loss.detach().cpu()),
            "prior": float(
                (conditional_prior_loss + prior_set_loss).detach().cpu()
            ),
            "latent_cycle": float(latent_cycle_loss.detach().cpu()),
            "prior_match": float(conditional_prior_loss.detach().cpu()),
            "set": float(prior_set_loss.detach().cpu()),
            "router": float(router_loss.detach().cpu()),
            "objective": float(objective_error.cpu()),
        }
    model.eval()
    return last_metrics


@torch.no_grad()
def _binary_group_polish(
    targets,
    fitness,
    scale,
    group_size,
    steps,
):
    """Optimize deceptive binary blocks through dominance-preserving group flips."""
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    threshold = (lower + upper) / 2.0
    polished = torch.where(
        targets >= threshold,
        targets.new_tensor(upper),
        targets.new_tensor(lower),
    )
    objectives = _evaluate_objectives(fitness, polished, scale)
    dimensions = polished.shape[1]
    groups = [
        (start, min(start + int(group_size), dimensions))
        for start in range(0, dimensions, int(group_size))
    ]
    for _ in range(max(int(steps), len(groups))):
        proposals = []
        for start, end in groups:
            high = polished.clone()
            low = polished.clone()
            high[:, start:end] = upper
            low[:, start:end] = lower
            proposals.extend((high, low))
        query = torch.stack(proposals, dim=0)
        query_objectives = _evaluate_objectives(
            fitness, query.reshape(-1, dimensions), scale
        ).reshape(2 * len(groups), len(polished), -1)
        dominance_tolerance = max(
            1e-10, 10.0 * torch.finfo(objectives.dtype).eps
        )
        no_worse = (
            query_objectives
            <= objectives.unsqueeze(0) + dominance_tolerance
        ).all(dim=2)
        strictly_better = (
            query_objectives
            < objectives.unsqueeze(0) - dominance_tolerance
        ).any(dim=2)
        dominates = no_worse & strictly_better
        improvement = (
            objectives.unsqueeze(0) - query_objectives
        ).clamp_min(0.0).sum(dim=2)
        improvement = improvement.masked_fill(~dominates, -torch.inf)
        best_move = improvement.argmax(dim=0)
        has_move = dominates.any(dim=0)
        column = torch.arange(len(polished), device=polished.device)
        polished[has_move] = query[best_move[has_move], column[has_move]]
        objectives[has_move] = query_objectives[
            best_move[has_move], column[has_move]
        ]
        if not bool(has_move.any()):
            break
    return polished.detach(), objectives.detach()


@torch.no_grad()
def _axis_coverage_seeds(
    fitness,
    decision_dim,
    scale,
    points,
    max_dimensions,
    device,
    dtype,
):
    """Find black-box trade-off axes and sample them on three baselines."""
    if points < 2 or max_dimensions < 1:
        return None, None
    lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
    middle = (lower + upper) / 2.0
    baselines = (lower, middle, upper)
    probe_values = torch.linspace(lower, upper, 9, device=device, dtype=dtype)
    probe_lines = []
    line_keys = []
    for baseline_index, baseline in enumerate(baselines):
        for dimension in range(decision_dim):
            line = torch.full(
                (len(probe_values), decision_dim),
                baseline,
                device=device,
                dtype=dtype,
            )
            line[:, dimension] = probe_values
            probe_lines.append(line)
            line_keys.append((baseline_index, dimension))
    probe = torch.cat(probe_lines, dim=0)
    probe_objectives = _evaluate_objectives(fitness, probe, scale).reshape(
        len(probe_lines), len(probe_values), -1
    )
    spread = (probe_objectives.amax(dim=1) - probe_objectives.amin(dim=1))
    objective_count = probe_objectives.shape[2]
    if objective_count == 2:
        # Preserve the original bi-objective axis detector used by the ZDT path.
        score = torch.prod(spread.clamp_min(1e-12), dim=1)
    else:
        normalized_probe = (
            (probe_objectives - probe_objectives.amin(dim=1, keepdim=True))
            / spread.unsqueeze(1).clamp_min(1e-12)
        )
        centered = normalized_probe - normalized_probe.mean(dim=1, keepdim=True)
        covariance = torch.einsum("lpm,lpn->lmn", centered, centered)
        upper_triangle = torch.triu(
            torch.ones(
                objective_count, objective_count,
                device=device, dtype=torch.bool,
            ),
            diagonal=1,
        )
        # Many-objective trade-off axes make objectives move in opposite
        # directions; distance axes tend to move them together.
        score = (-covariance[:, upper_triangle]).clamp_min(0.0).sum(dim=1)
    best_by_dimension = torch.zeros(decision_dim, device=device, dtype=dtype)
    for line_index, (_, dimension) in enumerate(line_keys):
        best_by_dimension[dimension] = torch.maximum(
            best_by_dimension[dimension], score[line_index]
        )
    selected = torch.argsort(best_by_dimension, descending=True)[
        : min(int(max_dimensions), decision_dim)
    ]

    values = torch.linspace(lower, upper, int(points), device=device, dtype=dtype)
    dense_lines = []
    if objective_count == 2:
        # Keep the proven ZDT strategy: dense independent lines, never a
        # joint surface across selected axes.
        for baseline in baselines:
            for dimension in selected.tolist():
                line = torch.full(
                    (len(values), decision_dim),
                    baseline,
                    device=device,
                    dtype=dtype,
                )
                line[:, dimension] = values
                dense_lines.append(line)
    elif len(selected) == 1:
        for baseline in baselines:
            line = torch.full(
                (len(values), decision_dim),
                baseline,
                device=device,
                dtype=dtype,
            )
            line[:, selected[0]] = values
            dense_lines.append(line)
    else:
        sobol = torch.quasirandom.SobolEngine(
            dimension=len(selected), scramble=True, seed=0
        ).draw(int(points)).to(device=device, dtype=dtype)
        exponents = (1.0, 0.5, 0.1, 0.01)
        transformed = sobol.clone()
        groups = torch.arange(len(sobol), device=device) % len(exponents)
        for group, exponent in enumerate(exponents):
            mask = groups == group
            transformed[mask] = sobol[mask].pow(exponent)
        transformed = transformed * (upper - lower) + lower
        for baseline in baselines:
            surface = torch.full(
                (len(transformed), decision_dim),
                baseline,
                device=device,
                dtype=dtype,
            )
            surface[:, selected] = transformed
            dense_lines.append(surface)
    candidates = torch.cat(dense_lines, dim=0)
    objectives = _evaluate_objectives(fitness, candidates, scale)
    ranks = pareto_front_ranks(objectives)
    keep = ranks == 0
    return candidates[keep].detach(), objectives[keep].detach()


@torch.no_grad()
def _sample_population(
    model,
    count,
    latent_dim,
    decision_dim,
    objective_count,
    device,
    dtype,
    epoch,
    exploration_fraction,
    scale,
    binary_decisions,
):
    model_count, exploration_count = _population_split(
        count, exploration_fraction, epoch
    )
    populations = []
    if model_count:
        z = torch.randn(model_count, latent_dim, device=device, dtype=dtype)
        preferences = sample_preferences(
            model_count, objective_count, device, dtype
        )
        populations.append(model(z, preferences))
    if exploration_count:
        lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
        uniform = torch.rand(
            exploration_count, decision_dim, device=device, dtype=dtype
        )
        populations.append(uniform * (upper - lower) + lower)
    population = torch.cat(populations, dim=0).detach()
    if epoch == 0 and len(population):
        lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
        if binary_decisions:
            seeds = []
            for cardinality in range(decision_dim + 1):
                seed = torch.full(
                    (decision_dim,), lower, device=device, dtype=dtype
                )
                seed[:cardinality] = upper
                seeds.append(seed)
            seeds = torch.stack(seeds[: len(population)])
            population[: len(seeds)] = seeds
            return population
        seeds = [
            torch.full((decision_dim,), lower, device=device, dtype=dtype),
            torch.full((decision_dim,), upper, device=device, dtype=dtype),
        ]
        for dimension in range(decision_dim):
            low_corner = torch.full(
                (decision_dim,), lower, device=device, dtype=dtype
            )
            low_corner[dimension] = upper
            seeds.append(low_corner)
            high_corner = torch.full(
                (decision_dim,), upper, device=device, dtype=dtype
            )
            high_corner[dimension] = lower
            seeds.append(high_corner)
        seeds = torch.stack(seeds[: len(population)])
        population[: len(seeds)] = seeds
    return population


@torch.no_grad()
def _evaluate_objectives(fitness, candidates, scale):
    evaluated = (candidates + 1.0) / 2.0 if scale else candidates
    result = fitness(evaluated.detach())
    if isinstance(result, (tuple, list)):
        objectives = torch.stack(
            [
                torch.as_tensor(
                    value, device=candidates.device, dtype=candidates.dtype
                ).reshape(-1)
                for value in result
            ],
            dim=1,
        )
    else:
        objectives = torch.as_tensor(
            result, device=candidates.device, dtype=candidates.dtype
        )
        if objectives.ndim == 1:
            objectives = objectives[:, None]
    if objectives.shape[0] != candidates.shape[0]:
        raise ValueError("fitness_function must return one objective vector per candidate")
    return objectives.detach()


def train_multiobjective(
    model,
    device,
    input_dim,
    latent_dim,
    batch_size,
    fitness_function=fitness_function,
    scale=True,
    optimization="Adam",
    div_weight=0.002,
    epochs=2000,
    log_interval=10,
    learning_rate=3e-4,
    elite_fraction=0.2,
    rank_temperature=0.2,
    exploration_fraction=0.2,
    train_steps=5,
    beta=1e-3,
    set_loss_weight=1.0,
    reconstruction_weight=0.25,
    archive_size=512,
    anchor_count=512,
    initial_polish_steps=32,
    dominance_polish_steps=64,
    dominance_step_start=1.0,
    dominance_step_end=1e-4,
    calibration_steps=1000,
    calibration_learning_rate=1e-3,
    calibration_latent_repeats=2,
    calibration_endpoint_weight=10.0,
    calibration_batch_size=256,
    calibration_beta=1e-4,
    calibration_prior_weight=0.25,
    calibration_latent_weight=0.01,
    calibration_objective_weight=0.1,
    calibration_objective_step=1e-3,
    calibration_objective_batch_size=32,
    discontinuity_gap_factor=16.0,
    axis_coverage_points=512,
    axis_coverage_max_dimensions=2,
    binary_decisions=False,
    binary_group_size=1,
    preference_polish_steps=64,
    preference_pbi_penalty=10.0,
    preference_conditioning=True,
    calibration_set_loss_weight=1.0,
):
    """Train a preference-conditioned VAE on detached Pareto-elite sets.

    The objective is never differentiated. A symmetric set loss covers the
    whole Pareto elite set, conditional reconstruction gives preferences a
    stable meaning, and a light KL term preserves prior generation.
    """
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    if beta < 0 or set_loss_weight < 0 or reconstruction_weight < 0:
        raise ValueError("invalid multi-objective loss weights")
    if calibration_set_loss_weight < 0:
        raise ValueError("calibration_set_loss_weight must be nonnegative")
    if discontinuity_gap_factor <= 1.0:
        raise ValueError("discontinuity_gap_factor must be greater than one")
    if binary_group_size < 1:
        raise ValueError("binary_group_size must be positive")
    if preference_pbi_penalty <= 0:
        raise ValueError("preference_pbi_penalty must be positive")
    if getattr(model, "decision_dim", input_dim) != input_dim:
        raise ValueError("model decision dimension does not match input_dim")
    if getattr(model, "vae_latent_dim", latent_dim) != latent_dim:
        raise ValueError("model VAE latent dimension does not match latent_dim")

    model = model.to(device)
    model.preference_conditioning_enabled = bool(preference_conditioning)
    optimizer = _optimizer(model, optimization, learning_rate)
    dtype = next(model.parameters()).dtype
    loss_history = []
    objective_count = model.preference_dim
    archive_x = None
    archive_objectives = None
    progress_bar = tqdm(
        range(epochs), desc="Preference-CVAE Pareto training", total=epochs
    )

    for epoch in progress_bar:
        model.eval()
        coverage_x = None
        coverage_objectives = None
        candidates = _sample_population(
            model,
            batch_size,
            latent_dim,
            input_dim,
            objective_count,
            device,
            dtype,
            epoch,
            exploration_fraction,
            scale,
            bool(binary_decisions),
        )
        objectives = _evaluate_objectives(fitness_function, candidates, scale)
        if epoch == 0 and initial_polish_steps:
            if binary_decisions:
                candidates, objectives = _binary_group_polish(
                    candidates,
                    fitness_function,
                    scale,
                    int(binary_group_size),
                    int(initial_polish_steps),
                )
            else:
                candidates, objectives = _dominance_polish(
                    candidates, fitness_function, scale, int(initial_polish_steps),
                    float(dominance_step_start), float(dominance_step_end),
                )
        if epoch == 0 and axis_coverage_points and not binary_decisions:
            coverage_x, coverage_objectives = _axis_coverage_seeds(
                fitness_function,
                input_dim,
                scale,
                int(axis_coverage_points),
                max(
                    int(axis_coverage_max_dimensions),
                    objective_count - 1 if objective_count > 2 else 1,
                ),
                device,
                dtype,
            )
            if coverage_x is not None and len(coverage_x):
                candidates = torch.cat((candidates, coverage_x), dim=0)
                objectives = torch.cat((objectives, coverage_objectives), dim=0)
        if objectives.shape[1] != objective_count:
            raise ValueError(
                f"model preference_dim={objective_count}, but objective returned "
                f"{objectives.shape[1]} objectives"
            )
        current_elite, _, current_objectives, _ = select_pareto_elites(
            candidates, objectives, elite_fraction, rank_temperature
        )
        if coverage_x is not None and len(coverage_x):
            archive_x, archive_objectives = _update_pareto_archive(
                archive_x,
                archive_objectives,
                coverage_x,
                coverage_objectives,
                int(archive_size),
            )
        archive_x, archive_objectives = _update_pareto_archive(
            archive_x, archive_objectives, current_elite,
            current_objectives, int(archive_size),
        )
        elite_x, elite_objectives, elite_preferences = _uniform_pareto_anchors(
            archive_x, archive_objectives,
            min(max(len(current_elite), 2), int(anchor_count)),
            float(discontinuity_gap_factor),
            fitness_function,
            scale,
        )
        weights = torch.full(
            (len(elite_x),), 1.0 / len(elite_x),
            device=device, dtype=dtype,
        )

        model.train()
        for _ in range(train_steps):
            reconstruction, mu, logvar = model.reconstruct(
                elite_x, elite_preferences
            )
            reconstruction_loss = weighted_reconstruction_loss(
                reconstruction, elite_x, weights
            )
            normalized_weights = weights / weights.sum().clamp_min(1e-12)
            kl_per_sample = -0.5 * (
                1.0 + logvar - mu.pow(2) - logvar.exp()
            ).sum(dim=1)
            kl_loss = torch.sum(normalized_weights * kl_per_sample)

            generated_preferences = sample_preferences(
                len(elite_x), objective_count, device, dtype
            )
            z = torch.randn(len(elite_x), latent_dim, device=device, dtype=dtype)
            generated = model(z, generated_preferences)
            set_loss = (
                pareto_set_loss(generated, elite_x)
                if set_loss_weight > 0.0
                else generated.new_zeros(())
            )
            loss = (
                set_loss_weight * set_loss
                + reconstruction_weight * reconstruction_loss
                + beta * kl_loss
            )
            if div_weight:
                loss = loss + div_weight * diversity_loss(generated)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_history.append(float(loss.detach().cpu()))
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            objective_means = elite_objectives.mean(dim=0)
            mean_text = ",".join(f"{value.item():.4f}" for value in objective_means)
            progress_bar.set_description(
                f"Preference-CVAE [{epoch + 1}/{epochs}] "
                f"set={set_loss.item():.6f} elite_mean=[{mean_text}]"
            )

    if binary_decisions:
        polished_archive, polished_objectives = _binary_group_polish(
            archive_x,
            fitness_function,
            scale,
            int(binary_group_size),
            int(dominance_polish_steps),
        )
    else:
        polished_archive, polished_objectives = _dominance_polish(
            archive_x, fitness_function, scale, int(dominance_polish_steps),
            float(dominance_step_start), float(dominance_step_end),
        )
    if binary_decisions:
        threshold = 0.0 if scale else 0.5
        lower, upper = (-1.0, 1.0) if scale else (0.0, 1.0)
        polished_archive = torch.where(
            polished_archive >= threshold,
            polished_archive.new_tensor(upper),
            polished_archive.new_tensor(lower),
        )
        polished_objectives = _evaluate_objectives(
            fitness_function, polished_archive, scale
        )
        polished_archive, polished_objectives = _update_pareto_archive(
            None, None, polished_archive, polished_objectives, int(archive_size)
        )
    if binary_decisions:
        final_targets, final_objectives, final_preferences = (
            _discrete_pareto_anchors(polished_archive, polished_objectives)
        )
    else:
        final_targets, desired_objectives, final_preferences = _uniform_pareto_anchors(
            polished_archive,
            polished_objectives,
            int(anchor_count),
            float(discontinuity_gap_factor),
            fitness_function,
            scale,
        )
        if objective_count == 2:
            final_targets, final_objectives = _preference_objective_polish(
                final_targets,
                desired_objectives,
                fitness_function,
                scale,
                int(preference_polish_steps),
                float(dominance_step_start),
                float(dominance_step_end),
            )
        else:
            final_targets, final_objectives = _preference_scalarized_polish(
                final_targets,
                final_preferences,
                fitness_function,
                scale,
                polished_objectives.amin(dim=0),
                polished_objectives.amax(dim=0),
                int(preference_polish_steps),
                float(dominance_step_start),
                float(dominance_step_end),
                float(preference_pbi_penalty),
            )
            final_targets, final_objectives = _dominance_polish(
                final_targets,
                fitness_function,
                scale,
                int(dominance_polish_steps),
                float(dominance_step_start),
                float(dominance_step_end),
            )
    model.pareto_archive = polished_archive.detach()
    model.pareto_archive_objectives = polished_objectives.detach()
    model.pareto_anchor_targets = final_targets.detach()
    model.pareto_anchor_objectives = final_objectives.detach()
    model.pareto_anchor_preferences = final_preferences.detach()

    component_labels = None
    if objective_count == 2:
        model.pareto_anchor_breaks = _validated_objective_discontinuities(
            final_targets,
            final_objectives,
            fitness_function,
            scale,
            float(discontinuity_gap_factor),
        ).detach()
        if len(model.pareto_anchor_breaks):
            component_labels = torch.cat(
                (
                    torch.zeros(
                        1, device=final_targets.device, dtype=torch.long
                    ),
                    torch.cumsum(
                        model.pareto_anchor_breaks.to(torch.long), dim=0
                    ),
                )
            )
    else:
        model.pareto_anchor_breaks = torch.zeros(
            0, device=final_targets.device, dtype=torch.bool
        )
        if len(final_preferences) > 1:
            neighbor_count = min(8, len(final_preferences))
            preference_distance = torch.cdist(
                final_preferences, final_preferences
            )
            neighbor_index = torch.topk(
                preference_distance,
                neighbor_count,
                largest=False,
                dim=1,
            ).indices
            lower_objective = final_objectives.amin(dim=0, keepdim=True)
            objective_span = (
                final_objectives.amax(dim=0, keepdim=True) - lower_objective
            ).clamp_min(1e-8)
            normalized_objectives = (
                final_objectives - lower_objective
            ) / objective_span
            local_objectives = normalized_objectives[neighbor_index]
            local_spread = torch.cdist(
                local_objectives, local_objectives
            ).amax(dim=(1, 2))
            positive_spread = local_spread[local_spread > 1e-8]
            model.pareto_interpolation_typical_spread = (
                float(positive_spread.median())
                if len(positive_spread)
                else 0.0
            )
            model.pareto_interpolation_neighbors = neighbor_count
            model.pareto_interpolation_gap_factor = 6.0
            component_labels = _many_objective_components(
                final_targets,
                final_objectives,
                final_preferences,
                fitness_function,
                scale,
                neighbor_count,
            ).detach()

    if component_labels is None:
        component_labels = torch.zeros(
            len(final_targets), device=final_targets.device, dtype=torch.long
        )
    # Remove empty numeric labels so routing classes are contiguous.
    _, component_labels = torch.unique(
        component_labels, sorted=True, return_inverse=True
    )
    model.pareto_anchor_components = component_labels.detach()
    model.pareto_disconnected = bool(
        torch.unique(component_labels).numel() > 1
    )

    fixed_mask, fixed_values, decision_weights = _pareto_output_statistics(
        final_targets,
        fitness_function,
        scale,
        bool(binary_decisions),
    )
    model.configure_pareto_output(
        fixed_mask, fixed_values, decision_weights
    )
    model.pareto_fixed_dimensions = torch.nonzero(
        fixed_mask, as_tuple=False
    ).flatten().detach()
    model.pareto_decision_sensitivity_weights = decision_weights.detach()

    # Output constraints must already be active while the neural decoder is
    # distilled. Binary problems use the straight-through estimator in PVAE.
    model.binary_decisions = bool(binary_decisions)
    model.binary_threshold = 0.0 if scale else 0.5
    model.binary_lower = -1.0 if scale else 0.0
    model.binary_upper = 1.0

    calibration_loss = _calibrate_preference_decoder(
        model,
        final_targets,
        final_preferences,
        component_labels if model.pareto_disconnected else None,
        int(calibration_steps),
        float(calibration_learning_rate),
        int(calibration_latent_repeats),
        float(calibration_endpoint_weight),
        int(calibration_batch_size),
        float(calibration_beta),
        float(calibration_prior_weight),
        float(calibration_latent_weight),
        decision_weights,
        fitness_function,
        scale,
        final_objectives,
        float(calibration_objective_weight),
        float(calibration_objective_step),
        int(calibration_objective_batch_size),
        float(calibration_set_loss_weight),
    )
    model.pareto_calibration_loss = calibration_loss
    # Normal inference is necessarily neural. Anchors remain stored solely for
    # diagnostics and explicit ablation through set_generation_mode("anchor").
    model.pareto_interpolation_enabled = False
    model.set_generation_mode("decoder")
    model.eval()
    print("Preference-conditioned VAE Pareto training completed.")
    return model, loss_history


# Backward-compatible module-level alias. New code should import the explicit
# public name from ``scripts.train``.
train = train_multiobjective

__all__ = [
    "train_multiobjective",
    "train",
    "sample_preferences",
    "simplex_reference_directions",
]
