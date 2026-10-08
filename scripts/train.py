"""Elite global search with orthogonal zero-order local refinement.

The black-box objective never enters autograd. Elite ranking supplies global
distribution targets, while central directional differences refine outputs.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from probfun.fitness import rosenbrock as fitness_function
from scripts.elite_training import (
    evaluate_objective,
    rank_weights,
    select_ranked_elites,
    weighted_elbo_loss,
    weighted_reconstruction_loss,
)


def diversity_loss(samples: torch.Tensor) -> torch.Tensor:
    """Encourage pairwise diversity in a generated decision population.

    The original full pairwise-distance mean is preserved so moving this
    helper does not change existing training results.
    """
    if samples.ndim != 2:
        raise ValueError("samples must have shape [batch_size, decision_dim]")
    return -torch.cdist(samples, samples).mean()


def _optimizer(model, optimization: str, learning_rate: float):
    """Build the AdamW optimizer shared by all training modules."""
    if optimization != "Adam":
        raise ValueError("Only optimization='Adam' is supported")
    return optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def _orthogonal_blackbox_gradient(
    objective_function,
    samples,
    step,
    maximize,
    max_directions,
    lower_bound,
    upper_bound,
):
    """Estimate a black-box gradient with an adaptive orthogonal basis.

    Low-dimensional problems use a complete orthogonal basis. Higher-dimensional
    problems use at most ``max_directions`` random orthogonal directions, keeping
    the query budget bounded while preserving an unbiased random-subspace scale.
    """
    if step <= 0:
        raise ValueError("blackbox_step must be positive")
    if max_directions < 1:
        raise ValueError("blackbox_max_directions must be positive")
    batch_size, dimensions = samples.shape
    direction_count = min(dimensions, int(max_directions))
    basis = torch.randn(
        dimensions,
        direction_count,
        device=samples.device,
        dtype=samples.dtype,
    )
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    directions = basis.transpose(0, 1)
    plus = samples.detach().unsqueeze(0) + step * directions.unsqueeze(1)
    minus = samples.detach().unsqueeze(0) - step * directions.unsqueeze(1)
    plus.clamp_(lower_bound, upper_bound)
    minus.clamp_(lower_bound, upper_bound)
    query = torch.cat((plus, minus), dim=0).reshape(
        2 * direction_count * batch_size, dimensions
    )
    values = evaluate_objective(objective_function, query).reshape(
        2 * direction_count, batch_size
    )
    directional_derivatives = (
        values[:direction_count] - values[direction_count:]
    ) / (2.0 * step)
    gradient = (dimensions / direction_count) * (
        directional_derivatives.transpose(0, 1) @ directions
    )
    return -gradient if maximize else gradient



def _scheduled_blackbox_weight(epoch, epochs, maximum_weight, start_fraction):
    """Linearly enable local zero-order refinement after global elite search."""
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("blackbox_start_fraction must be in [0, 1)")
    progress = epoch / max(epochs - 1, 1)
    phase_progress = max(0.0, progress - start_fraction) / (1.0 - start_fraction)
    return float(maximum_weight) * phase_progress


def _resolve_blackbox_directions(input_dim, requested):
    """Use a full basis up to the shared zero-order query cap."""
    input_dim = int(input_dim)
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if requested is None:
        return input_dim
    requested = int(requested)
    if requested < 1:
        raise ValueError("blackbox_max_directions must be positive")
    return min(input_dim, requested)


def _linear_schedule(start, end, progress):
    """Linearly interpolate a positive training hyperparameter."""
    if start <= 0 or end <= 0:
        raise ValueError("scheduled values must be positive")
    progress = min(1.0, max(0.0, float(progress)))
    return float(start) + progress * (float(end) - float(start))

def _blackbox_surrogate_loss(
    objective_function,
    samples,
    step,
    maximize,
    max_directions,
    lower_bound,
    upper_bound,
    weights=None,
):
    gradient = _orthogonal_blackbox_gradient(
        objective_function,
        samples,
        step,
        maximize,
        max_directions,
        lower_bound,
        upper_bound,
    )
    per_sample = (samples * gradient.detach()).sum(dim=1)
    if weights is None:
        return per_sample.mean()
    normalized = weights.detach() / weights.detach().sum().clamp_min(1e-12)
    return torch.sum(normalized * per_sample)

def _population_split(batch_size: int, exploration_fraction: float, epoch: int):
    """Split a population between model samples and global exploration."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in [0, 1]")
    exploration_count = (
        batch_size if epoch == 0 else round(batch_size * exploration_fraction)
    )
    exploration_count = min(batch_size, max(1, exploration_count))
    return batch_size - exploration_count, exploration_count


@torch.no_grad()
def _sample_ivae_population(
    model,
    count: int,
    input_dim: int,
    global_radius: float,
    device,
    dtype,
    epoch: int,
    exploration_fraction: float,
    archive=None,
    mutation_sigma: float = 0.1,
    mutation_fraction: float = 0.5,
):
    """Combine decoder samples, elite mutations, and global exploration."""
    if not 0.0 <= mutation_fraction <= 1.0:
        raise ValueError("mutation_fraction must be in [0, 1]")
    if epoch == 0 or archive is None or len(archive) == 0:
        model_count, exploration_count, mutation_count = 0, count, 0
    else:
        _, exploration_count = _population_split(
            count, exploration_fraction, epoch
        )
        learned_count = count - exploration_count
        mutation_count = round(learned_count * mutation_fraction)
        model_count = learned_count - mutation_count

    populations = []
    if model_count:
        z = torch.randn(model_count, model.latent_dim, device=device, dtype=dtype)
        populations.append(model.output_activation(model.decoder(z)))
    if mutation_count:
        source_count = min(len(archive), max(1, count // 4))
        indices = torch.randint(source_count, (mutation_count,), device=device)
        mutated = archive[indices] + mutation_sigma * torch.randn(
            mutation_count, input_dim, device=device, dtype=dtype
        )
        populations.append(mutated.clamp(-global_radius, global_radius))
    if exploration_count:
        uniform = (
            torch.rand(exploration_count, input_dim, device=device, dtype=dtype)
            * 2.0
            - 1.0
        ) * global_radius
        populations.append(uniform)
    return torch.cat(populations, dim=0).detach()


def train_IVAE(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function=fitness_function,
    optimization="Adam",
    objective="minimize",
    vae_weight=0.25,
    global_radius=5.0,
    epochs=1000,
    log_interval=100,
    learning_rate=3e-4,
    elite_fraction=0.2,
    rank_temperature=0.2,
    elite_fraction_end=0.05,
    rank_temperature_end=0.05,
    exploration_fraction=0.2,
    train_steps=1,
    archive_size=512,
    mutation_sigma_start=0.5,
    mutation_sigma_end=0.02,
    elite_mutation_fraction=0.5,
    blackbox_weight=0.05,
    blackbox_step=1e-3,
    blackbox_max_directions=None,
    blackbox_start_fraction=0.8,
    elite_guidance=True,
):
    """Train IVAE with optional elite guidance and zero-order refinement."""
    maximize = {"minimize": False, "maximize": True}.get(objective)
    if maximize is None:
        raise ValueError(f"Unsupported objective direction: {objective}")
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    if archive_size < 1:
        raise ValueError("archive_size must be positive")
    if mutation_sigma_start <= 0 or mutation_sigma_end <= 0:
        raise ValueError("mutation sigmas must be positive")
    blackbox_max_directions = _resolve_blackbox_directions(
        input_dim, blackbox_max_directions
    )

    model = model.to(device)
    optimizer = _optimizer(model, optimization, learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dtype = next(model.parameters()).dtype
    loss_history = []
    archive = None
    archive_values = None
    progress_bar = tqdm(range(epochs), desc="Elite+ZO IVAE training", total=epochs)

    for epoch in progress_bar:
        progress = epoch / max(epochs - 1, 1)
        effective_elite_fraction = _linear_schedule(
            elite_fraction, elite_fraction_end, progress
        )
        effective_rank_temperature = _linear_schedule(
            rank_temperature, rank_temperature_end, progress
        )
        mutation_sigma = mutation_sigma_start * (
            mutation_sigma_end / mutation_sigma_start
        ) ** progress
        effective_blackbox_weight = _scheduled_blackbox_weight(
            epoch, epochs, blackbox_weight, blackbox_start_fraction
        )

        model.eval()
        candidates = _sample_ivae_population(
            model,
            batch_size,
            input_dim,
            global_radius,
            device,
            dtype,
            epoch,
            exploration_fraction,
            archive if elite_guidance else None,
            mutation_sigma,
            elite_mutation_fraction,
        )
        values = evaluate_objective(fitness_function, candidates)
        if archive is None:
            combined_x, combined_values = candidates, values
        else:
            combined_x = torch.cat((archive, candidates), dim=0)
            combined_values = torch.cat((archive_values, values), dim=0)
        archive, archive_values = _select_diverse_archive(
            combined_x, combined_values, archive_size, 0.0, maximize
        )
        elite_x, weights, elite_values, _ = select_ranked_elites(
            archive,
            archive_values,
            effective_elite_fraction,
            effective_rank_temperature,
            maximize=maximize,
        )

        if elite_guidance:
            training_data = elite_x
            training_weights = weights
        else:
            training_data = candidates
            training_weights = torch.full(
                (len(training_data),),
                1.0 / len(training_data),
                device=device,
                dtype=dtype,
            )

        model.train()
        for _ in range(train_steps):
            reconstruction, mu, logvar = model(training_data)
            loss, _, _ = weighted_elbo_loss(
                reconstruction,
                training_data,
                mu,
                logvar,
                training_weights,
                beta=vae_weight,
            )
            if effective_blackbox_weight:
                loss = loss + effective_blackbox_weight * _blackbox_surrogate_loss(
                    fitness_function,
                    reconstruction,
                    blackbox_step,
                    maximize,
                    blackbox_max_directions,
                    -global_radius,
                    global_radius,
                    weights=training_weights,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach().cpu()))

        if (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            best = archive_values[0]
            progress_bar.set_description(
                f"Elite+ZO IVAE [{epoch + 1}/{epochs}] "
                f"objective={best.item():.6f}"
            )

    model.elite_archive = archive.detach()
    model.elite_archive_values = archive_values.detach()
    model.best_solution = archive[0].detach()
    model.best_objective = archive_values[0].detach()
    print("Elite+ZO IVAE training completed.")
    return model, loss_history


# ---------------------------------------------------------------------------
# VQ-VAE elite search
# ---------------------------------------------------------------------------


@torch.no_grad()
def _select_diverse_archive(
    candidates,
    values,
    archive_size,
    grid_size,
    maximize,
):
    """Keep the best candidate in each decision-space grid cell."""
    order = torch.argsort(values, descending=maximize)
    ordered_candidates = candidates[order]
    ordered_values = values[order]
    if grid_size <= 0:
        count = min(archive_size, len(ordered_candidates))
        return ordered_candidates[:count].detach(), ordered_values[:count].detach()

    grid_keys = (
        torch.round(ordered_candidates / grid_size).to(torch.int64).cpu().tolist()
    )
    selected = []
    occupied = set()
    for position, key_values in enumerate(grid_keys):
        key = tuple(key_values)
        if key in occupied:
            continue
        occupied.add(key)
        selected.append(position)
        if len(selected) >= archive_size:
            break

    indices = torch.as_tensor(selected, device=candidates.device, dtype=torch.long)
    return ordered_candidates[indices].detach(), ordered_values[indices].detach()


@torch.no_grad()
def _code_sampling_probabilities(
    code_values,
    maximize,
    temperature,
    uniform_mix,
):
    if temperature <= 0:
        raise ValueError("code_temperature must be positive")
    if not 0.0 <= uniform_mix <= 1.0:
        raise ValueError("code_uniform_mix must be in [0, 1]")
    count = len(code_values)
    order = torch.argsort(code_values, descending=maximize)
    ranks = torch.empty(count, device=code_values.device, dtype=code_values.dtype)
    ranks[order] = torch.arange(
        count, device=code_values.device, dtype=code_values.dtype
    )
    if count > 1:
        ranks = ranks / (count - 1)
    probabilities = torch.softmax(-ranks / temperature, dim=0)
    return ((1.0 - uniform_mix) * probabilities + uniform_mix / count).detach()


@torch.no_grad()
def _sample_vqvae_elite_population(
    model,
    count,
    input_dim,
    global_radius,
    device,
    dtype,
    epoch,
    exploration_fraction,
    code_probabilities,
    mutation_sigma,
):
    """Sample every prototype once, then mutate quality-weighted prototypes."""
    model_count, exploration_count = _population_split(
        count, exploration_fraction, epoch
    )
    populations = []
    prototype_count = 0

    if model_count:
        all_embeddings = model.vq.embeddings.weight
        decoded_codes = model.output_activation(model.decoder(all_embeddings))
        prototype_count = min(model_count, model.num_embeddings)
        if prototype_count == model.num_embeddings:
            populations.append(decoded_codes)
        else:
            prototype_indices = torch.multinomial(
                code_probabilities, prototype_count, replacement=False
            )
            populations.append(decoded_codes[prototype_indices])

        mutation_count = model_count - prototype_count
        if mutation_count:
            mutation_indices = torch.multinomial(
                code_probabilities, mutation_count, replacement=True
            )
            mutated = decoded_codes[mutation_indices]
            mutated = mutated + mutation_sigma * torch.randn_like(mutated)
            populations.append(mutated.clamp(-global_radius, global_radius))

    if exploration_count:
        uniform = (
            torch.rand(exploration_count, input_dim, device=device, dtype=dtype)
            * 2.0
            - 1.0
        ) * global_radius
        populations.append(uniform)

    return torch.cat(populations, dim=0).detach(), prototype_count


def _routing_code_count(model, requested=None):
    """Choose a small set whose maximum discrete entropy is about 2.6 bits."""
    if requested is None:
        requested = 6
    return min(model.num_embeddings, max(1, int(requested)))


@torch.no_grad()
def _quality_code_targets(model, archive, archive_values):
    """Map every code to a few distinct, highest-quality archive cells."""
    target_count = min(_routing_code_count(model), len(archive))
    pool = archive[:target_count]
    pool_values = archive_values[:target_count]
    decoded_codes = model.output_activation(
        model.decoder(model.vq.embeddings.weight)
    )
    assignments = torch.cdist(decoded_codes, pool).argmin(dim=1)
    assignments[:target_count] = torch.arange(
        target_count, device=archive.device
    )
    return pool[assignments].detach(), pool_values[assignments].detach()


def _weighted_vq_objective(model, training_data, weights, code_targets=None):
    """Rank-weight reconstruction, codebook, and commitment losses."""
    z = model.encoder(training_data)
    embeddings = model.vq.embeddings.weight
    distances = (
        z.pow(2).sum(dim=1, keepdim=True)
        + embeddings.pow(2).sum(dim=1)
        - 2.0 * z @ embeddings.t()
    )
    encoding_indices = torch.argmin(distances, dim=1)
    quantized_raw = model.vq.embeddings(encoding_indices)

    codebook_per_sample = (quantized_raw - z.detach()).pow(2).mean(dim=1)
    commitment_per_sample = (quantized_raw.detach() - z).pow(2).mean(dim=1)
    vq_per_sample = (
        codebook_per_sample
        + model.vq.commitment_cost * commitment_per_sample
    )

    quantized = z + (quantized_raw - z).detach()
    reconstruction = model.output_activation(model.decoder(quantized))
    reconstruction_loss = weighted_reconstruction_loss(
        reconstruction, training_data, weights
    )
    normalized_weights = weights.detach() / weights.detach().sum().clamp_min(1e-12)
    vq_loss = torch.sum(normalized_weights * vq_per_sample)

    decoded_codes = model.output_activation(model.decoder(embeddings))
    if code_targets is None:
        prototype_loss = decoded_codes.sum() * 0.0
    else:
        prototype_loss = F.mse_loss(decoded_codes, code_targets.detach())
    active_codes = torch.unique(encoding_indices).numel()
    return reconstruction, reconstruction_loss, vq_loss, prototype_loss, active_codes


def _train_vqvae_elite(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function=fitness_function,
    optimization="Adam",
    objective="minimize",
    vae_weight=0.1,
    div_weight=0.0,
    global_radius=5.0,
    epochs=1000,
    log_interval=100,
    learning_rate=3e-4,
    elite_fraction=0.2,
    rank_temperature=0.2,
    elite_fraction_end=0.05,
    rank_temperature_end=0.05,
    exploration_fraction=0.2,
    train_steps=1,
    archive_size=4096,
    archive_grid_size=0.1,
    single_peak=False,
    mutation_sigma_start=0.5,
    mutation_sigma_end=0.02,
    code_temperature=0.3,
    code_uniform_mix=0.05,
    code_target_weight=1.0,
    blackbox_weight=0.05,
    blackbox_step=1e-3,
    blackbox_max_directions=None,
    blackbox_start_fraction=0.8,
    elite_guidance=True,
):
    """Run VQ-VAE elite search with zero-order local refinement."""
    required = ("vq", "decoder", "encoder", "num_embeddings")
    if not all(hasattr(model, name) for name in required):
        raise TypeError(
            "train_VQVAE expects model.VAEs.VQVAE. "
            f"Received {type(model).__name__}."
        )
    maximize = {"minimize": False, "maximize": True}.get(objective)
    if maximize is None:
        raise ValueError(f"Unsupported objective direction: {objective}")
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    if archive_size < model.num_embeddings:
        raise ValueError("archive_size must be at least num_embeddings")
    if mutation_sigma_start <= 0 or mutation_sigma_end <= 0:
        raise ValueError("mutation sigmas must be positive")
    blackbox_max_directions = _resolve_blackbox_directions(
        input_dim, blackbox_max_directions
    )

    model = model.to(device)
    optimizer = _optimizer(model, optimization, learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dtype = next(model.parameters()).dtype
    code_probabilities = torch.full(
        (model.num_embeddings,),
        1.0 / model.num_embeddings,
        device=device,
        dtype=dtype,
    )
    archive = None
    archive_values = None
    loss_history = []
    progress_bar = tqdm(range(epochs), desc="Elite+ZO VQ-VAE training", total=epochs)

    for epoch in progress_bar:
        progress = epoch / max(epochs - 1, 1)
        effective_elite_fraction = _linear_schedule(
            elite_fraction, elite_fraction_end, progress
        )
        effective_rank_temperature = _linear_schedule(
            rank_temperature, rank_temperature_end, progress
        )
        effective_blackbox_weight = _scheduled_blackbox_weight(
            epoch, epochs, blackbox_weight, blackbox_start_fraction
        )
        mutation_sigma = mutation_sigma_start * (
            mutation_sigma_end / mutation_sigma_start
        ) ** progress

        model.eval()
        candidates, prototype_count = _sample_vqvae_elite_population(
            model,
            batch_size,
            input_dim,
            global_radius,
            device,
            dtype,
            epoch,
            exploration_fraction,
            code_probabilities,
            mutation_sigma,
        )
        values = evaluate_objective(fitness_function, candidates)
        elite_x, _, _, _ = select_ranked_elites(
            candidates,
            values,
            effective_elite_fraction,
            effective_rank_temperature,
            maximize=maximize,
        )
        elite_values = evaluate_objective(fitness_function, elite_x)

        if archive is None:
            combined_x, combined_values = elite_x, elite_values
        else:
            combined_x = torch.cat((archive, elite_x), dim=0)
            combined_values = torch.cat((archive_values, elite_values), dim=0)
        archive, archive_values = _select_diverse_archive(
            combined_x,
            combined_values,
            archive_size,
            0.0 if single_peak else archive_grid_size,
            maximize,
        )

        if elite_guidance:
            training_count = min(
                len(archive), max(len(elite_x), 2 * model.num_embeddings)
            )
            training_data = archive[:training_count]
            weights = rank_weights(
                training_count, effective_rank_temperature, device, dtype
            )
            code_targets, code_target_values = _quality_code_targets(
                model, archive, archive_values
            )
        else:
            training_data = candidates
            weights = torch.full(
                (len(training_data),),
                1.0 / len(training_data),
                device=device,
                dtype=dtype,
            )
            code_targets = None
            code_target_values = None

        model.train()
        for _ in range(train_steps):
            (
                reconstruction,
                reconstruction_loss,
                vq_loss,
                prototype_loss,
                active_codes,
            ) = _weighted_vq_objective(
                model, training_data, weights, code_targets
            )
            loss = (
                reconstruction_loss
                + vae_weight * vq_loss
                + (code_target_weight if elite_guidance else 0.0) * prototype_loss
            )
            if effective_blackbox_weight:
                loss = loss + effective_blackbox_weight * _blackbox_surrogate_loss(
                    fitness_function,
                    reconstruction,
                    blackbox_step,
                    maximize,
                    blackbox_max_directions,
                    -global_radius,
                    global_radius,
                    weights=weights,
                )
            if div_weight and not single_peak:
                loss = loss + div_weight * diversity_loss(reconstruction)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if elite_guidance:
            code_probabilities = _code_sampling_probabilities(
                code_target_values,
                maximize,
                code_temperature,
                code_uniform_mix,
            )
        loss_history.append(float(loss.detach().cpu()))

        if (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            best = archive_values[0]
            progress_bar.set_description(
                f"Elite+ZO VQ-VAE [{epoch + 1}/{epochs}] "
                f"objective={best.item():.6f} "
                f"active={active_codes}/{model.num_embeddings}"
            )

    model.elite_archive = archive.detach()
    model.elite_archive_values = archive_values.detach()
    model.best_solution = archive[0].detach()
    model.best_objective = archive_values[0].detach()
    model.code_sampling_probabilities = code_probabilities.detach()
    print("Elite+ZO VQ-VAE training completed.")
    return model, loss_history


# ---------------------------------------------------------------------------
# VQ-VAE independent quality and routing calibration
# ---------------------------------------------------------------------------


def _calibrate_decoder_and_codebook(model, target_pool, steps, learning_rate):
    """Make the routed anchor codes decode precisely to elite archive points."""
    if steps <= 0:
        return None

    route_count = len(target_pool)
    parameters = list(model.decoder.parameters()) + [model.vq.embeddings.weight]
    optimizer = optim.Adam(parameters, lr=learning_rate)
    code_targets = target_pool.detach()

    model.train()
    last_loss = None
    for _ in range(steps):
        decoded_codes = model.output_activation(
            model.decoder(model.vq.embeddings.weight[:route_count])
        )
        last_loss = F.mse_loss(decoded_codes, code_targets)
        optimizer.zero_grad()
        last_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return float(last_loss.detach().cpu())


def _calibrate_encoder_routing(
    model,
    input_dim,
    route_count,
    steps,
    batch_size,
    learning_rate,
):
    """Route Gaussian inputs uniformly over high-quality anchor codes."""
    if steps <= 0:
        return None

    device = model.vq.embeddings.weight.device
    dtype = model.vq.embeddings.weight.dtype
    parameters = list(model.encoder.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    all_embeddings = model.vq.embeddings.weight.detach()
    anchor_embeddings = all_embeddings[:route_count]
    sqrt_two = math.sqrt(2.0)

    model.train()
    last_loss = None
    for _ in range(steps):
        routing_inputs = torch.randn(
            batch_size, input_dim, device=device, dtype=dtype
        )
        uniform_coordinate = 0.5 * (
            1.0 + torch.erf(routing_inputs[:, 0] / sqrt_two)
        )
        target_indices = torch.clamp(
            (uniform_coordinate * route_count).long(), max=route_count - 1
        )
        encoded = model.encoder(routing_inputs)
        distances = (
            encoded.pow(2).sum(dim=1, keepdim=True)
            + all_embeddings.pow(2).sum(dim=1)
            - 2.0 * encoded @ all_embeddings.t()
        )
        classification_loss = F.cross_entropy(-distances / 0.1, target_indices)
        target_embeddings = anchor_embeddings[target_indices]
        last_loss = classification_loss + 0.05 * F.mse_loss(
            encoded, target_embeddings
        )
        optimizer.zero_grad()
        last_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return float(last_loss.detach().cpu())


def train_VQVAE(*args, **kwargs):
    """Train VQ-VAE and independently polish code quality and routing.

    Optional tuning keywords are ``routing_codes``,
    ``code_calibration_steps``, ``routing_calibration_steps``,
    ``calibration_batch_size``, and ``calibration_learning_rate``.
    """
    routing_codes = kwargs.pop("routing_codes", 6)
    code_steps = kwargs.pop("code_calibration_steps", 100)
    routing_steps = kwargs.pop("routing_calibration_steps", 100)
    calibration_batch_size = kwargs.pop("calibration_batch_size", 2048)
    calibration_learning_rate = kwargs.pop("calibration_learning_rate", None)

    bound = inspect.signature(_train_vqvae_elite).bind_partial(*args, **kwargs)
    model, loss_history = _train_vqvae_elite(*args, **kwargs)
    input_dim = bound.arguments.get("input_dim", getattr(model, "input_dim", None))
    if input_dim is None:
        raise ValueError("input_dim is required for VQ-VAE routing calibration")

    main_learning_rate = bound.arguments.get("learning_rate", 3e-4)
    if calibration_learning_rate is None:
        calibration_learning_rate = max(float(main_learning_rate), 1e-3)

    elite_guidance = bound.arguments.get("elite_guidance", True)
    route_count = min(_routing_code_count(model, routing_codes), model.num_embeddings)
    if elite_guidance:
        route_count = min(route_count, len(model.elite_archive))
        target_pool = model.elite_archive[:route_count].detach()
        code_loss = _calibrate_decoder_and_codebook(
            model,
            target_pool,
            int(code_steps),
            float(calibration_learning_rate),
        )
        routing_loss = _calibrate_encoder_routing(
            model,
            int(input_dim),
            route_count,
            int(routing_steps),
            int(calibration_batch_size),
            float(calibration_learning_rate),
        )
    else:
        code_loss = None
        routing_loss = None

    model.vq_routing_codes = route_count
    model.vq_code_calibration_loss = code_loss
    model.vq_routing_calibration_loss = routing_loss
    model.eval()
    return model, loss_history

# ---------------------------------------------------------------------------
# Multi-head VQ-VAE three-stage training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mhvq_decoded_head_codes(model):
    """Return all decoder/code combinations as [heads, codes, dimensions]."""
    embeddings = model.vq.embeddings.weight
    return torch.stack([decoder(embeddings) for decoder in model.decoders], dim=0)


@torch.no_grad()
def _sample_mhvqvae_population(
    model,
    count,
    input_dim,
    global_radius,
    device,
    dtype,
    epoch,
    exploration_fraction,
    prototype_probabilities,
    mutation_sigma,
):
    """Sample every head/code prototype, mutations, and uniform exploration."""
    model_count, exploration_count = _population_split(
        count, exploration_fraction, epoch
    )
    populations = []
    prototypes = _mhvq_decoded_head_codes(model).reshape(-1, input_dim)
    prototype_count = min(model_count, len(prototypes))

    if prototype_count:
        if prototype_count == len(prototypes):
            populations.append(prototypes)
        else:
            indices = torch.multinomial(
                prototype_probabilities, prototype_count, replacement=False
            )
            populations.append(prototypes[indices])

    mutation_count = model_count - prototype_count
    if mutation_count:
        indices = torch.multinomial(
            prototype_probabilities, mutation_count, replacement=True
        )
        mutated = prototypes[indices] + mutation_sigma * torch.randn_like(
            prototypes[indices]
        )
        populations.append(mutated.clamp(-global_radius, global_radius))

    if exploration_count:
        uniform = (
            torch.rand(exploration_count, input_dim, device=device, dtype=dtype)
            * 2.0
            - 1.0
        ) * global_radius
        populations.append(uniform)

    return torch.cat(populations, dim=0).detach()


@torch.no_grad()
def _select_mhvqvae_head_targets(
    archive,
    archive_values,
    n_heads,
    minimum_distance,
):
    """Choose the best archive points separated into different solution modes."""
    if len(archive) < n_heads:
        raise ValueError("The elite archive must contain at least one point per head")

    selected = [0]
    selected_mask = torch.zeros(len(archive), device=archive.device, dtype=torch.bool)
    selected_mask[0] = True

    # The archive is already sorted by objective quality. Select the first point
    # that is sufficiently far from all earlier modes, preserving quality first.
    while len(selected) < n_heads:
        chosen = None
        anchors = archive[torch.as_tensor(selected, device=archive.device)]
        distances = torch.cdist(archive, anchors).amin(dim=1)
        for candidate_index in range(len(archive)):
            if selected_mask[candidate_index]:
                continue
            if distances[candidate_index] >= minimum_distance:
                chosen = candidate_index
                break

        # If the requested separation is impossible, use the farthest remaining
        # point. This keeps the function usable for tightly packed optima.
        if chosen is None:
            distances = distances.masked_fill(selected_mask, -torch.inf)
            chosen = int(torch.argmax(distances))

        selected.append(chosen)
        selected_mask[chosen] = True

    indices = torch.as_tensor(selected, device=archive.device, dtype=torch.long)
    return archive[indices].detach(), archive_values[indices].detach()


@torch.no_grad()
def _align_mhvqvae_targets(targets, target_values, previous_targets):
    """Keep head identities stable when archive targets change between epochs."""
    if previous_targets is None:
        return targets, target_values

    distances = torch.cdist(previous_targets, targets)
    available = torch.ones(len(targets), device=targets.device, dtype=torch.bool)
    assignment = []
    for head_index in range(len(previous_targets)):
        row = distances[head_index].masked_fill(~available, torch.inf)
        target_index = int(torch.argmin(row))
        assignment.append(target_index)
        available[target_index] = False
    indices = torch.as_tensor(assignment, device=targets.device, dtype=torch.long)
    return targets[indices].detach(), target_values[indices].detach()


def _weighted_mhvqvae_objective(model, training_data, weights, head_targets=None):
    """Fit elites with best-head reconstruction and explicit head mode targets."""
    encoded = model.encoder(training_data)
    embeddings = model.vq.embeddings.weight
    distances = (
        encoded.pow(2).sum(dim=1, keepdim=True)
        + embeddings.pow(2).sum(dim=1)
        - 2.0 * encoded @ embeddings.t()
    )
    encoding_indices = torch.argmin(distances, dim=1)
    quantized_raw = model.vq.embeddings(encoding_indices)

    codebook_per_sample = (quantized_raw - encoded.detach()).pow(2).mean(dim=1)
    commitment_per_sample = (quantized_raw.detach() - encoded).pow(2).mean(dim=1)
    vq_per_sample = (
        codebook_per_sample
        + model.vq.commitment_cost * commitment_per_sample
    )
    normalized_weights = weights.detach() / weights.detach().sum().clamp_min(1e-12)
    vq_loss = torch.sum(normalized_weights * vq_per_sample)

    quantized = encoded + (quantized_raw - encoded).detach()
    reconstructions = torch.stack(
        [decoder(quantized) for decoder in model.decoders], dim=0
    )
    per_head_error = (reconstructions - training_data.unsqueeze(0)).pow(2)
    per_head_error = per_head_error.reshape(
        model.n_heads, len(training_data), -1
    ).mean(dim=2)
    best_head_error = per_head_error.amin(dim=0)
    reconstruction_loss = torch.sum(normalized_weights * best_head_error)

    decoded_codes = torch.stack(
        [decoder(embeddings) for decoder in model.decoders], dim=0
    )
    if head_targets is None:
        head_target_loss = decoded_codes.sum() * 0.0
    else:
        prototype_targets = head_targets[:, None, :].expand_as(decoded_codes)
        head_target_loss = F.mse_loss(decoded_codes, prototype_targets.detach())
    active_codes = torch.unique(encoding_indices).numel()
    return (
        reconstructions,
        reconstruction_loss,
        vq_loss,
        head_target_loss,
        active_codes,
    )


def _calibrate_mhvqvae_codes(
    model,
    head_targets,
    route_count,
    steps,
    learning_rate,
):
    """Make every routed code from head h decode to the h-th solution mode."""
    if steps <= 0:
        return None

    parameters = list(model.decoders.parameters()) + [model.vq.embeddings.weight]
    optimizer = optim.Adam(parameters, lr=learning_rate)
    progress = tqdm(range(steps), desc="MHVQVAE code calibration", leave=False)
    last_loss = None
    for _ in progress:
        routed_embeddings = model.vq.embeddings.weight[:route_count]
        decoded = torch.stack(
            [decoder(routed_embeddings) for decoder in model.decoders], dim=0
        )
        targets = head_targets[:, None, :].expand_as(decoded).detach()
        last_loss = F.mse_loss(decoded, targets)
        optimizer.zero_grad()
        last_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

    return float(last_loss.detach().cpu())

def train_MHVQVAE(
    model,
    device,
    input_dim,
    batch_size,
    fitness_function=fitness_function,
    optimization="Adam",
    objective="minimize",
    vae_weight=0.1,
    div_head_weight=0.0,
    dive_intra_weight=0.0,
    global_radius=5.0,
    epochs=1000,
    log_interval=100,
    learning_rate=3e-4,
    elite_fraction=0.2,
    rank_temperature=0.2,
    elite_fraction_end=0.05,
    rank_temperature_end=0.05,
    exploration_fraction=0.2,
    train_steps=1,
    archive_size=4096,
    archive_grid_size=0.1,
    head_min_distance=2.0,
    single_peak=False,
    mutation_sigma_start=0.5,
    mutation_sigma_end=0.02,
    code_temperature=0.3,
    code_uniform_mix=0.05,
    head_target_weight=1.0,
    routing_codes=6,
    code_calibration_steps=500,
    routing_calibration_steps=1000,
    calibration_batch_size=2048,
    calibration_learning_rate=None,
    blackbox_weight=0.05,
    blackbox_step=1e-3,
    blackbox_max_directions=None,
    blackbox_start_fraction=0.8,
    elite_guidance=True,
):
    """Train MHVQVAE with search, late zero-order refinement, and calibration."""
    del div_head_weight, dive_intra_weight
    required = ("vq", "encoder", "decoders", "n_heads")
    if not all(hasattr(model, name) for name in required):
        raise TypeError(
            "multi-peak train expects model.VAEs.MHVQVAE. "
            f"Received {type(model).__name__}."
        )
    maximize = {"minimize": False, "maximize": True}.get(objective)
    if maximize is None:
        raise ValueError(f"Unsupported objective direction: {objective}")
    if train_steps < 1:
        raise ValueError("train_steps must be positive")
    if archive_size < model.n_heads:
        raise ValueError("archive_size must be at least n_heads")
    if head_min_distance < 0:
        raise ValueError("head_min_distance cannot be negative")
    if mutation_sigma_start <= 0 or mutation_sigma_end <= 0:
        raise ValueError("mutation sigmas must be positive")
    blackbox_max_directions = _resolve_blackbox_directions(
        input_dim, blackbox_max_directions
    )

    model = model.to(device)
    optimizer = _optimizer(model, optimization, learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dtype = next(model.parameters()).dtype
    prototype_count = model.n_heads * model.vq.num_embeddings
    prototype_probabilities = torch.full(
        (prototype_count,), 1.0 / prototype_count, device=device, dtype=dtype
    )
    archive = None
    archive_values = None
    previous_targets = None
    loss_history = []
    progress_bar = tqdm(
        range(epochs), desc="Elite+ZO MHVQVAE search", total=epochs
    )

    for epoch in progress_bar:
        progress_fraction = epoch / max(epochs - 1, 1)
        effective_elite_fraction = _linear_schedule(
            elite_fraction, elite_fraction_end, progress_fraction
        )
        effective_rank_temperature = _linear_schedule(
            rank_temperature, rank_temperature_end, progress_fraction
        )
        effective_blackbox_weight = _scheduled_blackbox_weight(
            epoch, epochs, blackbox_weight, blackbox_start_fraction
        )
        mutation_sigma = mutation_sigma_start * (
            mutation_sigma_end / mutation_sigma_start
        ) ** progress_fraction

        model.eval()
        candidates = _sample_mhvqvae_population(
            model,
            batch_size,
            input_dim,
            global_radius,
            device,
            dtype,
            epoch,
            exploration_fraction,
            prototype_probabilities,
            mutation_sigma,
        )
        values = evaluate_objective(fitness_function, candidates)
        elite_x, _, _, _ = select_ranked_elites(
            candidates,
            values,
            effective_elite_fraction,
            effective_rank_temperature,
            maximize=maximize,
        )
        elite_values = evaluate_objective(fitness_function, elite_x)

        if archive is None:
            combined_x, combined_values = elite_x, elite_values
        else:
            combined_x = torch.cat((archive, elite_x), dim=0)
            combined_values = torch.cat((archive_values, elite_values), dim=0)
        archive, archive_values = _select_diverse_archive(
            combined_x,
            combined_values,
            archive_size,
            0.0 if single_peak else archive_grid_size,
            maximize,
        )
        if elite_guidance:
            head_targets, head_values = _select_mhvqvae_head_targets(
                archive,
                archive_values,
                model.n_heads,
                0.0 if single_peak else head_min_distance,
            )
            head_targets, head_values = _align_mhvqvae_targets(
                head_targets, head_values, previous_targets
            )
            previous_targets = head_targets
            training_count = min(
                len(archive), max(len(elite_x), 2 * prototype_count)
            )
            training_data = archive[:training_count]
            weights = rank_weights(
                training_count, effective_rank_temperature, device, dtype
            )
        else:
            head_targets = None
            head_values = archive_values[: model.n_heads]
            training_data = candidates
            weights = torch.full(
                (len(training_data),),
                1.0 / len(training_data),
                device=device,
                dtype=dtype,
            )

        model.train()
        for _ in range(train_steps):
            (
                reconstructions,
                reconstruction_loss,
                vq_loss,
                head_target_loss,
                active_codes,
            ) = _weighted_mhvqvae_objective(
                model, training_data, weights, head_targets
            )
            loss = (
                reconstruction_loss
                + vae_weight * vq_loss
                + (head_target_weight if elite_guidance else 0.0) * head_target_loss
            )
            if effective_blackbox_weight:
                # Apply the same rank-weighted local refinement to every
                # head; the existing head targets preserve distinct modes.
                multihead_weights = weights.repeat(model.n_heads)
                loss = loss + effective_blackbox_weight * _blackbox_surrogate_loss(
                    fitness_function,
                    reconstructions.reshape(-1, input_dim),
                    blackbox_step,
                    maximize,
                    blackbox_max_directions,
                    -global_radius,
                    global_radius,
                    weights=multihead_weights,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach().cpu()))

        if elite_guidance:
            with torch.no_grad():
                prototype_values = evaluate_objective(
                    fitness_function, _mhvq_decoded_head_codes(model).reshape(-1, input_dim)
                )
                prototype_probabilities = _code_sampling_probabilities(
                    prototype_values,
                    maximize,
                    code_temperature,
                    code_uniform_mix,
                )

        if (epoch + 1) % log_interval == 0 or epoch + 1 == epochs:
            values_text = ",".join(f"{value.item():.4f}" for value in head_values)
            progress_bar.set_description(
                f"Elite+ZO MHVQVAE [{epoch + 1}/{epochs}] "
                f"objectives=[{values_text}] active={active_codes}/{model.vq.num_embeddings}"
            )

    route_count = min(int(routing_codes), model.vq.num_embeddings)
    calibration_lr = (
        max(float(learning_rate), 1e-3)
        if calibration_learning_rate is None
        else float(calibration_learning_rate)
    )
    if elite_guidance:
        code_loss = _calibrate_mhvqvae_codes(
            model,
            head_targets,
            route_count,
            int(code_calibration_steps),
            calibration_lr,
        )
        routing_loss = _calibrate_encoder_routing(
            model,
            input_dim,
            route_count,
            int(routing_calibration_steps),
            int(calibration_batch_size),
            calibration_lr,
        )
    else:
        head_targets, head_values = _select_mhvqvae_head_targets(
            archive,
            archive_values,
            model.n_heads,
            0.0 if single_peak else head_min_distance,
        )
        code_loss = None
        routing_loss = None

    model.elite_archive = archive.detach()
    model.elite_archive_values = archive_values.detach()
    model.head_targets = head_targets.detach()
    model.head_target_values = head_values.detach()
    model.mhvq_routing_codes = route_count
    model.mhvq_code_calibration_loss = code_loss
    model.mhvq_routing_calibration_loss = routing_loss
    model.eval()
    print("Three-stage Elite+ZO MHVQVAE training completed.")
    return model, loss_history


def train_multiobjective(*args, **kwargs):
    """Train a preference-conditioned VAE for two or more objectives.

    The implementation is loaded lazily because it reuses optimizer and
    population helpers from this module.
    """
    from scripts.train_multiobjective import train_multiobjective as _train

    return _train(*args, **kwargs)


__all__ = [
    "train_IVAE",
    "train_VQVAE",
    "train_MHVQVAE",
    "train_multiobjective",
    "diversity_loss",
]






