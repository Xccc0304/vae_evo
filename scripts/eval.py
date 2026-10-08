import torch
import numpy as np
import os




# set fitness target and distance scale to unify the scale and slope of the fitness
fitness_target = {
    # --- 最大化: (-4,4)^D 上的最大值 ---
    "ackley": 12.5401,  # 与维度无关
    "rastrigin": 64.6249, # x_i = 3.51786 max=0 for all dimensions
    "rastrigin_4d": 129.2498,
    "rastrigin_32d": 1033.9980,
    "rastrigin_256d": 8271.9844,
    # --- 最小化: 2D ---
    "rosenbrock": 0,
    "beale": 0,
    "himmelblau": 0,
    # --- 最小化: 高维 10 / 30 / 100 ---
    "griewank_10d": 0,
    "griewank_30d": 0,
    "griewank_100d": 0,
    "weierstrass_10d": 0,
    "weierstrass_30d": 0,
    "weierstrass_100d": 0,
    "levy_10d": 0,
    "levy_30d": 0,
    "levy_100d": 0,
    "rosenbrock_10d": 0,
    "rosenbrock_30d": 0,
    "rosenbrock_100d": 0,
}


distance_scale = {
    # --- 最大化 ---
    "ackley": 2,
    "rastrigin": 30,
    "rastrigin_4d": 60,
    "rastrigin_32d": 500,
    "rastrigin_256d": 4000,
    # --- 最小化: 2D ---
    "rosenbrock": 287.51,
    "beale": 20,
    "himmelblau": 17.01,
    # --- 最小化: 高维 10 / 30 / 100(中位数默认值,可调) ---
    "griewank_10d": 1.0,
    "griewank_30d": 1.0,
    "griewank_100d": 1.1,
    "weierstrass_10d": 20,
    "weierstrass_30d": 60,
    "weierstrass_100d": 200,
    "levy_10d": 22,
    "levy_30d": 68,
    "levy_100d": 230,
    "rosenbrock_10d": 50000,
    "rosenbrock_30d": 160000,
    "rosenbrock_100d": 560000,
}


def _model_device_and_dtype(model, device=None):
    """Return the model parameter device/dtype, moving it when requested."""
    try:
        reference = next(model.parameters())
    except StopIteration as exc:
        raise TypeError("The evaluated model must contain trainable parameters") from exc
    if device is not None:
        requested_device = torch.device(device)
        if reference.device != requested_device:
            model.to(requested_device)
            reference = next(model.parameters())
    return reference.device, reference.dtype


def _sample_code_indices(model, count, route_attribute, device):
    """Sample a uniform categorical prior over calibrated VQ codes."""
    total_codes = int(model.vq.num_embeddings)
    route_count = min(
        total_codes,
        max(1, int(getattr(model, route_attribute, total_codes))),
    )
    return torch.randint(route_count, (count,), device=device)


def _sample_preferences(count, objective_count, device, dtype):
    """Sample approximately uniform preferences on the unit simplex."""
    exponential = -torch.log(
        torch.rand(count, objective_count, device=device, dtype=dtype).clamp_min(1e-7)
    )
    return exponential / exponential.sum(dim=1, keepdim=True)


@torch.no_grad()
def generate_decoder_samples(
    model,
    sample_count=10000,
    device=None,
    preferences=None,
):
    """Generate solutions directly from the latent prior and decoder.

    The encoder and reconstruction ``forward`` path are deliberately bypassed.
    VQ models sample discrete codebook entries; MHVQVAE returns one population
    per decoder head; preference-conditioned VAEs sample simplex preferences.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")

    device, dtype = _model_device_and_dtype(model, device)
    was_training = model.training
    generation_mode = getattr(model, "generation_mode", None)
    model.eval()
    try:
        if hasattr(model, "vq") and hasattr(model, "decoders"):
            indices = _sample_code_indices(
                model, sample_count, "mhvq_routing_codes", device
            )
            quantized = model.vq.embeddings(indices)
            return [decoder(quantized) for decoder in model.decoders]

        if hasattr(model, "vq") and hasattr(model, "decoder"):
            indices = _sample_code_indices(
                model, sample_count, "vq_routing_codes", device
            )
            quantized = model.vq.embeddings(indices)
            generated = model.decoder(quantized)
            activation = getattr(model, "output_activation", None)
            return activation(generated) if activation is not None else generated

        if hasattr(model, "decode") and hasattr(model, "vae_latent_dim"):
            # Never evaluate archive interpolation as neural generation.
            if generation_mode is not None:
                model.generation_mode = "decoder"
            if preferences is None:
                preferences = _sample_preferences(
                    sample_count,
                    int(model.preference_dim),
                    device,
                    dtype,
                )
            else:
                preferences = torch.as_tensor(
                    preferences, device=device, dtype=dtype
                )
                if preferences.shape != (sample_count, int(model.preference_dim)):
                    raise ValueError(
                        "preferences must have shape "
                        f"[{sample_count}, {model.preference_dim}]"
                    )
            latent = torch.randn(
                sample_count,
                int(model.vae_latent_dim),
                device=device,
                dtype=dtype,
            )
            return model.decode(latent, preferences)

        if hasattr(model, "decoder") and hasattr(model, "latent_dim"):
            latent = torch.randn(
                sample_count,
                int(model.latent_dim),
                device=device,
                dtype=dtype,
            )
            generated = model.decoder(latent)
            activation = getattr(model, "output_activation", None)
            return activation(generated) if activation is not None else generated

        raise TypeError(
            "Unsupported model: expected IVAE, VQVAE, MHVQVAE, or PVAE decoder API"
        )
    finally:
        if generation_mode is not None:
            model.generation_mode = generation_mode
        model.train(was_training)


def evaluate(
    model,
    device,
    input_dim,
    obj,
    obj_name: str,
    k=64,
    maximize=False,
    sample_count=10000,
):
    """Evaluate a single-decoder model by independent decoder generation."""
    x_generated = generate_decoder_samples(
        model, sample_count=sample_count, device=device
    )
    if isinstance(x_generated, list):
        raise TypeError("Use evaluate_muti_peak for a multi-head decoder model")
    if x_generated.ndim != 2 or x_generated.shape[1] != input_dim:
        raise ValueError(
            f"Decoder output must have shape [N, {input_dim}], "
            f"got {tuple(x_generated.shape)}"
        )

    pop = x_generated
    fitness = obj(x_generated)
    top_count = min(int(k), len(fitness))
    if top_count < 1:
        raise ValueError("k must be positive")
    top_fitness, indices = torch.topk(fitness, top_count, largest=maximize)
    top_pop = pop[indices]
    entropy_all = point_entropy(pop, fitness)
    entropy_top_k = point_entropy(top_pop, top_fitness)

    target = fitness_target[obj_name]
    scale = distance_scale[obj_name]
    map_fun = energy_wrapper(obj, target=target, scale=scale)
    fitness_reward_all = map_fun(pop).mean()
    fitness_reward_top_k = map_fun(top_pop).mean()

    return {
        "entropy": {
            "all": entropy_all,
            "top_k": entropy_top_k,
        },
        "fitness_reward": {
            "all": fitness_reward_all,
            "top_k": fitness_reward_top_k,
        },
        "average_fitness": {
            "all": fitness.mean().item(),
            "top_k": top_fitness.mean().item(),
        }
    }

def select_from_each_population(x_generated, obj, k, maximize=False):
    """
    Select k//n of the best individuals from each subpopulation

    Args:
        x_generated: list of tensors, each shape [N_i, D]
        obj: objective function (fitness function)
        k: total number of individuals to select
        maximize: True if higher fitness is better

    Returns:
        best_pop: selected individuals, shape [k, D]
        best_fitness: their fitness values, shape [k]
    """
    n_pops = len(x_generated)
    if int(k) < 1:
        raise ValueError("k must be positive")
    if n_pops == 0:
        raise ValueError("x_generated is empty")


    per_pop = k // n_pops
    remainder = k % n_pops

    selected_pops = []
    selected_fitness = []

    for i, pop in enumerate(x_generated):
        fitness = obj(pop)

        num_select = per_pop + (1 if i < remainder else 0)

        num_select = min(num_select, len(pop))

        if num_select > 0:
            best_vals, indices = torch.topk(fitness, num_select, largest=maximize)
            selected_pops.append(pop[indices])
            selected_fitness.append(best_vals)


    if selected_pops:
        best_pop = torch.cat(selected_pops, dim=0)
        best_fitness = torch.cat(selected_fitness, dim=0)
    else:
        best_pop = x_generated[0].new_empty((0, x_generated[0].shape[-1]))
        best_fitness = x_generated[0].new_empty((0,))

    return best_pop, best_fitness










def evaluate_muti_peak(
    model,
    input_dim,
    obj,
    obj_name: str,
    k=64,
    maximize=False,
    device=None,
    sample_count=10000,
):
    """Evaluate every MHVQVAE decoder head from independent VQ latents."""
    x_generated = generate_decoder_samples(
        model, sample_count=sample_count, device=device
    )
    if not isinstance(x_generated, list):
        raise TypeError("evaluate_muti_peak requires a multi-head decoder model")
    if any(pop.ndim != 2 or pop.shape[1] != input_dim for pop in x_generated):
        shapes = [tuple(pop.shape) for pop in x_generated]
        raise ValueError(
            f"Every decoder head must output shape [N, {input_dim}], got {shapes}"
        )

    top_pop, top_fitness = select_from_each_population(
        x_generated, obj, k=k, maximize=maximize
    )
    pop = torch.cat(x_generated, dim=0)
    fitness = obj(pop)

    entropy_all = point_entropy(pop, fitness)
    entropy_top_k = point_entropy(top_pop, top_fitness)

    target = fitness_target[obj_name]
    scale = distance_scale[obj_name]
    map_fun = energy_wrapper(obj, target=target, scale=scale)
    fitness_reward_all = map_fun(pop).mean()
    fitness_reward_top_k = map_fun(top_pop).mean()

    return {
        "entropy": {
            "all": entropy_all,
            "top_k": entropy_top_k,
        },
        "fitness_reward": {
            "all": fitness_reward_all,
            "top_k": fitness_reward_top_k,
        },
        "average_fitness": {
            "all": fitness.mean().item(),
            "top_k": top_fitness.mean().item(),
        }
    }

## Multi-objective optimization evaluation function



from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.spacing import SpacingIndicator


def _objective_matrix(objective_function, decisions):
    values = objective_function(decisions)
    if isinstance(values, (tuple, list)):
        values = torch.stack(
            [
                torch.as_tensor(
                    value, device=decisions.device, dtype=decisions.dtype
                ).reshape(-1)
                for value in values
            ],
            dim=1,
        )
    else:
        values = torch.as_tensor(
            values, device=decisions.device, dtype=decisions.dtype
        )
        if values.ndim == 1:
            values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(decisions):
        raise ValueError(
            "objective_function must return one objective vector per solution"
        )
    return values


def evaluate_multi_objective(
    model,
    true_pf,
    objective_function,
    device=None,
    sample_count=10000,
    preferences=None,
    scale=True,
):
    """Independently decode a Pareto population, then compute GD/IGD/Spacing."""
    generated = generate_decoder_samples(
        model,
        sample_count=sample_count,
        device=device,
        preferences=preferences,
    )
    if isinstance(generated, list):
        raise TypeError("Multi-objective evaluation expects one decoder population")
    evaluated_decisions = (generated + 1.0) / 2.0 if scale else generated
    approximate_pf = _objective_matrix(objective_function, evaluated_decisions)

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)

    true_pf = to_numpy(true_pf)
    approximate_pf = to_numpy(approximate_pf)

    gd_value = GD(true_pf).do(approximate_pf)
    igd_value = IGD(true_pf).do(approximate_pf)
    spacing_value = SpacingIndicator().do(approximate_pf)

    return {
        "GD": gd_value,
        "IGD": igd_value,
        "SP": spacing_value,
    }

## entropy


def get_top_values(values, pop, n):

    assert values.ndim == 1, "The values must be one-dimensional (N, )"
    assert values.size(0) == pop.size(0), "The first dimension sizes of values and items must be the same."

    #
    _, indices = torch.topk(values, k=n, largest=True, sorted=False)

    #
    return pop[indices]


def prob(x, scale=10):
    classification = torch.round(x * scale).long()
    # count the number of points in each class, return [class, num]
    classes, num = torch.unique(classification, return_counts=True, dim=0)
    prob = num.float() / num.sum()
    return prob


def entropy(x, scale=10):
    p = prob(x, scale)
    return torch.sum(-p * torch.log2(p))


def point_entropy(x, fitness, n=None, scale=10):
    """
    Calculate the distribution entropy of the given population

    Args:
        x: torch.Tensor, shape (N, D) 鈥?The last generation's solution
        fitness: torch.Tensor, shape (N) 鈥?The fitness value corresponding to the last generation's solution
        n: int, optional 鈥?top-n fitness
        scale: int 鈥?Grid refinement degree

    Returns:
        float 鈥?entropy
    """
    # Only retain the top-n optimal solutions
    if n is not None:
        x = get_top_values(fitness, x, n)


    return entropy(x, scale).item()



# reward


def top_reward(fitnesses, n=None):

    if n is not None:
        last_gen = fitnesses[-1]

        top_n = torch.topk(last_gen, n).values

        return top_n.mean().item()

    else:
        return fitnesses.mean().item()





### [0-1] reward

def energy_wrapper(
    obj,
    target=0.0,
    scale=1.0,
    epsilon=1e-8
):
    """
    energy = 1 / (1 + |obj(x) - target| / scale)
    range: (0, 1]
    """
    def wrapped_obj(x):
        distance = torch.abs(obj(x) - target) / scale
        energy = 1.0 / (1.0 + distance + epsilon)
        return energy

    return wrapped_obj


def energy_wrapper_np(obj, target=0.0, scale=1.0, epsilon=1e-8):


    def wrapped_obj(x):

        obj_vals = obj(x)

        distance = np.abs(obj_vals - target) / scale

        energy = 1.0 / (1.0 + distance + epsilon)

        return energy

    return wrapped_obj



def set_seed(seed=42):
    """ set_seed """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
