import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def _next_power_of_two(value):
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def adaptive_capacity(input_dim):
    """Return the shared dimension-dependent capacity used by VAE variants."""
    input_dim = int(input_dim)
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    return {
        "hidden_dim": min(256, max(64, (input_dim + 1) // 2)),
        "latent_dim": min(64, max(2, (input_dim + 1) // 2)),
        "num_embeddings": min(256, max(16, _next_power_of_two(input_dim))),
        "embedding_dim": min(
            128,
            max(8, _next_power_of_two((input_dim + 1) // 2)),
        ),
    }





def adaptive_vae_capacity(input_dim):
    """Return the validated dimension-specific capacity for IVAE."""
    input_dim = int(input_dim)
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if input_dim < 4:
        return {"hidden_dim": 96, "latent_dim": 2}
    if input_dim < 32:
        return {"hidden_dim": 64, "latent_dim": 2}
    if input_dim < 256:
        return {"hidden_dim": 32, "latent_dim": 16}
    return {"hidden_dim": 96, "latent_dim": 16}



class ScaledTanh(nn.Module):
    def __init__(self, scale=4.0, alpha=2.0):
        """
        :param scale: Output range scaling factor (default 4.0)
        :param alpha: Enter the scaling factor to control the steepness of the nonlinear region (default 2.0)
        """
        super().__init__()
        self.scale = scale
        self.alpha = alpha

        
    def forward(self, x):
        return self.scale * torch.tanh(x / self.alpha)



# VAE 
class IVAE(nn.Module):
    def __init__(
        self,
        input_dim=1,
        latent_dim=None,
        use_tanh=False,
        tanh_scale=4.0,
        tanh_alpha=2.0,
        hidden_dim=None,
    ):
        super().__init__()
        capacity = adaptive_vae_capacity(input_dim)
        self.input_dim = int(input_dim)
        self.latent_dim = int(
            capacity["latent_dim"] if latent_dim is None else latent_dim
        )
        self.hidden_dim = int(
            capacity["hidden_dim"] if hidden_dim is None else hidden_dim
        )
        if self.latent_dim < 1 or self.hidden_dim < 2:
            raise ValueError("latent_dim must be positive and hidden_dim >= 2")
        bottleneck_dim = max(1, self.hidden_dim // 2)
        self.use_tanh = use_tanh

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.input_dim),
        )
        if self.use_tanh:
            self.output_activation = ScaledTanh(
                scale=tanh_scale, alpha=tanh_alpha
            )
        else:
            self.output_activation = nn.Identity()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.output_activation(self.decoder(z))
        return reconstruction, mu, logvar

# VQ-VAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=8, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1, 1)

    def forward(self, z):
   
        flat_z = z.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.embeddings.weight.t()))
        
       
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape)
        
       
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        
        quantized = z + (quantized - z).detach()
        return quantized, loss


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim=1,
        num_embeddings=None,
        embedding_dim=None,
        use_tanh=False,
        tanh_scale=4.0,
        tanh_alpha=2.0,
        hidden_dim=None,
    ):
        super().__init__()
        capacity = adaptive_capacity(input_dim)
        self.input_dim = int(input_dim)
        self.num_embeddings = int(
            capacity["num_embeddings"]
            if num_embeddings is None
            else num_embeddings
        )
        self.embedding_dim = int(
            capacity["embedding_dim"]
            if embedding_dim is None
            else embedding_dim
        )
        self.hidden_dim = int(
            capacity["hidden_dim"] if hidden_dim is None else hidden_dim
        )
        if min(self.num_embeddings, self.embedding_dim, self.hidden_dim) < 1:
            raise ValueError("VQ-VAE capacity values must be positive")
        decoder_hidden_dim = max(1, self.hidden_dim // 2)
        self.use_tanh = use_tanh

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, self.input_dim),
        )
        if self.use_tanh:
            self.output_activation = ScaledTanh(
                scale=tanh_scale, alpha=tanh_alpha
            )
        else:
            self.output_activation = nn.Identity()

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss = self.vq(encoded)
        reconstruction = self.output_activation(self.decoder(quantized))
        return reconstruction, vq_loss

# MUTI-Head_VQ-VAE
class MHVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=2,
        num_embeddings=None,
        embedding_dim=None,
        n_heads=2,
        use_tanh=False,
        tanh_scale=4.0,
        tanh_alpha=2.0,
        hidden_dim=None,
    ):
        super().__init__()
        capacity = adaptive_capacity(input_dim)
        self.input_dim = int(input_dim)
        self.n_heads = int(n_heads)
        self.num_embeddings = int(
            capacity["num_embeddings"]
            if num_embeddings is None
            else num_embeddings
        )
        self.embedding_dim = int(
            capacity["embedding_dim"]
            if embedding_dim is None
            else embedding_dim
        )
        self.hidden_dim = int(
            capacity["hidden_dim"] if hidden_dim is None else hidden_dim
        )
        if min(
            self.n_heads,
            self.num_embeddings,
            self.embedding_dim,
            self.hidden_dim,
        ) < 1:
            raise ValueError("MHVQVAE capacity values must be positive")
        decoder_hidden_dim = max(1, self.hidden_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embedding_dim, decoder_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(decoder_hidden_dim, self.input_dim),
                    *(
                        [ScaledTanh(scale=tanh_scale, alpha=tanh_alpha)]
                        if use_tanh
                        else []
                    ),
                )
                for _ in range(self.n_heads)
            ]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss = self.vq(encoded)
        reconstructions = [decoder(quantized) for decoder in self.decoders]
        return reconstructions, vq_loss

# Pareto Variational Autoencoder (PVAE)



class TanhToUnit(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return (torch.tanh(x) + 1) * 0.5



class PVAE(nn.Module):
    """Preference-conditioned VAE for Pareto-set generation.

    The historical constructor used ``input_dim`` for the generator noise
    dimension and ``latent_dim`` for the decision dimension. Those positional
    semantics are retained for notebook compatibility. ``vae_latent_dim`` may
    be supplied explicitly when a different stochastic latent size is wanted.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim=128,
        activation=None,
        preference_dim=2,
        vae_latent_dim=None,
    ):
        super().__init__()
        self.activation = activation
        self.noise_dim = int(input_dim)
        self.decision_dim = int(latent_dim)
        self.preference_dim = int(preference_dim)
        self.hidden_dim = int(hidden_dim)
        self.vae_latent_dim = (
            self.noise_dim if vae_latent_dim is None else int(vae_latent_dim)
        )
        if self.preference_dim < 2:
            raise ValueError("preference_dim must be at least two")

        self.encoder = nn.Sequential(
            nn.Linear(self.decision_dim + self.preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.vae_latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.vae_latent_dim + self.preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.decision_dim),
        )
        # Normal inference is always neural. The anchor path is retained only
        # as an explicit ablation/debugging mode and is never enabled by train().
        self.generation_mode = "decoder"
        self.pareto_router_enabled = False
        # This switch changes neural inputs, not search reference directions.
        self.preference_conditioning_enabled = True
        self.register_buffer(
            "pareto_fixed_mask",
            torch.zeros(self.decision_dim, dtype=torch.bool),
        )
        self.register_buffer(
            "pareto_fixed_values",
            torch.zeros(self.decision_dim),
        )
        self.register_buffer(
            "pareto_decision_weights",
            torch.ones(self.decision_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def _validate_preferences(self, preferences, batch_size, reference):
        if preferences is None:
            preferences = torch.full(
                (batch_size, self.preference_dim),
                1.0 / self.preference_dim,
                device=reference.device,
                dtype=reference.dtype,
            )
        if preferences.shape != (batch_size, self.preference_dim):
            raise ValueError(
                "preferences must have shape "
                f"[{batch_size}, {self.preference_dim}]"
            )
        preferences = preferences.to(device=reference.device, dtype=reference.dtype)
        if not getattr(self, "preference_conditioning_enabled", True):
            return torch.full_like(preferences, 1.0 / self.preference_dim)
        return preferences / preferences.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def encode(self, decisions, preferences):
        preferences = self._validate_preferences(
            preferences, len(decisions), decisions
        )
        statistics = self.encoder(torch.cat((decisions, preferences), dim=1))
        return statistics.chunk(2, dim=1)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _interpolate_pareto_anchors(self, preferences):
        anchor_preferences = self.pareto_anchor_preferences.to(
            device=preferences.device, dtype=preferences.dtype
        )
        anchor_targets = self.pareto_anchor_targets.to(
            device=preferences.device, dtype=preferences.dtype
        )
        if self.preference_dim == 2:
            coordinate = preferences[:, 1].contiguous()
            anchor_coordinate = anchor_preferences[:, 1]
            order = torch.argsort(anchor_coordinate)
            anchor_coordinate = anchor_coordinate[order]
            anchor_targets = anchor_targets[order]
            right = torch.searchsorted(anchor_coordinate, coordinate).clamp(
                max=len(anchor_coordinate) - 1
            )
            left = (right - 1).clamp(min=0)
            if getattr(self, "binary_decisions", False):
                left_distance = (coordinate - anchor_coordinate[left]).abs()
                right_distance = (anchor_coordinate[right] - coordinate).abs()
                nearest = torch.where(right_distance < left_distance, right, left)
                return anchor_targets[nearest]
            denominator = (
                anchor_coordinate[right] - anchor_coordinate[left]
            ).clamp_min(1e-12)
            ratio = ((coordinate - anchor_coordinate[left]) / denominator).unsqueeze(1)
            ratio[right == left] = 0.0
            if hasattr(self, "pareto_anchor_breaks") and len(self.pareto_anchor_breaks):
                breaks = self.pareto_anchor_breaks.to(device=preferences.device)
                break_index = left.clamp(max=len(breaks) - 1)
                crosses_gap = (right > left) & breaks[break_index]
                ratio[crosses_gap] = (ratio[crosses_gap] >= 0.5).to(ratio.dtype)
            return torch.lerp(anchor_targets[left], anchor_targets[right], ratio)
        distance = torch.cdist(preferences, anchor_preferences)
        nearest_count = min(
            int(getattr(self, "pareto_interpolation_neighbors", 8)),
            len(anchor_preferences),
        )
        if hasattr(self, "pareto_anchor_components"):
            components = self.pareto_anchor_components.to(
                device=preferences.device
            )
            nearest_component = components[distance.argmin(dim=1)]
            same_component = (
                components.unsqueeze(0) == nearest_component.unsqueeze(1)
            )
            distance = distance.masked_fill(~same_component, torch.inf)
        nearest_distance, nearest_index = torch.topk(
            distance, nearest_count, largest=False, dim=1
        )
        if (
            getattr(self, "binary_decisions", False)
            or getattr(self, "pareto_disconnected", False)
            or nearest_count == 1
        ):
            return anchor_targets[nearest_index[:, 0]]

        local_targets = anchor_targets[nearest_index]
        inverse_distance = 1.0 / nearest_distance.clamp_min(1e-12)
        interpolation_weights = (
            inverse_distance / inverse_distance.sum(dim=1, keepdim=True)
        )
        exact = nearest_distance[:, 0] <= 1e-12
        if bool(exact.any()):
            interpolation_weights[exact] = 0.0
            interpolation_weights[exact, 0] = 1.0

        if hasattr(self, "pareto_anchor_objectives"):
            anchor_objectives = self.pareto_anchor_objectives.to(
                device=preferences.device, dtype=preferences.dtype
            )
            local_objectives = anchor_objectives[nearest_index]
            lower = anchor_objectives.amin(dim=0, keepdim=True)
            span = (
                anchor_objectives.amax(dim=0, keepdim=True) - lower
            ).clamp_min(1e-8)
            normalized_local = (local_objectives - lower) / span
            local_spread = torch.cdist(
                normalized_local, normalized_local
            ).amax(dim=(1, 2))
            typical = float(
                getattr(self, "pareto_interpolation_typical_spread", 0.0)
            )
            if typical > 0.0:
                discontinuous = (
                    local_spread
                    > typical
                    * float(getattr(self, "pareto_interpolation_gap_factor", 6.0))
                )
                interpolation_weights[discontinuous] = 0.0
                interpolation_weights[discontinuous, 0] = 1.0
        return torch.sum(
            local_targets * interpolation_weights.unsqueeze(2), dim=1
        )

    def configure_component_decoder(self, component_count, embedding_dim=None):
        """Create a neural hard-routed decoder for disconnected Pareto sets."""
        component_count = int(component_count)
        if component_count < 2:
            self.pareto_router_enabled = False
            return
        embedding_dim = int(
            min(16, component_count) if embedding_dim is None else embedding_dim
        )
        reference = next(self.parameters())
        self.pareto_router = nn.Sequential(
            nn.Linear(self.preference_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, component_count),
        ).to(device=reference.device, dtype=reference.dtype)
        self.pareto_component_embedding = nn.Embedding(
            component_count, embedding_dim
        ).to(device=reference.device, dtype=reference.dtype)
        self.pareto_component_decoder = nn.Sequential(
            nn.Linear(
                self.vae_latent_dim + self.preference_dim + embedding_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.decision_dim),
        ).to(device=reference.device, dtype=reference.dtype)
        for module in (
            self.pareto_router,
            self.pareto_component_embedding,
            self.pareto_component_decoder,
        ):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0)
                    nn.init.constant_(layer.bias, 0.0)
        self.pareto_component_count = component_count
        self.pareto_router_enabled = True

    def set_generation_mode(self, mode):
        """Select neural generation (default) or explicit anchor ablation."""
        if mode not in {"decoder", "anchor"}:
            raise ValueError("generation mode must be 'decoder' or 'anchor'")
        if mode == "anchor" and not hasattr(self, "pareto_anchor_targets"):
            raise RuntimeError("anchor generation requires a trained Pareto archive")
        self.generation_mode = mode
        return self

    def configure_pareto_output(self, fixed_mask, fixed_values, decision_weights):
        """Store learned Pareto-set statistics used by neural generation."""
        reference = next(self.parameters())
        self.pareto_fixed_mask = fixed_mask.to(
            device=reference.device, dtype=torch.bool
        )
        self.pareto_fixed_values = fixed_values.to(
            device=reference.device, dtype=reference.dtype
        )
        self.pareto_decision_weights = decision_weights.to(
            device=reference.device, dtype=reference.dtype
        )

    def _apply_output_constraints(self, decisions):
        if self.activation == "tanh":
            decisions = torch.tanh(decisions)
        fixed_mask = self.pareto_fixed_mask.to(device=decisions.device)
        if bool(fixed_mask.any()):
            fixed_values = self.pareto_fixed_values.to(
                device=decisions.device, dtype=decisions.dtype
            )
            decisions = torch.where(
                fixed_mask.unsqueeze(0), fixed_values.unsqueeze(0), decisions
            )
        if getattr(self, "binary_decisions", False):
            hard = torch.where(
                decisions >= self.binary_threshold,
                decisions.new_tensor(self.binary_upper),
                decisions.new_tensor(self.binary_lower),
            )
            # Hard values in the forward pass, identity gradient in the
            # backward pass: neural distillation remains trainable for ZDT5.
            decisions = decisions + (hard - decisions).detach()
        return decisions

    def decode_with_components(self, z, preferences, components):
        """Decode through the learned component expert without archive access."""
        preferences = self._validate_preferences(preferences, len(z), z)
        components = components.to(device=z.device, dtype=torch.long)
        embedding = self.pareto_component_embedding(components)
        decisions = self.pareto_component_decoder(
            torch.cat((z, preferences, embedding), dim=1)
        )
        return self._apply_output_constraints(decisions)

    def decode(self, z, preferences=None):
        preferences = self._validate_preferences(preferences, len(z), z)
        if getattr(self, "generation_mode", "decoder") == "anchor":
            decisions = self._interpolate_pareto_anchors(preferences)
            if getattr(self, "binary_decisions", False):
                decisions = self._apply_output_constraints(decisions)
            return decisions

        if getattr(self, "pareto_router_enabled", False):
            logits = self.pareto_router(preferences)
            components = logits.argmax(dim=1)
            return self.decode_with_components(z, preferences, components)

        decisions = self.decoder(torch.cat((z, preferences), dim=1))
        return self._apply_output_constraints(decisions)

    def reconstruct(self, decisions, preferences):
        mu, logvar = self.encode(decisions, preferences)
        reconstruction = self.decode(self.reparameterize(mu, logvar), preferences)
        return reconstruction, mu, logvar

    def forward(self, z, preferences=None):
        """Generate decisions; use ``reconstruct`` for the VAE training path."""
        return self.decode(z, preferences)
