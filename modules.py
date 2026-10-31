import torch
from torch import nn, Tensor
from jaxtyping import Float
import math

from torch.nn.functional import one_hot


class SinusoidalPEWithMLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        Args:
            embedding_dim: Dimension of the initial sinusoidal embedding (must be even).
            hidden_dim: Output dimension of the time embedding required by the main model.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # MLP to post-process the raw sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (B,) containing continuous time steps in [0, 1].
        Returns:
            Tensor of shape (B, hidden_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2

        if half_dim < 1:
            raise ValueError("embedding_dim must be at least 2")

        # Standard transformer-style log frequency scaling
        # Note: Often scaled up by 1000 or 10000 depending on framework defaults
        if half_dim == 1:
            frequencies = torch.ones(1, device=device)
        else:
            frequencies = torch.exp(
                torch.arange(half_dim, device=device)
                * -(math.log(10000) / (half_dim - 1))
            )

        # Compute angles: outer product of t and frequencies
        # Shape: (B, 1) * (1, half_dim) -> (B, half_dim)
        args = t[:, None] * frequencies[None, :]

        # Concatenate sin and cos components
        # Shape: (B, embedding_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Map to final hidden size
        return self.mlp(emb)


# part one, feedforward directly.
class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, c: Float[Tensor, "n K"]) -> Float[Tensor, "n 2"]:
        return self.sequential(c)


@torch.no_grad()
def create_dataset(
    n_sample: int,
    *,
    K: int = 8,
    r: float = 0.08,
    noise: float = 0.2,
    device: torch.device = torch.device("cuda"),
    manual_model_num: int | None = None,
    max_time: float = 0.95,
    endpoint_probability: float = 0.1,
):
    mode = torch.randint(
        low=0, high=K, size=(n_sample,), device=device, dtype=torch.long
    )
    if manual_model_num is not None:
        assert manual_model_num < K, (manual_model_num, K)
        mode.fill_(manual_model_num)

    centers = torch.stack(
        [
            2 * torch.cos(2 * torch.pi * mode / K),
            2 * torch.sin(2 * torch.pi * mode / K),
        ],
        dim=-1,
    )
    x0 = centers + noise * torch.randn_like(centers)
    cr = one_hot(mode, num_classes=K).float()
    sigma_sparse = 2.5
    sigma_rich = 0.08
    cr = cr + ((1 - r) * sigma_sparse + r * sigma_rich) * torch.randn_like(cr)

    t = max_time * torch.rand((n_sample,), dtype=torch.float32, device=device)
    endpoint = torch.rand((n_sample,), device=device) < endpoint_probability
    t = t.masked_fill(endpoint, 0.0)
    xt = (1 - t[..., None]) * torch.randn_like(x0) + t[..., None] * x0

    return x0, cr, mode, xt, t


class FlowMatchingNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = 256
        self.max_time = 0.95
        self.register_buffer(
            "frequencies",
            torch.exp(-math.log(10_000) * torch.arange(64) / 64),
        )
        self.context_embedding = nn.Sequential(
            nn.Linear(input_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(128, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        self.hidden = nn.ModuleList(
            [nn.Linear(input_dim + 2 if i == 0 else self.inner_dim, self.inner_dim)
             for i in range(3)]
        )
        self.film = nn.ModuleList(
            [nn.Linear(2 * self.inner_dim, 2 * self.inner_dim) for _ in self.hidden]
        )
        for layer in self.film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.readout = nn.Linear(self.inner_dim, 2)

    def forward(
        self, xt: Float[Tensor, "n 2"], t: Float[Tensor, "n"], c: Float[Tensor, "n k"]
    ) -> Float[Tensor, "n 2"]:
        t = t.reshape(-1, 1).to(dtype=xt.dtype).clamp(max=self.max_time)
        phase = 1_000 * t * self.frequencies[None, :]
        t_features = torch.cat((phase.cos(), phase.sin()), dim=-1)
        conditioning = torch.cat(
            (self.context_embedding(c), self.time_embedding(t_features)), dim=-1
        )
        h = torch.cat((xt, c), dim=-1)
        for layer, modulation in zip(self.hidden, self.film):
            scale, shift = modulation(conditioning).chunk(2, dim=-1)
            h = torch.nn.functional.silu((1 + scale) * layer(h) + shift)
        return self.readout(h)
