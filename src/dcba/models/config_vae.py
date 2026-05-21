"""Config VAE module -- reparameterised MLP VAE for ABCD configuration vectors."""

import torch
import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor

from dcba.models.types import ForwardOutput


class ConfigVAE(nn.Module):
    """
    Parametrisable MLP VAE that maps a config vector ``theta`` to a regularised embedding ``z``.

    The encoder projects ``input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1]`` (shared trunk),
    then splits into two linear heads producing ``mu`` and ``log_sigma`` of shape
    ``(batch, embedding_dim)``.  The decoder mirrors the plain AE: unconstrained reals output,
    no activation on the final layer.

    During training ``encode`` samples via the reparameterisation trick; at eval time it returns
    ``mu`` directly so that SupCon inference is deterministic.

    :param input_dim: Dimensionality of the input config vector (e.g. 9 for ABCD).
    :param hidden_dims: Sizes of intermediate hidden layers shared by trunk and decoder.
    :param embedding_dim: Dimensionality of the bottleneck ``z_theta``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
    ) -> None:
        """Build shared trunk, mu/log_sigma heads, and decoder from architecture parameters."""
        super().__init__()

        trunk_dims = [input_dim] + list(hidden_dims)
        trunk_layers: list[nn.Module] = []
        for i in range(len(trunk_dims) - 1):
            trunk_layers.append(nn.Linear(trunk_dims[i], trunk_dims[i + 1]))
            trunk_layers.append(nn.ReLU())
        self._trunk = nn.Sequential(*trunk_layers)
        trunk_out_dim = trunk_dims[-1]

        self._mu_head = nn.Linear(trunk_out_dim, embedding_dim)
        self._log_sigma_head = nn.Linear(trunk_out_dim, embedding_dim)

        dec_dims = [embedding_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*dec_layers)

    def encode_distribution(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the variational distribution parameters for the input config vector.

        :param x: Float tensor of shape ``(batch, input_dim)``.

        :returns: Pair ``(mu, log_sigma)`` each of shape ``(batch, embedding_dim)``.
        """
        h = self._trunk(x)
        return self._mu_head(h), self._log_sigma_head(h)

    def encode(self, x: Tensor) -> Tensor:
        """
        Map a config vector to a latent sample (training) or ``mu`` (eval).

        :param x: Float tensor of shape ``(batch, input_dim)``.

        :returns: Latent tensor ``z_theta`` of shape ``(batch, embedding_dim)``.
        """
        mu, log_sigma = self.encode_distribution(x)
        if self.training:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(log_sigma)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Reconstruct a config vector from a latent sample.

        :param z: Latent tensor of shape ``(batch, embedding_dim)``.

        :returns: Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self._decoder(z)

    def forward(self, batch: DCBAHeteroData) -> ForwardOutput:
        """
        Run the full VAE forward pass.

        :param batch: Batched heterogeneous graph data; the numeric config tensor is read from
            ``batch.config``.

        :returns: :class:`~dcba.models.types.ForwardOutput` with ``embedding`` set to
            ``z_theta`` and ``reconstruction`` set to ``theta_hat``.
        """
        config = batch.config.reshape(batch.batch_size, -1)
        z_theta = self.encode(config)
        theta_hat = self.decode(z_theta)
        return ForwardOutput(embedding=z_theta, reconstruction=theta_hat)
