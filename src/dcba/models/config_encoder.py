"""Config encoder module — MLP autoencoder for ABCD configuration vectors."""

import torch.nn as nn
from torch import Tensor


class ConfigEncoder(nn.Module):
    """
    Parametrisable MLP autoencoder that maps a config vector ``q`` to an embedding ``h_q``.

    The encoder projects ``input_dim → hidden_dims[0] → … → embedding_dim`` with ReLU
    activations between layers.  The decoder mirrors the encoder in reverse with no
    activation on the output layer, so reconstructed values are unconstrained reals.

    :param input_dim: Dimensionality of the input config vector (e.g. 9 for ABCD).
    :param hidden_dims: Sizes of intermediate hidden layers shared by encoder and decoder.
    :param embedding_dim: Dimensionality of the bottleneck embedding ``h_q``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
    ) -> None:
        """Build encoder and decoder MLPs from the supplied architecture parameters."""
        super().__init__()

        enc_dims = [input_dim] + hidden_dims + [embedding_dim]
        enc_layers: list[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc_layers.append(nn.ReLU())
        self._encoder = nn.Sequential(*enc_layers)

        dec_dims = [embedding_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        """
        Map a config vector to its embedding.

        :param x: Float tensor of shape ``(batch, input_dim)``.

        :returns: Embedding tensor of shape ``(batch, embedding_dim)``.
        """
        return self._encoder(x)

    def decode(self, h: Tensor) -> Tensor:
        """
        Reconstruct a config vector from its embedding.

        :param h: Embedding tensor of shape ``(batch, embedding_dim)``.

        :returns: Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self._decoder(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run the full autoencoder pass.

        :param x: Float tensor of shape ``(batch, input_dim)``.

        :returns: Tuple ``(h_q, x_hat)`` where ``h_q`` is the embedding and
            ``x_hat`` is the reconstruction.
        """
        h_q = self.encode(x)
        x_hat = self.decode(h_q)
        return h_q, x_hat
