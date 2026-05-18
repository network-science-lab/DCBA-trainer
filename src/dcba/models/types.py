"""Shared output types for DCBA encoder models."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ForwardOutput:
    """
    Unified return type for all DCBA encoder ``forward()`` methods.

    embedding: L2-normalised latent vector of shape ``(batch, embedding_dim)``.
    reconstruction: Reconstructed config tensor of shape ``(batch, input_dim)``, or ``None``
        when the encoder does not produce a numeric reconstruction.
    aux_loss: Scalar auxiliary loss, or ``None`` when not applicable.
    """

    embedding: Tensor
    reconstruction: Tensor | None = None
    aux_loss: Tensor | None = None
