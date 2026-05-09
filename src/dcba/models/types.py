"""Shared output types for DCBA encoder models."""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class EncoderOutput:
    """
    Unified return type for all DCBA encoder ``forward()`` methods.

    embedding: L2-normalised latent vector of shape ``(batch, embedding_dim)``.
    reconstruction: Reconstructed config tensor of shape ``(batch, input_dim)``, or ``None``
        when the encoder does not produce a numeric reconstruction.
    aux_loss: Scalar auxiliary loss, or ``None`` when not applicable.
    """

    embedding: Tensor
    reconstruction: Optional[Tensor] = None
    aux_loss: Optional[Tensor] = None
