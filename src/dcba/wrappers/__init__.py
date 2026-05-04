"""Lightning wrappers for DCBA training regimes."""

from dcba.wrappers.autoencoder import DCBAAutoencoderWrapper
from dcba.wrappers.supcon import DCBASupConWrapper

__all__ = ["DCBAAutoencoderWrapper", "DCBASupConWrapper"]
