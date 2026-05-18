"""Constraint-penalty augmented MSE loss for ABCD config reconstruction."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ABCDConstraintPenaltyLoss(nn.Module):
    """
    MSE reconstruction loss augmented with squared-hinge penalties for ABCD config constraints.

    All arithmetic operates in normalised space (values expected in ``[0, 1]``), matching the
    model's output domain.  The total loss is::

        MSE(x_hat, target) + lambda * sum(relu(violation)^2)

    Per-feature range penalties ensure every predicted feature stays within ``[0, 1]``.
    Cross-parameter ordering penalties enforce the structural ABCD constraints:

    - ``c_min <= c_max``
    - ``d_min <= d_max``
    - ``c_max <= n``
    - ``d_max <= n``
    - ``nout <= n``

    .. note::
        ``n`` (index 0) is exempt from the above-range penalty (``> 1`` in normalised space)
        to accommodate edge cases where the model predicts graphs larger than ``n_max``.

    Feature indices follow :data:`~dcba.dataset.transforms.ABCD_CONFIG_KEYS`:
    ``[n, t1, t2, xi, c_min, c_max, d_min, d_max, nout]`` -> indices 0-8.

    :param lambda_penalty: Weight applied to the sum of constraint penalty terms.
    """

    def __init__(self, lambda_penalty: float = 1.0) -> None:
        """Initialise the loss with the given penalty weight."""
        super().__init__()
        self.lambda_penalty = lambda_penalty

    def forward(self, x_hat: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE + lambda * sum(relu(violation)^2).

        :param x_hat: Reconstructed normalised config tensor of shape ``(batch, 9)``.
        :param target: Ground-truth normalised config tensor of shape ``(batch, 9)``.

        :returns: Scalar loss tensor.
        """
        mse = F.mse_loss(x_hat, target)

        # Per-feature range: each feature should lie in [0, 1].
        # n (idx 0) is exempt from the above-range check -- the model may predict n > n_max
        # in edge cases with larger networks, which we do not want to penalise.
        below = F.relu(-x_hat)
        above = F.relu(x_hat[:, 1:] - 1.0)  # features 1-8 only
        range_penalty = (below**2).sum(dim=1).mean() + (above**2).sum(dim=1).mean()

        # Cross-parameter ordering constraints
        # c_min (idx 4) <= c_max (idx 5)
        c_order = F.relu(x_hat[:, 4] - x_hat[:, 5]) ** 2
        # d_min (idx 6) <= d_max (idx 7)
        d_order = F.relu(x_hat[:, 6] - x_hat[:, 7]) ** 2
        # c_max (idx 5) <= n (idx 0)
        c_max_n = F.relu(x_hat[:, 5] - x_hat[:, 0]) ** 2
        # d_max (idx 7) <= n (idx 0)
        d_max_n = F.relu(x_hat[:, 7] - x_hat[:, 0]) ** 2
        # nout (idx 8) <= n (idx 0)
        nout_n = F.relu(x_hat[:, 8] - x_hat[:, 0]) ** 2

        ordering_penalty = (c_order + d_order + c_max_n + d_max_n + nout_n).mean()

        penalty = range_penalty + ordering_penalty
        return mse + self.lambda_penalty * penalty
