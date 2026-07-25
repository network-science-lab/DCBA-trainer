"""Constraint-penalty augmented MSE loss for ABCD config reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dcba.dataset.scalers import ABCDBaseConfigScaler, ABCDIdentityConfigScaler


class ABCDConstraintPenaltyLoss(nn.Module):
    """
    MSE reconstruction loss plus squared-hinge penalties for ABCD config constraints.

    Total loss: ``weighted_MSE(x_hat, target) + lambda * sum(relu(violation)^2)``.

    Per-feature range penalties keep every predicted feature in ``[0, 1]``. Ordering penalties
    enforce, in raw scale via the attached scaler's :meth:`denormalise` (attach with
    :meth:`set_scaler`):

    - ``c_min <= c_max``
    - ``d_min <= d_max``
    - ``c_max <= n``
    - ``d_max <= n``
    - ``nout <= n``

    Violations are divided by the target's raw ``n`` to keep the hinge dimensionless.

    .. note::
        ``n`` is exempt from the above-range penalty, to allow predictions above ``n_max``.
        Under :class:`~dcba.dataset.scalers.ABCDIdentityConfigScaler` the whole above-range
        penalty is skipped, since its features stay in raw scale and exceed ``1`` by design.

    Feature order follows :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS`.

    :param lambda_penalty: Weight applied to the penalty terms.
    :param weights: Optional per-feature weight for the squared error, length 9 in
        :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS` order. ``None`` weights every feature
        equally (plain MSE).
    """

    def __init__(
        self,
        lambda_penalty: float = 1.0,
        weights: list[float] | None = None,
    ) -> None:
        """Initialise the loss with the given penalty weight and optional per-feature weights."""
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self._scaler: ABCDBaseConfigScaler | None = None
        self.register_buffer(
            "_weights", None if weights is None else torch.tensor(weights, dtype=torch.float32)
        )

    def set_scaler(self, scaler: ABCDBaseConfigScaler | None) -> None:
        """
        Attach the scaler whose :meth:`denormalise` maps ``x_hat`` to raw scale.

        :param scaler: Scaler providing a differentiable ``denormalise``, or ``None`` to detach.
        """
        self._scaler = scaler

    def _ordering_penalty(self, x_hat: Tensor, target: Tensor) -> Tensor:
        """
        Compute the mean squared-hinge ordering penalty for ``x_hat`` in raw scale.

        :param x_hat: Reconstructed normalised config tensor of shape ``(batch, 9)``.
        :param target: Ground-truth normalised config tensor of shape ``(batch, 9)``; its
            denormalised ``n`` provides the per-sample violation scale.

        :returns: Scalar penalty tensor.
        """
        if self._scaler is None:
            raise RuntimeError(
                "ABCDConstraintPenaltyLoss requires a scaler; call set_scaler() first."
            )
        values = self._scaler.denormalise(x_hat)
        # Normalise by the TARGET's raw n: constant, strictly positive, and independent of the
        # prediction, so the hinge stays linear with a correct-direction gradient and cannot
        # blow up (dividing by the predicted n, or a self-normalising denominator, breaks one
        # or the other).
        scale = self._scaler.denormalise(target.detach())[:, 0].clamp(min=1.0)

        # c_min (idx 4) <= c_max (idx 5)
        c_order = F.relu((values[:, 4] - values[:, 5]) / scale) ** 2
        # d_min (idx 6) <= d_max (idx 7)
        d_order = F.relu((values[:, 6] - values[:, 7]) / scale) ** 2
        # c_max (idx 5) <= n (idx 0)
        c_max_n = F.relu((values[:, 5] - values[:, 0]) / scale) ** 2
        # d_max (idx 7) <= n (idx 0)
        d_max_n = F.relu((values[:, 7] - values[:, 0]) / scale) ** 2
        # nout (idx 8) <= n (idx 0)
        nout_n = F.relu((values[:, 8] - values[:, 0]) / scale) ** 2
        return (c_order + d_order + c_max_n + d_max_n + nout_n).mean()

    def forward(self, x_hat: Tensor, target: Tensor) -> Tensor:
        """
        Compute weighted MSE + lambda * sum(relu(violation)^2).

        :param x_hat: Reconstructed normalised config tensor of shape ``(batch, 9)``.
        :param target: Ground-truth normalised config tensor of shape ``(batch, 9)``.

        :returns: Scalar loss tensor.
        """
        if self._weights is None:
            mse = F.mse_loss(x_hat, target)
        else:
            mse = (((x_hat - target) ** 2) * self._weights).mean()

        # Per-feature range: each feature should lie in [0, 1].
        # n (idx 0) is exempt from the above-range check -- the model may predict n > n_max
        # in edge cases with larger networks, which we do not want to penalise.
        below = F.relu(-x_hat)
        if isinstance(self._scaler, ABCDIdentityConfigScaler):
            # Raw-scale features exceed 1 by design, so only the below-zero half applies.
            range_penalty = (below**2).sum(dim=1).mean()
        else:
            above = F.relu(x_hat[:, 1:] - 1.0)  # features 1-8 only
            range_penalty = (below**2).sum(dim=1).mean() + (above**2).sum(dim=1).mean()

        penalty = range_penalty + self._ordering_penalty(x_hat, target)
        return mse + self.lambda_penalty * penalty
