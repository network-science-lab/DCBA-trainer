"""Constraint-penalty augmented MSE loss for ABCD config reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dcba.dataset.scalers import ABCDBaseConfigScaler


class ABCDConstraintPenaltyLoss(nn.Module):
    """
    MSE reconstruction loss augmented with squared-hinge penalties for ABCD config constraints.

    The total loss is::

        weighted_MSE(x_hat, target) + lambda * sum(relu(violation)^2)

    Per-feature range penalties ensure every predicted feature stays within ``[0, 1]``
    (normalised space). Cross-parameter ordering penalties enforce the structural constraints
    the ABCD generator requires to run at all:

    - ``d_min <= d_max``
    - ``c_max <= n``
    - ``d_max <= n``
    - ``nout <= n``

    Ordering comparisons always happen in **raw scale**, obtained by passing ``x_hat`` through
    the attached scaler's differentiable
    :meth:`~dcba.dataset.scalers.ABCDBaseConfigScaler.denormalise` (attach it with
    :meth:`set_scaler`; the training entry point does this automatically). Each
    violation is divided by the target's raw ``n`` so the hinge is dimensionless and O(1). This
    is correct under any scaler.

    .. note::
        ``c_min <= c_max`` is deliberately not enforced here: every scaler expresses ``c_min`` as
        ``c_min / c_max`` (see :class:`~dcba.dataset.scalers.ABCDConfigScaler`), so the
        ordering is guaranteed by construction together with the range penalty.

    .. note::
        ``n`` (index 0) is exempt from the above-range penalty (``> 1`` in normalised space)
        to accommodate edge cases where the model predicts graphs larger than ``n_max``.

    Feature indices follow :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS`:
    ``[n, t1, t2, xi, c_min, c_max, d_min, d_max, nout]`` -> indices 0-8.

    :param lambda_penalty: Weight applied to the sum of constraint penalty terms.
    :param weights: Optional per-feature weight applied to the squared error before averaging,
        length 9 in :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS` order. Use this to give
        harder-to-reconstruct parameters more gradient priority. ``None`` (default) weights every
        feature equally, identical to plain MSE.
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

        Required before the first forward pass. Kept out of the constructor so config-driven
        construction (``_build_loss(args)``) stays purely declarative -- the training entry point
        attaches the data pipeline's scaler after both are built.

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

        # d_min (idx 6) <= d_max (idx 7)
        d_order = F.relu((values[:, 6] - values[:, 7]) / scale) ** 2
        # c_max (idx 5) <= n (idx 0)
        c_max_n = F.relu((values[:, 5] - values[:, 0]) / scale) ** 2
        # d_max (idx 7) <= n (idx 0)
        d_max_n = F.relu((values[:, 7] - values[:, 0]) / scale) ** 2
        # nout (idx 8) <= n (idx 0)
        nout_n = F.relu((values[:, 8] - values[:, 0]) / scale) ** 2
        return (d_order + c_max_n + d_max_n + nout_n).mean()

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
        above = F.relu(x_hat[:, 1:] - 1.0)  # features 1-8 only
        range_penalty = (below**2).sum(dim=1).mean() + (above**2).sum(dim=1).mean()

        penalty = range_penalty + self._ordering_penalty(x_hat, target)
        return mse + self.lambda_penalty * penalty
