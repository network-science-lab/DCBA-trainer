"""Transforms for DCBA dataset records."""

import torch
from dcba_data_set.graph_io.data_models import ConfigRecord
from torch import Tensor
from torch_geometric.transforms import BaseTransform

#: Ordered list of numerical ABCD config keys used as model input features.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]


class ABCDConfigToTensor(BaseTransform):
    """
    Transform an ABCD :class:`~dcba_data_set.graph_io.data_models.ConfigRecord` to a float tensor.

    Extracts the 9 numerical ABCD config fields in the order defined by
    :data:`ABCD_CONFIG_KEYS` and returns a 1-D ``float32`` tensor of shape ``(9,)``.
    """

    def forward(self, data: ConfigRecord) -> Tensor:
        """
        Apply the transform.

        :param data: An ABCD config record.

        :returns: A 1-D float tensor of shape ``(9,)``.
        """
        values = [float(data.data[key]) for key in ABCD_CONFIG_KEYS]
        return torch.tensor(values, dtype=torch.float32)
