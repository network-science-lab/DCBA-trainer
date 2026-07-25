"""Node-feature transforms for DCBA dataset records."""

import torch
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch_geometric.transforms import BaseTransform


class CommunityToSize(BaseTransform):
    """
    Convert raw community IDs to normalised community-size node features.

    For each node/layer, computes the fraction of active nodes (``community != 0``) sharing that
    node's community label; inactive nodes get ``0.0``. Writes shape ``[num_actors, num_layers]``
    into ``data["actor"].x``.
    """

    def forward(self, data: DCBAHeteroData) -> DCBAHeteroData:
        """
        Apply the transform to a single (non-batched) graph.

        :param data: A single heterogeneous graph.

        :returns: The same graph with ``data["actor"].x`` set to normalised community-size features.
        """
        community = data["actor"].community  # [N, L], long
        n, num_layers = community.shape
        sizes = torch.zeros(n, num_layers, dtype=torch.float32)
        for layer in range(num_layers):
            c = community[:, layer]
            active = c != 0
            n_active = int(active.sum())
            if n_active == 0:
                continue
            # inv maps each active node back to its index in the unique-values array,
            # so counts[inv] broadcasts the per-community count to a per-node count.
            _, inv, counts = torch.unique(c[active], return_inverse=True, return_counts=True)
            sizes[active, layer] = counts[inv].float() / n_active
        data["actor"].x = sizes
        return data


class ConstantNodeFeatures(BaseTransform):
    """
    Set node features to a constant (no community information).

    Writes ones of shape ``[num_actors, num_layers]``, matching :class:`CommunityToSize`'s output.
    """

    def forward(self, data: DCBAHeteroData) -> DCBAHeteroData:
        """
        Apply the transform to a single (non-batched) graph.

        :param data: A single heterogeneous graph.

        :returns: The same graph with ``data["actor"].x`` set to all ones.
        """
        n, num_layers = data["actor"].community.shape
        data["actor"].x = torch.ones(n, num_layers, dtype=torch.float32)
        return data
