"""Node-feature transforms for DCBA dataset records."""

import torch
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch_geometric.transforms import BaseTransform


class CommunityToSize(BaseTransform):
    """
    Convert raw community IDs to normalised community-size node features.

    For each node and each layer, computes the fraction of *active* nodes
    (``community != 0``) that share the same community label.  Inactive nodes
    (``community == 0``, mABCD only) receive ``0.0``.

    Writes a float tensor of shape ``[num_actors, num_layers]`` into
    ``data["actor"].x``, replacing any existing value.
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
    Set node features to a constant, carrying no community information at all.

    Ablation baseline for :class:`CommunityToSize`, to isolate how much of the graph encoder's
    predictive power actually comes from the community-size feature versus pure graph structure.
    Writes a tensor of ones with the same shape ``[num_actors, num_layers]`` that
    :class:`CommunityToSize` would produce, so :class:`~dcba.models.gps_encoder.GPSEncoder` needs
    no changes to consume it.
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
