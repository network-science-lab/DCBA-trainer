"""GIN-based graph encoder with a config-prediction head."""

import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor, stack
from torch_geometric.nn import AttentionalAggregation, GINConv, Sequential

from dcba.models.types import ForwardOutput


class LayerwiseAggregation(nn.Module):
    """Auxiliary class for trainable custom aggregation of mln-layers embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.attn = nn.Linear(hidden_channels, 1)

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        Trainable aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict ``{nb_mln_layers: [hidden_dim, nb_mln_actors]}``.

        :returns: A tensor of shape ``[hidden_dim, nb_mln_actors]``.
        """
        stacked = stack(list(h.values()))
        attn_scores = self.attn(stacked)
        attn_scores = nn.functional.softmax(attn_scores, dim=0)
        return (attn_scores * stacked).sum(dim=0)


class GINEncoder(nn.Module):
    """
    GIN-based graph encoder that maps a graph to an embedding ``z_g`` and predicts ``theta_hat``.

    No representation constraint is applied -- this is the regression baseline used in Stage 1
    as part of the joint ``L_reg + lambda * L_SupCon`` objective. Produces an embedding suitable
    for contrastive learning when paired with
    :class:`~dcba.models.config_autoencoder.ConfigAutoEncoder`.

    :param input_dim: Node feature dimensionality.
    :param hidden_dims: Sizes of intermediate GIN layers.
    :param embedding_dim: Dimensionality of the graph embedding ``z_g``.
    :param output_dim: Dimensionality of the predicted config vector (e.g. 9 for ABCD).
    :param dropout: Dropout probability applied after each GIN layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Build encoder and decoder MLPs from the supplied architecture parameters."""
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        enc_dims = hidden_dims + [embedding_dim]
        layers = []
        for i in range(len(enc_dims) - 1):
            mlp = nn.Sequential(
                nn.Linear(enc_dims[i], enc_dims[i + 1]),
                nn.ReLU(),
                nn.Linear(enc_dims[i + 1], enc_dims[i + 1]),
            )
            layers.append((GINConv(nn=mlp, train_eps=True), "x, edge_index -> x"))
            if i < len(enc_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(enc_dims[i + 1]))

        self._encoder = Sequential("x, edge_index", layers)
        self._dropout = nn.Dropout(dropout)
        self._aggregator = LayerwiseAggregation(embedding_dim)
        self._pool = AttentionalAggregation(gate_nn=nn.Linear(embedding_dim, 1))

        dec_dims = [embedding_dim] + list(reversed(hidden_dims)) + [output_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*dec_layers)

    def encode(self, data: DCBAHeteroData) -> Tensor:
        """
        Compute graph embedding ``z_g`` from a batch of heterogeneous graphs.

        :param data: Batched heterogeneous graph data.

        :returns: Embedding tensor of shape ``(batch, embedding_dim)``.
        """
        x = self.input_proj(data["actor"].x)

        y_relations = {}
        for relation, edge_index in data.edge_index_dict.items():
            h = self._encoder(x, edge_index)
            y_relations[relation] = self._dropout(h)

        agg = self._aggregator(y_relations)
        return self._pool(agg, index=data["actor"].batch)

    def decode(self, z: Tensor) -> Tensor:
        """
        Predict config vector ``theta_hat`` from embedding ``z_g``.

        :param z: Embedding tensor of shape ``(batch, embedding_dim)``.

        :returns: Predicted config tensor of shape ``(batch, output_dim)``.
        """
        return self._decoder(z)

    def forward(self, batch: DCBAHeteroData) -> ForwardOutput:
        """
        Run the full graph-to-config forward pass.

        :param batch: Batched heterogeneous graph data.

        :returns: :class:`~dcba.models.types.ForwardOutput` with ``embedding`` set to ``z_g``
            and ``reconstruction`` set to ``theta_hat``.
        """
        z_g = self.encode(batch)
        theta_hat = self.decode(z_g)
        return ForwardOutput(embedding=z_g, reconstruction=theta_hat)
