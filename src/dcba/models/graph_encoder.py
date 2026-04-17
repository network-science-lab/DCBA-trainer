"""Graph encoder module — Phase 2 stub."""

import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor, stack
from torch_geometric.nn import GINConv, Sequential, global_mean_pool


class LayerwiseAggregation(nn.Module):
    """Auxuilary class for trainable custom aggregation of mln-layers embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.attn = nn.Linear(hidden_channels, 1)

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        Trainable aggregation of mln layers' embeddings.

        :param h: mln layers' embeddings dict `{nb_mln_layers: [hidden_dim, nb_mln_actors]}`
        :return: a tensor of shape `[hidden_dim, nb_mln_actors]`
        """
        h = stack(list(h.values()))
        attn_scores = self.attn(h)
        attn_scores = nn.functional.softmax(attn_scores, dim=0)
        weighted_sum = (attn_scores * h).sum(dim=0)
        return weighted_sum


class GraphEncoder(nn.Module):
    """Graph encoder that maps a graph to a latent embedding ``h_g``."""

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
                nn.ReLU(inplace=True),
                nn.Linear(enc_dims[i + 1], enc_dims[i + 1]),
            )
            layers.append((GINConv(nn=mlp, train_eps=True), "x, edge_index -> x"))
            if i < len(enc_dims) - 2:
                layers.append(nn.ReLU(inplace=True))

        self._encoder = Sequential("x, edge_index", layers)
        self._dropout = nn.Dropout(dropout)
        self._aggregator = LayerwiseAggregation(embedding_dim)

        dec_dims = [embedding_dim] + list(reversed(hidden_dims)) + [output_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*dec_layers)

    def encode(self, data: DCBAHeteroData) -> Tensor:
        """
        Map a config vector to its embedding.

        :param x: DCBAHeteroData.

        :returns: Embedding tensor of shape ``(batch, embedding_dim)``.
        """
        x = self.input_proj(data["actor"].community.float())

        y_relations = {}
        for relation, edge_index in data.edge_index_dict.items():
            h = self._encoder(x, edge_index)
            y_relations[relation] = self._dropout(h)

        agg = self._aggregator(y_relations)

        return global_mean_pool(agg, batch=None)

    def decode(self, h: Tensor) -> Tensor:
        """
        Reconstruct a config vector from its embedding.

        :param h: Embedding tensor of shape ``(batch, embedding_dim)``.

        :returns: Reconstructed tensor of shape ``(batch, input_dim)``.
        """
        return self._decoder(h)

    def forward(self, x: tuple[Tensor, DCBAHeteroData]) -> tuple[Tensor, Tensor]:
        """
        Run the full autoencoder pass.

        :param x: Tuple of a float tensor Float tensor of shape ``(batch, input_dim)``
        and DCBAHeteroData.

        :returns: Tuple ``(h_q, x_hat)`` where ``h_q`` is the embedding and
            ``x_hat`` is the reconstruction.
        """
        _, _x = x
        h_q = self.encode(_x)
        x_hat = self.decode(h_q)
        return h_q, x_hat
