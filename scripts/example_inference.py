"""Run a trained DCBA checkpoint on a graph it has never seen: ``G -> theta``."""

from pathlib import Path

import networkx as nx
import torch
from bidict import bidict
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from dcba.dataset.scalers import ABCD_CONFIG_KEYS
from dcba.eval.checkpoints import load_supcon_wrapper
from dcba.training.trainer import _build_node_transform

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = "gps-ae-supcon-log-scaler"
# MODEL = "gps-ae-supcon-log-scaler-comm"
# MODEL = "gps-ae-supcon-log-scaler-comm-b"

CHECKPOINT = REPO_ROOT / "checkpoints" / f"{MODEL}.ckpt"
CONFIG = REPO_ROOT / "configs" / f"{MODEL}.yaml"

LFR = {
    "n": 5000,
    "tau1": 3.0,
    "tau2": 1.5,
    "mu": 0.2,
    "average_degree": 15,
    "min_community": 100,
    "max_community": 600,
    "seed": 42,
}


def to_dcba_graph(graph: nx.Graph) -> DCBAHeteroData:
    """
    Convert a networkx graph into the single-layer structure the encoders read.

    :param graph: Undirected graph whose nodes carry LFR's ``community`` attribute.

    :returns: The graph as a :class:`~dcba_data_set.graph_io.data_models.DCBAHeteroData`.
    """
    nodes = sorted(graph.nodes())
    index = {node: position for position, node in enumerate(nodes)}

    edge_index = torch.tensor(
        [[index[u] for u, _ in graph.edges()], [index[v] for _, v in graph.edges()]],
        dtype=torch.long,
    )
    communities = sorted({frozenset(graph.nodes[n]["community"]) for n in graph}, key=len)
    labels = torch.zeros(len(nodes), 1, dtype=torch.long)
    for community_id, members in enumerate(communities, start=1):
        for node in members:
            labels[index[node], 0] = community_id

    data = DCBAHeteroData()
    data["actor", "l_0", "actor"].edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data["actor"].community = labels
    data.instance_id = "lfr-example"
    data.replica = torch.tensor([0], dtype=torch.long)
    data.actors_map = bidict({str(node): position for node, position in index.items()})
    data.layers_map = bidict({"0": "l_0"})
    return data


def main() -> None:
    """Generate the example graph, run the checkpoint over it, and print the predicted theta."""
    graph = nx.LFR_benchmark_graph(**LFR)
    graph.remove_edges_from(nx.selfloop_edges(graph))  # ABCD graphs have no self-loops
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    raw_config = OmegaConf.load(CONFIG)
    raw_config.pop("hydra", None)
    config = OmegaConf.to_container(raw_config, resolve=True)
    wrapper, scaler = load_supcon_wrapper(config, CHECKPOINT, torch.device("cpu"))
    node_transform = _build_node_transform(config["data"].get("node_transform"))
    print(f"Model: {MODEL} ({config['data']['node_transform']})\n")

    batch = Batch.from_data_list([node_transform(to_dcba_graph(graph))])
    batch.config = torch.zeros(1, len(ABCD_CONFIG_KEYS))
    _, _, theta_hat = wrapper.encode_and_reconstruct(batch)
    predicted = scaler.inverse_transform(theta_hat)[0] if scaler else theta_hat[0]

    reference = {
        "n": graph.number_of_nodes(),
        "t1": LFR["tau1"],
        "t2": LFR["tau2"],
        "xi": LFR["mu"],
        "c_min": min(len(c) for c in {frozenset(graph.nodes[n]["community"]) for n in graph}),
        "c_max": max(len(c) for c in {frozenset(graph.nodes[n]["community"]) for n in graph}),
        "d_min": min(d for _, d in graph.degree()),
        "d_max": max(d for _, d in graph.degree()),
        "nout": 0,  # LFR assigns every node to a community
    }

    print(f"{'param':>6}  {'predicted':>12}  {'LFR reference':>14}")
    for key, value in zip(ABCD_CONFIG_KEYS, predicted.tolist(), strict=True):
        print(f"{key:>6}  {value:>12.3f}  {reference[key]:>14.3f}")


if __name__ == "__main__":
    main()
