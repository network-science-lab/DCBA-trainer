"""
Compute test predictions for a supcon run, on its own or a held-out dataset.

Each (run, dataset) pair is written to a local JSON file and logged as a W&B artifact on the
run itself, named after both the run and the dataset so a held-out evaluation (e.g. an
abcd-big-trained run scored against abcd-borderline) cannot be mistaken for the run's own test
split.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor
from tqdm import tqdm
from wandb.apis.public import Run

from dcba.dataset import ABCDBaseConfigScaler
from dcba.dataset.scalers import ABCD_CONFIG_KEYS
from dcba.eval.checkpoints import (
    apply_config_overrides,
    download_best_checkpoint,
    load_supcon_wrapper,
)
from dcba.eval.predictions_table import build_predictions_table
from dcba.eval.test_data import build_test_dataloader
from dcba.training.trainer import _build_node_transform, _build_transform
from dcba.utils.paths import DATA_ROOT
from dcba.wrappers.supcon import DCBASupConWrapper

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RUN_IDS = ["9dbudg6f", "sr3gfwb4", "vmp4lilb"]

DEFAULT_DATASETS = ["abcd-borderline"]

DEFAULT_ENTITY = "network-science-lab"
DEFAULT_PROJECT = "dcba"
DEFAULT_OUTPUT_DIR = Path(".analysis")
DEFAULT_CACHE_DIR = Path(".analysis/checkpoints")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_ids",
        nargs="*",
        default=DEFAULT_RUN_IDS,
        help="W&B run ids (bare id, or a full 'entity/project/run_id' path) to compute "
        f"predictions for. Default: {' '.join(DEFAULT_RUN_IDS)}.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset directory names under DCBA_DATA_ROOT to evaluate each run on. A dataset the "
        "run was trained on is evaluated on its own test split; any other one is evaluated in its "
        f"entirety. Default: {' '.join(DEFAULT_DATASETS)}.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="W&B entity used for bare run ids.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project used for bare run ids.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the '<run_id>/predictions_<dataset>.table.json' files are written under. "
        "Each file is also logged as a W&B artifact on its source run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory downloaded checkpoint artifacts are cached under.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def _resolve_run_path(run_id: str, entity: str, project: str) -> str:
    """
    Expand a bare run id into a full ``entity/project/run_id`` W&B path.

    :param run_id: Bare run id, or an already-full ``entity/project/run_id`` path.
    :param entity: W&B entity to use for a bare run id.
    :param project: W&B project to use for a bare run id.

    :returns: A full W&B run path.
    """
    return run_id if "/" in run_id else f"{entity}/{project}/{run_id}"


def _build_meta(
    run: Run,
    run_path: str,
    cfg: dict,
    ckpt_path: Path,
    dataset_name: str,
    trained_on: str,
    rows: list[tuple[str, int, list[float], list[float], list[float]]],
) -> dict:
    """
    Describe which run, checkpoint, and dataset a predictions file was computed from.

    Stored alongside ``columns``/``data`` so every downstream artefact can state what it is
    measuring instead of relying on the file's name or directory (see
    ``scripts/analyse_predictions.py``, which turns this into the report's title and file names).

    :param run: The W&B run the predictions come from.
    :param run_path: Full ``entity/project/run_id`` path the run was resolved from.
    :param cfg: Run config as logged to W&B, after :func:`apply_config_overrides`.
    :param ckpt_path: Local path of the checkpoint the predictions were computed with.
    :param dataset_name: Name of the dataset that was evaluated.
    :param trained_on: Name of the dataset the run was trained on.
    :param rows: The collected prediction rows, for the sample count.

    :returns: A JSON-serialisable metadata dict.
    """
    scaler_cfg = cfg["data"].get("scaler") or {}
    return {
        "run_id": run.id,
        "run_path": run_path,
        "run_name": run.name,
        "dataset": dataset_name,
        "trained_on": trained_on,
        # "own_test_split" means the run's actual held-out split of the dataset it trained on;
        # "whole_dataset" means every instance of a dataset the run has never seen.
        "test_scope": "own_test_split" if dataset_name == trained_on else "whole_dataset",
        "checkpoint": ckpt_path.name,
        "scaler": scaler_cfg.get("name"),
        "num_samples": len(rows),
    }


@torch.no_grad()
def _collect_predictions(
    wrapper: DCBASupConWrapper,
    scaler: ABCDBaseConfigScaler | None,
    cfg: dict,
    dataset_root: Path,
    whole_dataset_as_test: bool,
    device: torch.device,
) -> list[tuple[str, int, list[float], list[float], list[float]]]:
    """
    Run the trained wrapper over every sample of a dataset's test split.

    :param wrapper: Loaded :class:`~dcba.wrappers.supcon.DCBASupConWrapper` in eval mode.
    :param scaler: Scaler used to train the wrapper, for inverse-transforming predictions back to
        raw ABCD units.
    :param cfg: Run config as logged to W&B.
    :param dataset_root: Dataset directory to evaluate against.
    :param whole_dataset_as_test: Whether ``dataset_root`` is a fully held-out set that should be
        used in its entirety as "test" rather than split via the run's own
        ``val_ratio``/``test_ratio``.
    :param device: Device to run inference on.

    :returns: One ``(instance_id, replica, orig, recon, cross)`` tuple per sample.
    """
    data_cfg = cfg["data"]
    loader = build_test_dataloader(
        dataset_root=dataset_root,
        data_cfg=data_cfg,
        scaler=scaler,
        transform=_build_transform(data_cfg.get("transform")),
        node_transform=_build_node_transform(data_cfg.get("node_transform")),
        whole_dataset_as_test=whole_dataset_as_test,
        seed=cfg.get("random_seed", 42),
    )

    rows: list[tuple[str, int, list[float], list[float], list[float]]] = []
    for batch in tqdm(loader, desc=str(dataset_root.name), unit="batch"):
        batch = batch.to(device)
        config, theta_hat_regr, theta_hat_cross = wrapper.encode_and_reconstruct(batch)

        config_cpu: Tensor = config.detach().cpu()
        recon_cpu: Tensor = theta_hat_regr.detach().cpu()
        cross_cpu: Tensor = theta_hat_cross.detach().cpu()
        if scaler is not None:
            config_cpu = scaler.inverse_transform(config_cpu)
            recon_cpu = scaler.inverse_transform(recon_cpu)
            cross_cpu = scaler.inverse_transform(cross_cpu)

        for instance_id, replica, orig, recon, cross in zip(
            batch.instance_id,
            batch.replica.detach().cpu().tolist(),
            config_cpu.tolist(),
            recon_cpu.tolist(),
            cross_cpu.tolist(),
            strict=True,
        ):
            rows.append((instance_id, replica, orig, recon, cross))
    return rows


def main() -> None:
    """Compute and save test predictions for every requested (run, dataset) pair."""
    args = _parse_args()
    device = torch.device(args.device)
    api = wandb.Api()

    for run_id in args.run_ids:
        run_path = _resolve_run_path(run_id, args.entity, args.project)
        logger.info(f"Processing run {run_path}...")
        run = api.run(run_path)
        ckpt_path = download_best_checkpoint(run, args.cache_dir)
        cfg = apply_config_overrides(run.config, run.id)
        wrapper, scaler = load_supcon_wrapper(cfg, ckpt_path, device)
        trained_on = Path(cfg["data"]["dataset_root"]).name

        with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="must") as wrt_r:
            for dataset_name in args.datasets:
                # The run never saw a dataset other than the one it trained on, so no split there.
                whole_dataset_as_test = dataset_name != trained_on
                scope = "whole dataset" if whole_dataset_as_test else "own test split"
                logger.info(f"Evaluating {run.id} on {dataset_name} ({scope})...")
                rows = _collect_predictions(
                    wrapper,
                    scaler,
                    cfg,
                    DATA_ROOT / dataset_name,
                    whole_dataset_as_test,
                    device,
                )
                columns, data = build_predictions_table(
                    rows, ABCD_CONFIG_KEYS, has_scaler=scaler is not None
                )
                payload = {
                    "columns": columns,
                    "data": data,
                    "meta": _build_meta(
                        run, run_path, cfg, ckpt_path, dataset_name, trained_on, rows
                    ),
                }
                out_path = args.output_dir / run.id / f"predictions_{dataset_name}.table.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                logger.info(
                    f"Saved {len(rows)} predictions for {run.id}/{dataset_name} -> {out_path}"
                )

                # Named after both the run and the dataset (not just the dataset) so the artifact
                # is never confused with another run's evaluation of the same dataset.
                artifact_name = f"predictions-{run.id}-{dataset_name}"
                wrt_r.log_artifact(
                    artifact_or_path=str(out_path), name=artifact_name, type="predictions"
                )
                logger.info(f"Logged W&B artifact '{artifact_name}' for {run.id}/{dataset_name}")


if __name__ == "__main__":
    main()
