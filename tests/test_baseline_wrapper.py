"""Unit tests for DCBABaselineWrapper -- heuristic ABCD baseline evaluated via Trainer.test."""

import lightning.pytorch as pl
import pytest
from dcba_data_set.graph_io import load_dataset
from torch_geometric.loader import DataLoader

from dcba.dataset import ABCDConfigScaler, ABCDConfigToTensor, ABCDDataset
from dcba.dataset.transforms import ABCD_CONFIG_KEYS
from dcba.utils.paths import TEST_ABCD_DATASET
from dcba.wrappers import DCBABaselineWrapper


@pytest.fixture(scope="module")
def test_loader() -> DataLoader:
    """Build a batch_size=1 dataloader over the small ABCD test dataset."""
    records = load_dataset(TEST_ABCD_DATASET)
    dataset = ABCDDataset(
        records=records,
        scaler=ABCDConfigScaler(),
        transform=ABCDConfigToTensor(),
        single_replica_per_instance=True,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def _run_test(wrapper: DCBABaselineWrapper, test_loader: DataLoader) -> list[dict]:
    """Run Trainer.test for the wrapper against the given dataloader, without any logger."""
    wrapper.set_scaler(ABCDConfigScaler())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        max_epochs=0,
        num_sanity_val_steps=0,
    )
    return trainer.test(wrapper, dataloaders=test_loader)


class TestDCBABaselineWrapperMockedTraining:
    """Training/validation must be no-ops; nothing should ever be optimised."""

    def test_configure_optimizers_returns_none(self) -> None:
        """configure_optimizers returns None so Trainer.fit runs without an optimiser."""
        wrapper = DCBABaselineWrapper()
        assert wrapper.configure_optimizers() is None

    def test_training_step_is_a_noop(self) -> None:
        """training_step returns None regardless of input."""
        wrapper = DCBABaselineWrapper()
        assert wrapper.training_step(None, 0) is None

    def test_validation_step_is_a_noop(self) -> None:
        """validation_step returns None regardless of input."""
        wrapper = DCBABaselineWrapper()
        assert wrapper.validation_step(None, 0) is None

    def test_fit_is_a_noop_with_max_epochs_zero(self, test_loader: DataLoader) -> None:
        """Trainer.fit runs zero epochs and never touches a batch when max_epochs=0."""
        wrapper = DCBABaselineWrapper()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            max_epochs=0,
            num_sanity_val_steps=0,
        )
        trainer.fit(wrapper, train_dataloaders=test_loader, val_dataloaders=test_loader)
        assert trainer.current_epoch == 0


class TestDCBABaselineWrapperTestStep:
    """Predictions extracted from real graphs in the small ABCD test dataset."""

    def test_rejects_batched_graphs(self, test_loader: DataLoader) -> None:
        """test_step refuses a batch with more than one graph."""
        batch = next(iter(DataLoader(test_loader.dataset, batch_size=2)))
        wrapper = DCBABaselineWrapper()
        wrapper.set_scaler(ABCDConfigScaler())
        with pytest.raises(ValueError, match="batch_size=1"):
            wrapper.test_step(batch, 0)

    def test_ground_truth_communities(self, test_loader: DataLoader) -> None:
        """detect_communities=False produces one prediction row per graph."""
        wrapper = DCBABaselineWrapper(detect_communities=False)
        metrics = _run_test(wrapper, test_loader)
        assert len(wrapper._test_rows) == len(test_loader.dataset)
        assert "test_loss" in metrics[-1]
        for _, _, orig, pred in wrapper._test_rows:
            assert len(orig) == len(ABCD_CONFIG_KEYS)
            assert len(pred) == len(ABCD_CONFIG_KEYS)

    def test_detected_communities(self, test_loader: DataLoader) -> None:
        """detect_communities=True (Leiden) also produces one prediction row per graph."""
        wrapper = DCBABaselineWrapper(detect_communities=True, leiden_seed=42)
        metrics = _run_test(wrapper, test_loader)
        assert len(wrapper._test_rows) == len(test_loader.dataset)
        assert "test_loss" in metrics[-1]

    def test_predicted_n_matches_true_n(self, test_loader: DataLoader) -> None:
        """Predicted node count always matches ground truth exactly (n is read off the graph)."""
        wrapper = DCBABaselineWrapper()
        _run_test(wrapper, test_loader)
        n_idx = ABCD_CONFIG_KEYS.index("n")
        for _, _, orig, pred in wrapper._test_rows:
            assert orig[n_idx] == pytest.approx(pred[n_idx])
