"""Smoke tests for torch, CUDA, and PyG extension availability."""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class TestTorch:
    """Verify torch version and CUDA reachability."""

    def test_torch_version(self) -> None:
        """Installed torch carries a cu121 build tag."""
        assert "cu121" in torch.__version__, (
            f"Expected a cu121 build of torch, got {torch.__version__!r}"
        )

    def test_cuda_available(self) -> None:
        """CUDA must be reachable with the current driver."""
        assert CUDA_AVAILABLE, (
            "CUDA is not available — check driver/torch CUDA version compatibility"
        )

    @cuda_only
    def test_cuda_tensor_roundtrip(self) -> None:
        """A tensor can be moved to GPU and back without error."""
        cpu = torch.tensor([1.0, 2.0, 3.0])
        gpu = cpu.cuda()
        assert gpu.device.type == "cuda"
        assert torch.allclose(gpu.cpu(), cpu)


class TestTorchScatter:
    """Verify torch_scatter import and basic GPU operation."""

    def test_import(self) -> None:
        """torch_scatter imports successfully."""
        import torch_scatter  # noqa: F401

    @cuda_only
    def test_scatter_add_cuda(self) -> None:
        """scatter_add produces correct results on GPU."""
        from torch_scatter import scatter

        src = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        index = torch.tensor([0, 0, 1, 1], device="cuda")
        out = scatter(src, index, reduce="sum")
        expected = torch.tensor([3.0, 7.0], device="cuda")
        assert torch.allclose(out, expected)


class TestTorchSparse:
    """Verify torch_sparse import and basic GPU operation."""

    def test_import(self) -> None:
        """torch_sparse imports successfully."""
        import torch_sparse  # noqa: F401

    @cuda_only
    def test_sparse_tensor_cuda(self) -> None:
        """A sparse COO matrix can be constructed on GPU."""
        from torch_sparse import SparseTensor

        row = torch.tensor([0, 1, 1], device="cuda")
        col = torch.tensor([1, 0, 2], device="cuda")
        val = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        mat = SparseTensor(row=row, col=col, value=val, sparse_sizes=(3, 3))
        assert mat.nnz() == 3


class TestPygLib:
    """Verify pyg_lib import."""

    def test_import(self) -> None:
        """pyg_lib imports successfully."""
        import pyg_lib  # noqa: F401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
