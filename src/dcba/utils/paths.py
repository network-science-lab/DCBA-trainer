"""Canonical paths to datasets shared across training and evaluation code."""

import os
from pathlib import Path

#: Root of the dcba-data-set data directory. Override via ``DCBA_DATA_ROOT``.
DATA_ROOT = Path(os.environ.get("DCBA_DATA_ROOT", "/workspace/dev/DCBA-data-set/data"))

#: Small ABCD dataset used for tests.
TEST_ABCD_REPORT = DATA_ROOT / "test/dataset_abcd/report.json"

#: Small mABCD dataset used for tests.
TEST_MABCD_REPORT = DATA_ROOT / "test/dataset_mabcd/report.json"
