# Migrate to lazy-loader API from dcba-data-set

Adapts DCBA to the breaking changes introduced in dcba-data-set commit `f646ab99` (PR #6 — "Lazy
loader"):

- `load_report` now returns `list[InstanceRecord]` instead of
  `tuple[dict[str, ConfigRecord], list[DCBAHeteroData]]`
- `ConfigRecord` is renamed to `DCBAInstanceConfig`
- Graphs and configs are no longer loaded eagerly; callers must invoke
  `DCBAInstanceConfig.from_instance_record(record)` and
  `DCBAHeteroData.from_replica_record(replica, instance_id, net_type)`

## Steps

### 0. Re-lock and sync the environment

The `uv.lock` in DCBA-data-set was updated as part of the lazy-loader PR. Sync the DCBA virtual
environment to pick up any transitive-dependency changes and confirm the editable install is
healthy.

```bash
uv sync
```

Verify:
`uv run python -c "from dcba_data_set import InstanceRecord, DCBAInstanceConfig; print('ok')"`
should print `ok`.

No commit needed for this step.

---

### 1. Rename `ConfigRecord` → `DCBAInstanceConfig` in `transforms.py`

`src/dcba/dataset/transforms.py` imports the removed `ConfigRecord` class and uses it as the type
annotation for `ABCDConfigToTensor.forward`.

- Line 6: `from dcba_data_set.graph_io.data_models import ConfigRecord` →
  `from dcba_data_set.graph_io.data_models import DCBAInstanceConfig`
- Line 107: parameter annotation `data: ConfigRecord` → `data: DCBAInstanceConfig`
- Update the class docstring reference on line 101 accordingly.

Commit as a standalone mechanical rename.

---

### 2. Rewrite `ABCDDataset` with lazy graph loading

`src/dcba/dataset/dcba_dataset.py` currently eagerly loads all graphs and configs into memory.
Replace with a path-index approach: store `InstanceRecord`/`ReplicaRecord` proxies in `__init__`,
load graphs on demand in `__getitem__`. Configs (lightweight YAML) are loaded eagerly and converted
to tensors during `__init__`.

Dataset items become a single `DCBAHeteroData` with the ABCD config tensor attached as `.config` and
`.y` (no separate tensor return value).

**New `__init__` signature:**

```python
def __init__(
    self,
    records: list[InstanceRecord],
    scaler: ABCDConfigScaler | None = None,
    single_replica_per_instance: bool = True,
) -> None:
```

Internally, build two parallel lists:

- `self._replicas: list[tuple[ReplicaRecord, str, str]]` — `(replica, instance_id, net_type)`
- `self._tensors: list[Tensor]` — config tensor for each replica entry

Populate them by iterating `records`: for each record, call
`DCBAInstanceConfig.from_instance_record(record)` once to get the config tensor, then append either
only `record.replicas[0]` or all `record.replicas` depending on `single_replica_per_instance`.

**New `__getitem__`:**

```python
def __getitem__(self, idx: int) -> DCBAHeteroData:
    replica, instance_id, net_type = self._replicas[idx]
    g = DCBAHeteroData.from_replica_record(replica, instance_id, net_type)
    t = self._tensors[idx]
    g["actor"].x = zeros((len(g.actors_map), 5))
    g.config = t
    g.y = t.clone()
    g.actors_map = None
    g.layers_map = None
    return g
```

Update `from_report` to pass `records` directly:

```python
@classmethod
def from_report(cls, report_path: Path, scaler=None) -> "ABCDDataset":
    return cls(records=load_report(report_path), scaler=scaler)
```

Update imports:

```python
from dcba_data_set.graph_io.data_models import DCBAInstanceConfig, DCBAHeteroData, InstanceRecord, ReplicaRecord
```

Commit as a self-contained dataset rewrite.

---

### 3. Update `ABCDDataModule.setup`

`src/dcba/datamodule.py` currently tuple-unpacks `load_report` and builds filtered dicts. Replace
with list-based splitting on `InstanceRecord`.

```python
records = load_report(self._report_path)
instance_ids = [r.instance_id for r in records]
```

Compute split sizes as before, then filter:

```python
train_records = [r for r in records if r.instance_id in train_ids]
val_records   = [r for r in records if r.instance_id in val_ids]
test_records  = [r for r in records if r.instance_id in test_ids]
```

Pass each list as `records=…` to `ABCDDataset`. The three dataset objects cover disjoint sets of
configs — no replica from a given instance appears in more than one split.

Also propagate the `unique_configs` → `single_replica_per_instance` rename:

- `ABCDDataModule.__init__`: rename the stored parameter and the kwarg passed to `ABCDDataset`
- `src/dcba/training/trainer.py:73`: rename the kwarg at the call site (`unique_configs=True if …` →
  `single_replica_per_instance=True if …`)

Commit alongside or immediately after step 2.

---

### 4. Update `tests/test_dataset_loading.py`

The two smoke tests unpack `load_report` as a tuple — replace with assertions against
`list[InstanceRecord]`:

```python
records = load_report(report_path)
assert len(records) > 0
assert any(len(r.replicas) > 0 for r in records)
```

Rename the test methods so their names and docstrings stay accurate.

Run the suite to confirm everything passes:

```bash
uv run pytest
```

Commit as a standalone test update.
