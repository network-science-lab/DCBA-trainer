"""Schema for the flattened orig/regr/cross prediction table shared by wandb and local storage."""


def build_predictions_table(
    rows: list[tuple[str, int, list[float], list[float], list[float]]],
    config_keys: list[str],
    has_scaler: bool,
) -> tuple[list[str], list[list]]:
    """
    Flatten per-sample prediction rows into the ``{i}-orig``/``{i}-regr``/``{i}-crsm`` table layout.

    Each sample occupies three output rows:

    - ``{i}-orig``: original theta
    - ``{i}-regr``: MLP reconstruction -- ``config_encoder.decode(config_encoder.encode(theta))``
    - ``{i}-crsm``: cross-modal prediction -- ``config_encoder.decode(graph_encoder.encode(G))``

    Shared by :meth:`~dcba.wrappers.supcon.DCBASupConWrapper.on_test_epoch_end` (which logs the
    result as a ``wandb.Table``) and ``scripts/compute_test_predictions.py`` (which writes it to a
    local JSON file), so both paths produce byte-identical schema for
    ``scripts/analyse_predictions.py`` to consume.

    :param rows: One ``(instance_id, replica, orig, recon, cross)`` tuple per sample.
    :param config_keys: ABCD config feature names, in the same order as each row's value lists.
    :param has_scaler: Whether the values in ``rows`` are already in raw (unscaled) units --
        controls whether column names get a ``_norm`` suffix.

    :returns: ``(columns, data)`` -- ``columns`` is ``["sample", "instance", "replica", <keys>]``;
        ``data`` has three rows per input sample, in ``orig``/``regr``/``crsm`` order.
    """
    suffix = "" if has_scaler else "_norm"
    columns = ["sample", "instance", "replica"] + [f"{k}{suffix}" for k in config_keys]
    data = []
    for i, (instance, replica, orig, recon, cross) in enumerate(rows):
        data.append([f"{i}-orig", instance, replica] + orig)
        data.append([f"{i}-regr", instance, replica] + recon)
        data.append([f"{i}-crsm", instance, replica] + cross)
    return columns, data
