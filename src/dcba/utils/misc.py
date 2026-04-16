"""Sweep config helpers."""


def unflatten_dict(cfg: dict) -> dict:
    """
    Flatten sweep dict config to hydra format.

    :param cfg: Wandb sweep dict.

    :returns: Formated plain dict with model hyperparameters.
    """
    result = {}
    for key, value in cfg.items():
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result
