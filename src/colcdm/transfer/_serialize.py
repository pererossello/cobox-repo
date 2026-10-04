"""Tagged dictionaries for transfers: {"type": ..., **config}."""

from .matter import MatterTransfer

TRANSFER_TYPES = {"matter": MatterTransfer}


def transfer_from_dict(config: dict):
    """Rebuild any registered transfer from its tagged dictionary."""
    kind = config.get("type")
    if kind not in TRANSFER_TYPES:
        raise ValueError(
            f"unknown transfer type: {kind!r}; expected one of {sorted(TRANSFER_TYPES)}."
        )
    return TRANSFER_TYPES[kind].from_dict(config)
