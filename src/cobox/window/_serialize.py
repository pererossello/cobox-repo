from __future__ import annotations

from typing import Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .window import Window
    from .windows import Windows

_WINDOW_FIELDS = ("kind", "scale", "normalized")


# --------------
# --- WINDOW ---
# --------------


def window_to_dict(w: Window) -> dict:
    return {
        "kind": str(w.kind),
        "scale": float(w.scale),
        "normalized": bool(w.normalized),
    }


def window_from_dict(config: dict) -> Window:
    from .window import Window, is_window_kind

    kind = str(config["kind"])
    if not is_window_kind(kind):
        raise ValueError(f"unknown window kind: {kind!r}")
    return Window(
        kind=kind,
        scale=float(config["scale"]),
        normalized=bool(config["normalized"]),
    )


def window_to_yaml(window: Window) -> str:
    import yaml

    return yaml.safe_dump(window_to_dict(window), sort_keys=False)


def window_from_yaml(s: str, cls: Type[Window]) -> Window:
    import yaml

    config = yaml.safe_load(s)
    window = window_from_dict(config)
    if not isinstance(window, cls):
        raise TypeError(
            f"Window.from_yaml expected {cls.__name__}; got {type(window).__name__}."
        )
    return window


# ---------------
# --- WINDOWS ---
# ---------------


def windows_to_list(chain: Optional[Windows]) -> Optional[list]:
    if chain is None:
        return None
    return [window_to_dict(w) for w in chain.windows]


def windows_from_list(config) -> Optional[Windows]:
    if config is None:
        return None
    from .windows import Windows

    return Windows(tuple(window_from_dict(c) for c in config))


def windows_to_yaml(chain: Windows) -> str:
    import yaml

    return yaml.safe_dump(windows_to_list(chain), sort_keys=False)


def windows_from_yaml(s: str, cls: Type[Windows]) -> Windows:
    import yaml

    config = yaml.safe_load(s)
    chain = windows_from_list(config)
    if not isinstance(chain, cls):
        raise TypeError(
            f"Windows.from_yaml expected {cls.__name__}; got {type(chain).__name__}."
        )
    return chain
