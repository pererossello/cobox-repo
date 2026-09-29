from typing import Optional, TYPE_CHECKING

import equinox as eqx
import jax

from .window import Window

if TYPE_CHECKING:
    from ..box import Box


class Windows(eqx.Module):
    """A chain of Windows"""

    windows: tuple[Window, ...]

    def __init__(self, windows: tuple[Window, ...]):
        if isinstance(windows, Window):
            raise TypeError(
                "Windows(...) expects a tuple/list of Window, not one Window"
            )
        self.windows = tuple(windows)
        self._validate_inputs()

    def win_hat_on_box(self, box: "Box") -> jax.Array:
        out = self.windows[0].win_hat_on_box(box)
        for w in self.windows[1:]:
            out = out * w.win_hat_on_box(box)
        return out

    # -----------------------
    # --- SPECIAL METHODS ---
    # -----------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Window:
        return self.windows[idx]

    def __repr__(self) -> str:
        return f"Windows([{', '.join(repr(w) for w in self.windows)}])"

    # ---------------------
    # --- SERIALIZATION ---
    # ---------------------

    def to_yaml(self) -> str:
        from . import _serialize

        return _serialize.windows_to_yaml(self)

    @classmethod
    def from_yaml(cls, s: str) -> "Windows":
        from . import _serialize

        return _serialize.windows_from_yaml(s, cls)

    # ------------------
    # --- VALIDATION ---
    # ------------------

    def _validate_inputs(self) -> None:
        if len(self.windows) == 0:
            raise ValueError("Windows must have at least one Window")
        for i, w in enumerate(self.windows):
            if not isinstance(w, Window):
                raise TypeError(
                    f"factor[{i}] must be a Window; got {type(w).__name__}."
                )
            # if not w.normalized:
            #     raise ValueError(
            #         f"all factors must be normalized; factor[{i}] = {w!r} is not."
            #     )


def coerce_windows(arg) -> Optional[Windows]:
    """Normalise a user-supplied `windows` argument to `Optional[Windows]`.
    Accepts:
      - ``None`` -> ``None`` (sampled field).
      - ``Window`` -> ``Windows((Window,))``.
      - ``Windows`` -> the same object, unchanged.
      - ``tuple`` / ``list`` of ``Window`` -> ``Windows(tuple(...))``.
    """
    if arg is None:
        return None
    if isinstance(arg, Windows):
        return arg
    if isinstance(arg, Window):
        return Windows((arg,))
    if isinstance(arg, (tuple, list)):
        return Windows(tuple(arg))
    raise TypeError(
        f"windows= expects None, Window, Windows, or tuple/list of Window; "
        f"got {type(arg).__name__}."
    )
