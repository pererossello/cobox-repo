"""Common harmonic-mode selection for scalar and joint shell samplers."""

import jax


def sampling_mask(shell, restrictions):
    # Mode counts and packing indices are static, including under JIT.
    with jax.ensure_compile_time_eval():
        mask = shell.is_mode
        for restriction in restrictions:
            mask = mask & restriction.mask(shell)
        return mask.at[0, 0].set(False)
