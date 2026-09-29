from typing import TYPE_CHECKING, Union

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..box import Box

# ------------------
# --- PRIMITIVES ---
# ------------------


def fft(field: jax.Array, *, cell_v: float) -> jax.Array:
    """Forward transform of a scalar field. (*SHAPE) -> (*KSHAPE)."""
    out = jnp.fft.rfftn(field)
    return out * cell_v


def ifft(field_hat: jax.Array, *, inv_cell_v: float) -> jax.Array:
    """Inverse transform of a scalar field. (*KSHAPE) -> (*SHAPE)."""
    out = jnp.fft.irfftn(field_hat)
    return out * inv_cell_v


def fft_full(field: jax.Array, *, cell_v: float) -> jax.Array:
    """Forward transform of a scalar not-assumed-to-be-real field. (*SHAPE) -> (*SHAPE)."""
    out = jnp.fft.fftn(field)
    return out * cell_v


def ifft_full(field_hat: jax.Array, *, inv_cell_v: float) -> jax.Array:
    """Inverse transform of a scalar not-assumed-to-be-real field. (*SHAPE) -> (*SHAPE)."""
    out = jnp.fft.ifftn(field_hat)
    return out * inv_cell_v


def fft_on_vec(vec_field: jax.Array, *, D: int, cell_v: float) -> jax.Array:
    """Forward transform of each component. (D, *SHAPE) -> (D, *KSHAPE)."""
    axes = tuple(range(1, D + 1))
    out = jnp.fft.rfftn(vec_field, axes=axes)
    return out * cell_v


def ifft_on_vec(vec_field_hat: jax.Array, *, D: int, inv_cell_v: float) -> jax.Array:
    """Inverse transform of each component. (D, *KSHAPE) -> (D, *SHAPE)."""
    axes = tuple(range(1, D + 1))
    out = jnp.fft.irfftn(vec_field_hat, axes=axes)
    return out * inv_cell_v


# -----------------------
# --- FUNKY FFT STUFF ---
# -----------------------


def corners_mask(box: "Box") -> jax.Array:
    """KSHAPE boolean mask: True at the 2^D self-paired (corner) modes."""
    N = box.N
    mask = jnp.ones(box.KSHAPE, dtype=bool)
    for z in box.k_idx_axes:
        mask = mask & ((2 * z) % N == 0)
    return mask


def nyquist_slab_mask(box: "Box") -> jax.Array:
    """KSHAPE boolean mask: True where the compressed (last, rfft) axis alone
    sits at a self-paired index (0 or N//2).
    """
    N = box.N
    return jnp.broadcast_to((2 * box.k_idx_axes[-1]) % N == 0, box.KSHAPE)


def conj_reverse(x: jax.Array) -> jax.Array:
    """Reindex a KSHAPE array by each mode's in-array conjugate-partner index:
    negate every axis but the last (compressed) one, mod N; the last axis is
    left untouched since it was never mirrored by rfft compression.
    Self-inverse: ``conj_reverse(conj_reverse(x)) == x``.
    """
    for axis in range(x.ndim - 1):
        x = jnp.roll(jnp.flip(x, axis=axis), shift=1, axis=axis)
    return x


def canonical_mask(box: "Box") -> jax.Array:
    """KSHAPE boolean mask: True at exactly one representative per conjugate
    pair, plus every unpaired (free) mode. Together with ``corners_mask`` this
    gives the minimal, non-redundant parameterization of a real field's rfft.
    """
    rank = jnp.zeros(box.KSHAPE, dtype=jnp.int32)
    for axis in range(box.D - 1):
        shape = [1] * box.D
        shape[axis] = box.N
        idx = jnp.arange(box.N, dtype=jnp.int32).reshape(shape)
        rank = rank + idx * box.N**axis
    slab = nyquist_slab_mask(box)
    return jnp.where(slab, rank <= conj_reverse(rank), True)


def dof_weight(box: "Box") -> jax.Array:
    """KSHAPE int array: real DOF contributed by each mode (1 at corners, 2 at
    every other canonical mode, 0 at non-canonical/redundant positions)."""
    return jnp.where(corners_mask(box), 1, jnp.where(canonical_mask(box), 2, 0))


def n_dof(mask: jax.Array, box: "Box") -> int:
    """Total real DOF represented by ``mask`` (a KSHAPE boolean array, read
    only at canonical positions -- see ``canonical_mask``)."""
    with jax.ensure_compile_time_eval():
        return int(jnp.sum(dof_weight(box) * mask))


def _dof_partition(
    mask: jax.Array, box: "Box"
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split ``mask & canonical_mask(box)`` into (corner, paired, free)."""
    keep = mask & canonical_mask(box)
    corner = keep & corners_mask(box)
    slab = nyquist_slab_mask(box)
    paired = keep & slab & ~corner
    free = keep & ~slab
    return corner, paired, free


def _static_layout(
    mask: jax.Array, box: "Box"
) -> tuple[jax.Array, jax.Array, jax.Array, int, int]:
    """(corner, paired, general, n_c, n_g) for ``mask``.

    The mask depends only on static data, so this runs at trace time
    (``ensure_compile_time_eval``) and the sizes are Python ints under jit.
    """
    with jax.ensure_compile_time_eval():
        corner, paired, free = _dof_partition(mask, box)
        general = paired | free
        return corner, paired, general, int(jnp.sum(corner)), int(jnp.sum(general))


def _rank_within_mask(mask_flat: jax.Array) -> jax.Array:
    return jnp.cumsum(mask_flat) - 1


def _compact(x_flat: jax.Array, mask_flat: jax.Array, n: int) -> jax.Array:
    idx = jnp.where(mask_flat, _rank_within_mask(mask_flat), n)  # OOB -> dropped
    return jnp.zeros(n, dtype=x_flat.dtype).at[idx].set(x_flat, mode="drop")


def pack(coeffs: jax.Array, mask: jax.Array, box: "Box") -> jax.Array:
    """Inverse of ``unpack``: flat real vector of length ``n_dof(mask, box)``
    -> KSHAPE complex array. ``mask`` must be a concrete (static) array.
    """
    corner, paired, general, n_c, n_g = _static_layout(mask, box)

    gen_vals = coeffs[n_c : n_c + n_g] + 1j * coeffs[n_c + n_g :]

    if n_g > 0:
        general_flat = general.ravel()
        g_idx = _rank_within_mask(general_flat)
        out_flat = jnp.where(general_flat, gen_vals[jnp.clip(g_idx, 0, n_g - 1)], 0)
    else:
        # Empty and corner-only masks have no complex pairs to gather.
        out_flat = jnp.zeros(mask.size, dtype=gen_vals.dtype)

    if n_c > 0:
        corner_flat = corner.ravel()
        c_idx = _rank_within_mask(corner_flat)
        corner_vals = coeffs[:n_c].astype(gen_vals.dtype)
        out_flat = out_flat + jnp.where(
            corner_flat, corner_vals[jnp.clip(c_idx, 0, n_c - 1)], 0
        )

    out = out_flat.reshape(box.KSHAPE)
    paired_vals = jnp.where(paired, out, 0)
    return out + conj_reverse(jnp.conj(paired_vals))


def unpack(field_hat: jax.Array, mask: jax.Array, box: "Box") -> jax.Array:
    """Inverse of ``pack``: KSHAPE complex array -> flat real vector of length
    ``n_dof(mask, box)``. ``mask`` must be a concrete (static) array.
    """
    corner, _, general, n_c, n_g = _static_layout(mask, box)

    flat = field_hat.ravel()
    gen_vals = _compact(flat, general.ravel(), n_g)
    corner_vals = _compact(flat, corner.ravel(), n_c)
    return jnp.concatenate([corner_vals.real, gen_vals.real, gen_vals.imag])


def pad_rfft(field_hat: jax.Array, NEW_N: int, D: int) -> jax.Array:

    N = 2 * (field_hat.shape[-1] - 1)
    if NEW_N % 2 != 0:
        raise ValueError(f"NEW_N must be even; got {NEW_N}.")
    if NEW_N <= N:
        raise ValueError(f"NEW_N must be > N={N}; got {NEW_N}.")

    ndim = field_hat.ndim
    full_axes = tuple(range(ndim - D, ndim - 1))
    out = field_hat

    if full_axes:
        out = jnp.fft.fftshift(out, axes=full_axes)
        for ax in full_axes:
            idx: list[Union[slice, int]] = [slice(None)] * out.ndim
            idx[ax] = 0
            out = out.at[tuple(idx)].divide(2.0)
        pad = NEW_N - N
        before, after = pad // 2, pad - pad // 2
        pad_width = [(0, 0)] * out.ndim
        for ax in full_axes:
            pad_width[ax] = (before, after)
        out = jnp.pad(out, pad_width)
        for ax in full_axes:
            idx_lo: list[Union[slice, int]] = [slice(None)] * out.ndim
            idx_lo[ax] = before
            idx_hi: list[Union[slice, int]] = [slice(None)] * out.ndim
            idx_hi[ax] = before + N
            out = out.at[tuple(idx_hi)].set(out[tuple(idx_lo)])
        out = jnp.fft.ifftshift(out, axes=full_axes)

    nyq_idx: list[Union[slice, int]] = [slice(None)] * out.ndim
    nyq_idx[-1] = N // 2
    out = out.at[tuple(nyq_idx)].divide(2.0)
    last_pad = (NEW_N // 2 + 1) - (N // 2 + 1)
    pad_width_last = [(0, 0)] * out.ndim
    pad_width_last[-1] = (0, last_pad)
    return jnp.pad(out, pad_width_last)


def crop_rfft(field_hat: jax.Array, NEW_N: int, D: int) -> jax.Array:
    """Crop the last D spatial axes, summing coincident +/- Nyquist modes.

    Leading component/batch axes are preserved. A valid input real-field
    spectrum produces a valid output spectrum without an IFFT/FFT round trip.
    """
    N = 2 * (field_hat.shape[-1] - 1)
    if NEW_N % 2 != 0:
        raise ValueError(f"NEW_N must be even; got {NEW_N}.")
    if NEW_N >= N:
        raise ValueError(f"NEW_N must be < N={N}; got {NEW_N}.")

    ndim = field_hat.ndim
    full_axes = tuple(range(ndim - D, ndim - 1))
    out = field_hat

    if full_axes:
        out = jnp.fft.fftshift(out, axes=full_axes)
        s = (N - NEW_N) // 2
        for ax in full_axes:
            idx_lo: list[Union[slice, int]] = [slice(None)] * out.ndim
            idx_lo[ax] = s
            idx_hi: list[Union[slice, int]] = [slice(None)] * out.ndim
            idx_hi[ax] = s + NEW_N
            out = out.at[tuple(idx_lo)].add(out[tuple(idx_hi)])
        idx: list[Union[slice, int]] = [slice(None)] * out.ndim
        for ax in full_axes:
            idx[ax] = slice(s, s + NEW_N)
        out = out[tuple(idx)]
        out = jnp.fft.ifftshift(out, axes=full_axes)

    idx_last: list[Union[slice, int]] = [slice(None)] * out.ndim
    idx_last[-1] = slice(0, NEW_N // 2 + 1)
    out = out[tuple(idx_last)]
    # The missing negative-frequency plane folds onto the new Nyquist plane.
    # Its coefficients are the conjugates at reversed transverse wavevectors,
    # not generally a second copy of the positive-frequency coefficients.
    boundary = out[..., -1:]
    partner = jnp.conj(boundary)
    for ax in full_axes:
        partner = jnp.roll(jnp.flip(partner, axis=ax), shift=1, axis=ax)
    return out.at[..., -1:].set(boundary + partner)


def rfft_to_full(field_hat: jax.Array) -> jax.Array:
    """Expand an rfft-compressed array (shape KSHAPE, as returned by ``fft``)
    into the full array ``fft_full`` would give, via Hermitian symmetry.
    """
    D = field_hat.ndim
    N = 2 * (field_hat.shape[-1] - 1)
    expected = (N,) * (D - 1) + (N // 2 + 1,)
    if field_hat.shape != expected:
        raise ValueError(
            f"expected an rfft-compressed KSHAPE array {expected} (inferred "
            f"N={N}, D={D} from shape[-1]); got shape {field_hat.shape!r}."
        )

    mirrored = jnp.conj(conj_reverse(field_hat))
    tail = mirrored[..., N // 2 - 1 : 0 : -1]
    return jnp.concatenate([field_hat, tail], axis=-1)
