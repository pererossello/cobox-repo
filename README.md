# cobox

A JAX framework for cosmological fields, spectra and catalogues. Version 0.2.0
establishes a small numerical backbone for research code built on top of it.

One distribution provides four import packages:

| Package | Responsibility | Main public objects |
| --- | --- | --- |
| `cobox` | Periodic Cartesian fields, Fourier operations, sampling and statistics | `Box`, `ScalarField`, `VectorField`, `TensorField`, `ModeSupport`, `GRFSampler`, `Spectrum`, `PowerSpectrum` |
| `colcdm` | Cosmological parameters and implemented physical models | `Cosmology`, `BackgroundCosmo`, `PrimordialCosmo`, `Growth`, `MatterTransfer`, `PrimordialSpectrum`, `LinearPower`, `AngularPower`, `Observable`, `ProjectionTerm`, `LPT`, `LagBias` |
| `cosphere` | Scalar sky fields, harmonic sampling, full-sky statistics and radial projection mathematics | `Shell`, `ShellField`, `ShellGRFSampler`, `JointShellGRFSampler`, `EllModeMask` |
| `cozcat` | Observed angles/redshifts, selection, sampling and weighted counts | `RedshiftCatalog` |

Generic field operations do not import cosmological models. Research-specific
estimators, survey processing and choices about galaxy/electron physics belong
in downstream projects.

## Installation

Python 3.12 or newer. From this checkout:

```sh
python -m pip install .
python -m pip install '.[sphere]'  # also enable pixel/harmonic transforms
python -m pip install '.[nufft]'   # optional nonuniform Fourier operations
```

Use `-e` for editable development. A built wheel accepts the same extras, e.g.
`python -m pip install './cobox-0.2.0-py3-none-any.whl[sphere]'`.
The `sphere` extra installs s2fft. Harmonic sampling, harmonic statistics,
pixel geometry and catalogue counts work without it; `.sht()` and `.isht()`
require it. The NUFFT extra is independent of spherical transforms.

Precision is controlled by the caller. Set it before constructing arrays or
compiling functions when float64 is needed:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Importing cobox does not change global precision. Float64 validation does not
establish float32 accuracy for every integration or cosmological regime.

## Linear power as a callable

```python
import jax.numpy as jnp
from colcdm import Cosmology, LinearPower, MatterTransfer, PrimordialSpectrum

cosmo = Cosmology.from_preset("planck18")
power = LinearPower(PrimordialSpectrum(), MatterTransfer())
pk = power.bind(a=1.0, cosmology=cosmo)
k = jnp.geomspace(1e-3, 1.0, 32)
values = pk(k)
sigma8 = power.sigma8(1.0, cosmo)
```

Models are callable objects. `.bind()` fixes named arguments while retaining a
JAX PyTree, so no lambda wrapper is needed. `Spectrum` shares integration and
plotting; only `PowerSpectrum` has power statistics such as `delta_sq()` and
`sigma8()`. A matter transfer is a `Spectrum`, not a power spectrum.
`integrate()` computes an integral over **d ln k**. The model and weight may
return a scalar or an array whose leading axis is k; trailing batch axes
broadcast against one another.

## Correlated skies and measured spectra

```python
import jax.numpy as jnp
from cosphere import Shell, JointShellGRFSampler

shell = Shell(nside=8, L=8)

def cl_fn(ell):
    covariance = jnp.array([[1.0, -0.4], [-0.4, 2.0]])
    return covariance[None, :, :] / (ell[:, None, None] + 1.0)**2

a, b = JointShellGRFSampler(2).sample_from_seed(7, shell, cl_fn)
measured = a.stats.cross_spectrum(b)
cross_cl = measured.ab.values

def smooth(ell):
    return jnp.exp(-0.01 * ell * (ell + 1))

smoothed = a.filter_ell(smooth)  # remains harmonic
# pixels = smoothed.isht()      # requires the sphere extra
```

The sampler accepts any callable returning `(L, n_fields, n_fields)` positive
semidefinite covariance matrices. This may be an analytic model, a bound
`colcdm.AngularPower.matrix` method (e.g. using `equinox.Partial`), or a downstream interpolator. Negative cross-power is
valid; the complete covariance must remain positive semidefinite. Samplers
omit the monopole. Estimates retain it if present and do not subtract a mean.

`filter_ell()` accepts a finite real scalar, an `(L,)` array, or a callable of
ell. It returns harmonic coefficients even for pixel input (transformed once).
It applies an isotropic filter, not a survey mask. Auto-power scales by
`b_ell**2`; cross-power scales by the product of the two filters.

## Catalogue weighted counts

```python
import jax.numpy as jnp
from cozcat import RedshiftCatalog
from cosphere import Shell

catalog = RedshiftCatalog.from_radec(
    ra=[0., 90., 180.], dec=[0., 30., -20.], z=[0.5, 0.7, 1.0], degrees=True
)
weights = jnp.array([1., 0.5, -0.25])
counts = catalog.shell_counts(Shell(nside=8), weights=weights)
assert jnp.allclose(counts.data.sum(), weights.sum())
```

These are weighted sums per pixel, with signed weights allowed. Completeness,
random-catalogue normalization, overdensity definitions and shot-noise models
are choices for the analysis using them.

Simulation catalogues are available as `colcdm.halo.HaloCatalog` and
`colcdm.halo.protohalo.ProtohaloCatalog` / `ProtohaloFinder`.
`HaloCatalog.to_redshift_catalog()` connects a light-cone catalogue to observed
angles and redshifts, with an optional specified Doppler shift. This does not
supply a calibrated galaxy population or an electron/optical-depth model.

## Numerical conventions and boundaries

- A box uses an even grid size and a consistent length unit. Its forward FFT
  includes the cell volume; Fourier power is normalized by box volume.
  Cosmological distances use Mpc/h, wavenumbers h/Mpc and matter power
  (Mpc/h)^3. `PrimordialSpectrum` instead takes physical k in 1/Mpc;
  `LinearPower` handles the h conversion. `As1e9` means 10^9 times A_s.
- `ModeSupport` is a trusted bound on **each Cartesian mode-index component**,
  not the full retained-mode mask. `support=5` is normalized to
  `ModeSupport(5)`. Declaring support does not filter or verify the data;
  false declarations can invalidate dealiased products. Use mode restriction
  operations when data actually need filtering.
- `Shell` uses HEALPix RING pixels, with multipoles `0 <= ell < L`. Real maps
  store `(L, L)` coefficients at `[ell, m >= 0]`; `m > ell` entries are unused.
  Harmonics use `a_lm = integral f Y_lm* dOmega` and power uses the sum over m
  divided by `2 ell + 1`. HEALPix transforms are numerical approximations;
  check band limit, pixel resolution and forward iteration convergence.
  Low requested band limits are padded internally to the backend minimum
  `2*nside` and cropped on return; this does not reduce transform cost.
- `cosphere.projection` exposes `RadialWindow`, `ThinShellWindow`,
  `TopHatWindow`, `GaussianWindow`, `TabulatedWindow`, `radial_kernel`,
  `angular_cl` and `spherical_jn`. These are cosmology-independent. Radial
  integration measures and Cartesian smoothing kernels have distinct roles.
- Direct non-Limber projection is implemented. Integration grids and cutoffs
  require convergence checks. “Exact” describes the projection formula, not
  exact numerical quadrature. `colcdm.AngularPower` supplies cosmological
  inputs for its implemented scalar, spatially flat projection model.
- Matter transfer/growth, LPT, Lagrangian bias and halo/protohalo calculations
  implement particular approximations. They are not calibrated predictions
  for every scale, redshift or tracer. Full-sky scalar estimates do not correct
  partial-sky coupling, survey selection, beams or noise automatically.

Deferred: bispectrum measurement, polar/fixed-amplitude box GRFs, Limber and
FFTLog solvers, automatic integration resolution, Bessel derivatives, spin-2
maps, relativistic survey observables, ARF/kSZ-specific models and estimators,
and survey likelihood/instrument integrations.

## Migrating to 0.2.0

- Theoretical power remains `cobox.PowerSpectrum`. Measured Cartesian results
  are `cobox.field.stats.PowerSpectrumEstimate` and `CrossSpectrumEstimate`.
  Estimator functions (`power_spectrum`, `cross_spectrum`) keep their names.
  Angular results remain `cosphere.field.stats.AngularSpectrumEstimate` and
  `AngularCrossSpectrumEstimate`. Old Cartesian result class names are removed.
- The unfinished bispectrum module is removed. Unsupported polar GRF choices
  now fail at construction. `LinearPower` requires a valid `transfer_x`.
- Current Cartesian HDF5 files include a field `type` and support metadata.
  Older files without those keys are not accepted or silently migrated.
  Read them using the matching older version and convert explicitly if needed.
- Halo imports live under `colcdm.halo`; the old `colcdm.protohalo` path is gone.
- NumPy is a direct dependency (>=2.0 for its dtype/copy conversion contract).
  Pydantic is no longer required. All four import packages report `0.2.0`.

The existing **v0.1.0 bridge-compatible tag must remain unchanged**. Bridge can
stay pinned to it (or its commit); new research code can pin a separately
reviewed 0.2.0 release once committed and tagged.

## Validation and development

Tests are local and Git-ignored by this repository's policy. Run focused files
while editing and the full `pytest` once before release; the suite uses four
workers and preserves `.jax_cache/` to reuse JAX compilation. Ignored tests
and scratch reports are not distributed in the wheel.

The isolated CPU validation environment uses Python 3.12.3, JAX/jaxlib
0.11.2, NumPy 2.5.3, Equinox 0.13.8, NumPyro 0.22.0 and s2fft 1.4.0.

The release validation record and exact dependency snapshots are in
`slop/library_release_20261006/`. Supported dependency bounds express intended
compatibility; validation of one resolved environment does not certify every
allowed combination. The NUFFT extra and accelerators require their own
backend-specific validation.
