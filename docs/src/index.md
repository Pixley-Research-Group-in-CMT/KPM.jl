```@meta
CurrentModule = KPM
```

# KPM

```@index
```
# Moment calculation

The first step in KPM is calculating moments using Hamiltonians (and current operators for conductivity, etc.).
Functions with `!` are the more efficient in-place versions; functions without `!` are convenient wrappers that call the in-place implementations.

```@docs; canonical=false
kpm_1d
kpm_1d!
kpm_2d
kpm_2d!
```

# Applications

## DOS

To calculate the density of states (DOS) first calculate moments using `kpm_1d` / `kpm_1d!` with default (random) input vectors.
Then use the returned moments (`mu`) to evaluate the DOS.
There is also a convenience overload that accepts a Hamiltonian directly and performs the moment calculation for you via `dos`.

```@docs; canonical=false
dos
```

# Kernels

Kernels are functions with signature
```
kernel(n::Int64, N::Int64) -> Float64
```
such that when `n == 0` the kernel returns `1`, and when `n == N-1` it returns a small number close to `0`.

The package provides the `JacksonKernel` (the default for most applications) and `LorentzKernels`.

```@docs; canonical=false
JacksonKernel
```

The Lorentz kernel is useful for Green's functions because it preserves certain symmetries.
`LorentzKernels(λ)` returns a kernel function parameterized by λ.

```@docs; canonical=false
LorentzKernels
```

# Installation

This package is currently unregistered, so install it from GitHub:

```bash
] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
```

The package supports recent CUDA.jl versions (4 and 5). After installation, import with:

# full API reference

## API overview

Below is a concise list of the main public APIs provided by the package.

- Moment / KPM core:
  - `kpm_1d`, `kpm_1d!`
  - `kpm_1d_current`, `kpm_1d_current!`
  - `kpm_2d`, `kpm_2d!`
  - `kpm_3d`, `kpm_3d!`

- DOS / LDOS:
  - `dos`, `dos0`
  - `ldos_mu`

- Conductivity (DC / optical):
  - `d_dc_cond`, `dc_cond0`, `dc_cond_single`
  - `optical_cond1`, `d_optical_cond1`
  - `optical_cond2`, `d_optical_cond2`

- Nonlinear / CPGE:
  - `cpge`, `d_cpge`
  - Integration helpers: `Λnmp`, `Λn`, `Λnm`, `gn_R`, `gn_A`, `Δn`

- Kernels:
  - `JacksonKernel`, `LorentzKernels`

- Utilities (KPM.Utils / device helpers):
  - `wrapAdd`, `normalizeH`, `isNotBoundary`, `timestamp`
  - device helpers: `whichcore`, `maybe_to_device`, `maybe_to_host`, `maybe_on_device_rand`, `maybe_on_device_zeros`

For more details see the full API reference below.

```@autodocs
Modules = [KPM, KPM.Utils]
```
