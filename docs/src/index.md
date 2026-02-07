```@meta
CurrentModule = KPM
```

# KPM

```@index
```

## Installation

This package is currently unregistered. Install the latest version directly from GitHub:

```bash
] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
```

Notes:

- The package supports CUDA.jl versions 4 and 5.
- After installation import with:

```julia
using KPM
```

- To update the package run:

```bash
] update KPM
```
and provide your GitHub username/password if prompted.

For more details see the project's README.

## Quick examples

### 1) Density of States (DOS) — concise example

```julia
using KPM, LinearAlgebra, SparseArrays, Plots

# small 1D tight-binding (periodic)
function tb1dchain(N; t=1.0)
  H = spzeros(N,N)
  for i in 1:N-1
    H[i,i+1] = -t; H[i+1,i] = -t
  end
  H[1,N] = -t; H[N,1] = -t
  return H
end

N = 1000
NC = 256
NR = 4
H = tb1dchain(N)
# rescale H to (-1,1)
a, Hn = KPM.normalizeH(H)
mu = KPM.kpm_1d(Hn, NC, NR)
E, rho = KPM.dos(mu, a; kernel=KPM.JacksonKernel, N_tilde=500)

plot(E, rho, xlabel="E", ylabel="DOS", legend=false)
savefig("docs/dos_example.pdf")
```

Reference (full example):

```1:54:paper/example.jl
using KPM
using LinearAlgebra
using SparseArrays

# Simple dense 1D tight-binding Hamiltonian (periodic)
function tb1dchain(N::Integer; t::Real=1.0)
    H = zeros(Float64, N, N)
    for i in 1:(N-1)
        H[i, i+1] = -t
        H[i+1, i] = -t
    end
    H[1, N] = -t
    H[N, 1] = -t
    return H
end
```

### 2) Optical conductivity (graphene) — concise example

```julia
using KPM, Plots
include("examples/GrapheneModel.jl") # provides GrapheneLattice

L = 60
Ham, Jx, Jy, Jxx, Jxy, Jyy = GrapheneLattice(L, L)
a = 3.5
Hn = Ham / a
NC = 256; NR = 6
mu2d = zeros(ComplexF64, NC, NC)
psi = exp.(2π*1im*rand(Hn.n, NR))
KPM.normalize_by_col(psi, NR)
KPM.kpm_2d!(Hn, Jy, Jy, NC, NR, Hn.n, mu2d, psi, psi)

# compute a sample optical conductivity (2D contribution) at ω=0.5
ω = 0.5
σ2 = KPM.d_optical_cond2(mu2d, NC, ω, 0.0)
println("Optical conductivity (2D part) at ω=", ω, " : ", σ2)
```

Reference (full example):

```1:131:examples/OpticalGraphene.jl
using Plots
using LaTeXStrings
using KPM

include("GrapheneModel.jl") # Include the GrapheneLattice function and related structures

function full_optical_condT0(mu1d,mu2d, NC, ω; δ=1e-5, λ=0.0, kernel=KPM.JacksonKernel,
    h = 0.001, Emin= -0.8, Emax = 0.0
    )
    # This function is used to calculate the full optical conductivity
    # by combining the 1D and 2D contributions.
    x_all = collect(Emin:h:Emax)
    y_1 = zeros(ComplexF64, length(x_all))
    y_2 = zeros(ComplexF64, length(x_all))
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
Modules = [KPM]
```
