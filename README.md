# KPM

[![Build Status](https://github.com/Pixley-Research-Group-in-CMT/KPM.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Pixley-Research-Group-in-CMT/KPM.jl/actions)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://pixley-research-group-in-cmt.github.io/KPM.jl/dev/)
[![Julia](https://img.shields.io/badge/julia-1.10%2B-blue.svg)](https://julialang.org)

Kernel Polynomial Method (KPM) for tight-binding Hamiltonians: density of
states, DC and optical conductivity, and second-order (CPGE) response, with
optional CUDA GPU acceleration.

## Conventions

All quantities are expanded in Chebyshev polynomials of the rescaled
Hamiltonian `H_norm = (H - b I) / a`, whose spectrum must lie inside (-1, 1).
`KPM.normalizeH` computes the rescaling: by default it assumes a spectrum
symmetric about zero and returns `(a, H_norm)`; with `center=true` it finds
both spectral edges and returns `(a, b, H_norm)` — use this whenever the
spectrum is not particle-hole symmetric.

The DOS moments are `μ_n = Tr[T_n(H_norm)] / D`, estimated stochastically
with `NR` random-phase vectors (unit-normalized columns), so that `μ_0 = 1`
and the reconstructed DOS integrates to one:

ρ(E) = Σₙ hₙ gₙ μₙ Tₙ(x) / (a π √(1-x²)),  x = (E - b)/a,

with h₀ = 1, hₙ = 2 (n ≥ 1) and gₙ a damping kernel (Jackson by default).
`kpm_1d` computes `NC` moments from `NC/2` matrix-vector recurrence steps
(moment doubling), so `NC` must be even.

## Typed front end (recommended)

`KPM.rescale` packages the rescaling `(a, b)` together with the rescaled
Hamiltonian, and the moment constructors record everything the reconstruction
step needs (`a`, `b`, `NH`, `NR`), so none of that bookkeeping is threaded by
hand — forgetting `b` or `NH` becomes impossible rather than silently wrong:

```julia
h = KPM.rescale(H; center=true)           # RescaledHamiltonian: H_norm, a, b
m = KPM.dos_moments(h; NC=1024, NR=12)    # DosMoments (records a, b, NH, NR)
E, rho = KPM.dos(m)                       # a, b applied automatically

m2  = KPM.cond_moments(h, Jx, Jy; NC=256, NR=8)   # J from the UNRESCALED H (bond convention)
σxy = KPM.kubo_bastin_cond(m2, Ef; area=A)        # in e²/h
dσE = KPM.d_dc_cond(m2, E_values)                 # Kubo–Bastin integrand
```

For reproducible random-phase probe vectors pass an explicit RNG:
`KPM.dos_moments(h; NC, NR, rng=Xoshiro(42))` (`using Random`). The raw-array
interface below remains fully supported; the typed methods are thin wrappers
over the same code paths. Typed wrappers for the optical (`optical_cond1/2`)
and nonlinear (`cpge`) responses are not yet available — those functions take
energies in rescaled units (see their docstrings).

## Capability

KPM for density of states (DOS) (RevModPhys.78.275):
```julia
a, H_norm = KPM.normalizeH(H)            # or a, b, H_norm = KPM.normalizeH(H; center=true)
mu = KPM.kpm_1d(H_norm, NC, NR)
E, rho = KPM.dos(mu, a)                  # pass b=b for a centered rescaling
```
where `NC` is the expansion order and `NR` is the number of random vectors
for the stochastic trace.

KPM for DC conductivity (linear response) (RevModPhys.78.275; Garcia et al.,
PRL 114, 116602):
```julia
μ2Dxy = KPM.kpm_2d(H_norm, Jx, Jy, NC, NR, NH)
σxy = KPM.kubo_bastin_cond(μ2Dxy, a, Ef; b=b, NH=NH, area=A)   # in e²/h
dσxyE = KPM.d_dc_cond(μ2Dxy, a, Evals; b=b)                     # integrand only
```
where `Jx, Jy` are bond-current operators `(J_α)_ij = H_ij (r_i - r_j)_α`
and `NH` is the dimension of `H_norm`. `kubo_bastin_cond` returns the
Kubo–Bastin conductivity at Fermi energy `Ef` in units of e²/h (`A` is the
sample area); its normalization is validated against exact diagonalization
(quantized Hall plateaus on the Haldane model — see
`test/kubo_bastin_test.jl`).

KPM for frequency-dependent nonlinear response (arXiv:1810.03732):
```julia
mu_3d_xyz = KPM.kpm_3d(H_norm, Jx, Jy, Jz, NC, NR, NH)
dchi_xyz = KPM.d_cpge(mu_3d_xyz, NC, w1, w2, E)
```
where `dchi_xyz` is the differential second-order conductivity
(arXiv:2312.14244) and `w1, w2` are the two drive frequencies.

## Installation

This is an [unregistered package](https://pkgdocs.julialang.org/v1/managing-packages/#Adding-unregistered-packages); install it from the GitHub URL:

```
] add https://github.com/Pixley-Research-Group-in-CMT/KPM.jl
```

Requires Julia ≥ 1.10. Import with

```julia
using KPM
```

and update with `] update KPM`.

## Getting started with DOS

You will need a Hamiltonian `H` (sparse or dense, Hermitian). The simplest
call computes everything in one step:

```julia
E, rhoE = KPM.dos(H)
```

For more control, rescale once, compute moments, then reconstruct — the
moments object remembers the rescaling, so reconstruction needs no extra
arguments:

```julia
h = KPM.rescale(H)                          # use center=true for asymmetric spectra
m = KPM.dos_moments(h; NC=1024, NR=13)
rho_0 = KPM.dos0(m)                         # DOS at E = h.b (the rescaling center)
d2rho_0 = KPM.dos0(m; dE_order=2)           # and its second derivative
E_grid, rho_E = KPM.dos(m; N_tilde=2048)
```

The same computation through the explicit `(a, b)` interface:

```julia
NC = 1024; NR = 13
a, H_norm = KPM.normalizeH(H)
mu = KPM.kpm_1d(H_norm, NC, NR)
rho_0 = KPM.dos0(mu, a)                    # DOS at the band center
d2rho_0 = KPM.dos0(mu, a; dE_order=2)      # and its second derivative
E_grid, rho_E = KPM.dos(mu, a; N_tilde=2048)
```

When supplying your own input vectors to `kpm_1d`, the left and right vectors
must be random and identical (as kets); this is what the default random-phase
vectors do.

## Notes about GPU

CUDA support is an optional package extension: CPU usage works without
CUDA.jl installed. To enable the GPU path, load CUDA before (or alongside)
KPM in an environment where CUDA.jl is installed:

```julia
using CUDA
using KPM
```

The extension activates automatically when a functional GPU is present;
check with

```julia
KPM.whichcore()   # true when the GPU path is active
```
