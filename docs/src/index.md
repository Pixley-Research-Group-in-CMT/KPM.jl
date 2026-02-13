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
```

Reference (full example):

```julia
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

# Parameters
N = 1000               # system size
NC = 1024               # Chebyshev order
NR = 10               # number of random vectors for stochastic trace
nE = 1000             # output energy grid points

H = tb1dchain(N)
# Rescale H -> (-1, 1)
#Hsparse = sparse(H.*(1+0*1im)) # make the Hamiltonian sparse under complex number
b, H_norm = KPM.normalizeH(H)

# Compute Chebyshev moments (DOS)
mu = KPM.kpm_1d(H_norm, NC, NR)    # returns moments (array-like)

# Reconstruct DOS on a grid and map energies back to physical scale
E, rho1024 = KPM.dos(mu, b;kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho64 = KPM.dos(mu[1:64], b;kernel = KPM.JacksonKernel, N_tilde=nE)
E, rho32 = KPM.dos(mu[1:32], b;kernel = KPM.JacksonKernel, N_tilde=nE)

# Analytical DOS 
rho_exact = zeros(length(E))
mask = abs.(E) .< 2
rho_exact[mask] = 1.0 ./ (π * sqrt.(4 .- E[mask].^2))

# plot the DOS
plot(xlabel=L"E", ylabel="DOS"*L"\;\rho(E)",
        legend = :top, 
        #xlim=[-0.1,0.8586],ylim=[-0.001,0.035],
        framestyle = :box,grid=false,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,
        )
plot!(E, [rho1024 rho64 rho32], lw=[4 3 2],label=[L"N_C=1024" L"N_C=64" L"N_C=32"])
plot!(E, rho_exact, c=:black, ls=:dash, label=L"\mathrm{Analytic}")
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

```julia
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
    mu1d_dev = KPM.maybe_to_device(mu1d[1:NC])
    mu2d_dev = KPM.maybe_to_device(mu2d[1:NC, 1:NC])

    for (i, x) in enumerate(x_all)
        y_1[i] += KPM.d_optical_cond1(mu1d_dev, NC, x; δ=δ, λ=λ, kernel=kernel)
        y_2[i] += KPM.d_optical_cond2(mu2d_dev, NC, ω, x; δ=δ, λ=λ, kernel=kernel)
    end
    return (sum(y_1) * h * (-1im / ω), sum(y_2) * h * (-1im / ω))
    #y_all = y_1 .+ y_2;
    #y_integral = sum(y_all) * h;
    
    #return y_integral*(-1im / ω) # -ie^2 / (ħ^2 * ω)
end

L = 200
Ham, Jx, Jy,Jxx,Jxy,Jyy = GrapheneLattice(L,L);

a = 3.5
H_norm = Ham ./ a
NC = 512 #512
NR = 10
NH = H_norm.n
mu_2d_yy = zeros(ComplexF64, NC, NC)
psi_in_l = exp.(2pi * 1im * rand(NH, NR));
KPM.normalize_by_col(psi_in_l, NR)
psi_in_r = psi_in_l
@time KPM.kpm_2d!(H_norm, Jy, Jy, NC, NR, NH, mu_2d_yy, psi_in_l, psi_in_r; verbose=1);

mu_1d_yy = KPM.kpm_1d_current(H_norm,Jyy, NC, NR; verbose=1)

t = 2.3
μ = 0.466
Ef = μ/t/a
λ = 38.8*10^(-3)/t/a
ωs = collect(LinRange(0.03, 0.982, 100))
res = zeros(ComplexF64, length(ωs))
res2 = zeros(ComplexF64, length(ωs))
for (i, ω) in enumerate(ωs)
    res[i], res2[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    #res[i] = full_optical_condT0(mu_1d_yy,mu_2d_yy, NC, ω;λ=λ,Emax=Ef)
    println(i)
end
σyyreal = real.(res2)./a
σyyimag = imag.(t*a*res.+res2)./a
ωsreno = ωs*t*a

plot(ylabel = L"\sigma^{yy}/\sigma_0",xlabel = L"\hbar \omega(\mathrm{eV})",
     framestyle = :box,grid=false,legend=:topright,
        xtickfontsize=12, ytickfontsize=12,
        xguidefontsize=12, yguidefontsize=12,
        legendfontsize=12,#titlefontsize=12,
         ylim=(-2,8)
        )
scatter!(ωs*t*a, σyyreal, label="real",markerstrokewidth=0.0)
scatter!(ωs*t*a, σyyimag, label="imag",markerstrokewidth=0.0)
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
