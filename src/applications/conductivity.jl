using DocStringExtensions
using ProgressBars
using Zygote
using Logging
include("dc_cond_util.jl")
include("dc_cond_long.jl")

"""
$(METHODLIST)

Kubo–Bastin conductivity integrand at the energies `E` (physical units):
dσ(E) = Re[Σ_nm Γnm(x) μ̃nm] / ((1-x²)² a²) with x = (E - b)/a. Pass `b` when
the rescaling was centered. `dE_order ≥ 1` returns energy derivatives instead
(via ForwardDiff).

Units: the physical conductivity is the Fermi-weighted integral of this
quantity,

    σ_αβ(Ef) = -(2 e²/h) · (D/(A a)) · ∫ dE f(E) dσ(E),

with D the Hilbert-space dimension and A the sample area — this is what
[`kubo_bastin_cond`](@ref) evaluates (with a Gauss–Chebyshev quadrature
rather than a uniform E grid); prefer it for absolute values.

- `H_rescale_factor` is the normalization of H. Needed when μ is passed.

- `NR` random vectors. Needed when H is passed

"""
function d_dc_cond end

"""
$(METHODLIST)

Bare Chebyshev Fermi-surface sum at the band center (Garcia et al. Supp.
Eq. 25 structure): Σ_{nm} μ̃nm Tn(0) Tm(0) / a. Proportional to the
longitudinal Kubo–Greenwood conductivity at Ef = b, but **not** in physical
units — the D/A, (1-x_F²) and e²/h factors are not included. For an
absolutely normalized, ED-validated conductivity use [`kubo_bastin_cond`](@ref).
"""
function dc_cond0 end

"""
$(METHODLIST)

Bare Chebyshev Fermi-surface sum at Fermi energy `Ef` (physical units):
Σ_{nm} μ̃nm Tn(x_F) Tm(x_F) / a with x_F = (Ef - b)/a. Proportional to the
longitudinal Kubo–Greenwood conductivity, but **not** in physical units —
the D/A, (1-x_F²) and e²/h factors are not included. For an absolutely
normalized, ED-validated conductivity use [`kubo_bastin_cond`](@ref).
"""
function dc_cond_single end




"""
    kubo_bastin_cond(mu2D, a, Ef; b=0.0, NH, area, beta=Inf, kernel=JacksonKernel,
                     NC=size(mu2D, 1), quad_N=8*NC, edge_cutoff=1e-3)

DC conductivity σ_αβ(Ef) from the Kubo–Bastin formula, **in units of e²/h**
(for a two-dimensional sample; see the dimension note below).

`mu2D` are the 2D moments from `kpm_2d(H_norm, Jα, Jβ, NC, NR, NH)`, where
`H_norm = (H - b I)/a` (see `normalizeH`) and the current operators are built
from the **original, unrescaled** Hamiltonian with the bond convention
`(J_α)_ij = H_ij (r_i - r_j)_α`, i.e. J_α = iħ v_α (as in
`examples/GrapheneModel.jl`; building J from `H_norm` instead would divide
the result by a²). With unit-norm probe vectors the moments estimate
`mu2D[n, m] ≈ Tr[Jα T_m(H̃) Jβ T_n(H̃)] / D`, `D = NH`.

Starting from the Kubo–Bastin formula [Bastin et al. 1971; García, Covaci &
Rappoport, PRL 114, 116602 (2015)]

    σ_αβ(μ, T) = (i e² ħ / A) ∫ dε f(ε) Tr[ v_α δ(ε-H) v_β ∂_ε G⁺(ε)
                                           - v_α ∂_ε G⁻(ε) v_β δ(ε-H) ]

and inserting the Chebyshev expansions of δ(x - H̃) and G±(x) on the rescaled
Hamiltonian (x = (ε-b)/a), the trace collapses to the Γnm contraction of
`Γnmμnmαβ`, and the physical conductivity becomes

    σ_αβ(Ef) = -(2 e²/h) · (D / (A a²)) · ∫_{-1}^{1} dx f(a x + b)
               Re[Σ_nm Γnm(x) μ̃nm] / (1-x²)² ,

with μ̃ the kernel- and hn-improved moments. The integral is evaluated with a
`quad_N`-node Gauss–Chebyshev quadrature (organized as BLAS-3 products in
node chunks), dropping nodes within `edge_cutoff` of the band edges ±1.
GPU moment matrices are transferred to the host for this reconstruction.

Arguments: `a`, `b` — rescaling of H; `Ef` — Fermi energy (physical units);
`NH` — Hilbert-space dimension D; `area` — sample area A, in the same length
units as the bond displacements in J; `beta` — inverse temperature (Inf for
T=0).

Dimension note: J²/a² carries length² and D/A carries length^(-d), so the
returned number is the dimensionless σ/(e²/h) only in d = 2. In d = 3 pass
the volume and read the result as e²/h per unit length; in d = 1 as
e²·length/h.

Index convention: α (the `Jα` passed to `kpm_2d`) is the response direction,
β the field direction; σ_xy = +1 e²/h corresponds to Chern number C = +1,
where C is anchored independently through a Fukui–Hatsugai–Suzuki Berry-flux
calculation in `test/kubo_bastin_test.jl`.
"""
function kubo_bastin_cond(mu2D, a::Real, Ef::Real; b::Real=0.0,
                          NH::Integer, area::Real,
                          beta::Real=Inf, kernel=JacksonKernel,
                          NC::Int64=size(mu2D, 1), quad_N::Int64=8*NC,
                          edge_cutoff::Real=1e-3)
    a > 0 || throw(ArgumentError("kubo_bastin_cond: a must be positive."))
    area > 0 || throw(ArgumentError("kubo_bastin_cond: area must be positive."))
    NH > 0 || throw(ArgumentError("kubo_bastin_cond: NH must be positive."))
    quad_N > 0 || throw(ArgumentError("kubo_bastin_cond: quad_N must be positive."))
    0 <= edge_cutoff < 1 || throw(ArgumentError("kubo_bastin_cond: edge_cutoff must be in [0, 1)."))

    NC = min(NC, size(mu2D, 1), size(mu2D, 2))
    μtilde = maybe_to_host(mu2D_apply_kernel_and_h(mu2D[1:NC, 1:NC], NC, kernel))
    ff = fermiFunctions(dt_real(Ef), dt_real(beta))

    # nodes inside the edge cutoff with nonzero Fermi weight; Gauss–Chebyshev
    # absorbs 1/√(1-x²), leaving (1-x²)^{-3/2} in the effective weight φ
    nodes, weights = gausschebyshevt(quad_N)
    keep = [k for k in eachindex(nodes)
            if abs(nodes[k]) < 1 - edge_cutoff && ff(a * nodes[k] + b) != 0]
    isempty(keep) && return 0.0
    x = nodes[keep]
    φ = [weights[k] * ff(a * nodes[k] + b) / (1 - nodes[k]^2)^(3/2) for k in keep]

    # Γnm(x) = (x - i m s)e^{imθ} T_n(x) + (x + i n s)e^{-inθ} T_m(x) factorizes
    # in n and m, so the (node × moment) contraction is two GEMMs per chunk:
    #   Σ_k φ_k Σ_nm μ̃[n,m] Γnm(x_k) = Σ_k φ_k [(μ̃ᵀC)ᵀ⋅WM + (μ̃C)ᵀ⋅WN]_k
    ns = 0:(NC - 1)
    acc = zero(dt_cplx)
    chunk = max(1, min(length(x), cld(1_000_000, NC)))  # bound workspace to ~16 MB/matrix
    for lo in 1:chunk:length(x)
        hi = min(lo + chunk - 1, length(x))
        xs = transpose(view(x, lo:hi))
        φs = transpose(view(φ, lo:hi))
        θ = acos.(xs)
        s = sqrt.(1 .- xs .^ 2)
        C = cos.(ns .* θ)                      # NC × K: T_n(x_k)
        E = cis.(ns .* θ)                      # NC × K: e^{inθ_k}
        WM = @. (xs - im * ns * s) * E
        WN = @. (xs + im * ns * s) * conj(E)
        U = transpose(μtilde) * C              # U[m,k] = Σ_n μ̃[n,m] T_n(x_k)
        V = μtilde * C                         # V[n,k] = Σ_m μ̃[n,m] T_m(x_k)
        acc += sum(φs .* (U .* WM)) + sum(φs .* (V .* WN))
    end
    return -2 * NH / (area * a^2) * real(acc)
end


function d_dc_cond_old(μ, a::Float64; E_grid=nothing, NC::Integer=0, kernel=JacksonKernel)
    μ=complex(μ)
    μ=maybe_to_device(μ) # temporary

    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else
        if NC > size(μ)[1]
            @warn "NC=$(NC) exceeds the maximum size of μ; decreased to $(size(μ)[1])."
        end
        NC = min(size(μ)[1],NC)
    end

    if isnothing(E_grid)
        Erange = [-a+0.01,a-0.01]
        Ntilde = 2 * NC
        E_grid = collect(((0:(Ntilde)).*(Erange[2]-Erange[1]))./Ntilde .+ Erange[1])
    end

    dσE_full = zeros(ComplexF64, size(E_grid))
    idx = findall(abs.(E_grid) .< abs(a))

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)


    for idx_ in idx
        ϵ = E_grid[idx_] / a
        dσE_full[idx_] = Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2)
    end

    return E_grid, dσE_full
end


d_dc_cond(μ, a::Float64, E::Float64; kwargs...) = d_dc_cond(μ, a, [E]; kwargs...)

function d_dc_cond(μ, a::Float64, E::Array{Float64, 1}; b::Float64=0.0, NC::Integer=0, kernel=JacksonKernel, dE_order=0)
    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else
        if NC > size(μ)[1]
            @warn "NC=$(NC) exceeds the maximum size of μ; decreased to $(size(μ)[1])."
        end
        NC = min(size(μ)[1],NC)
    end

    dσE = zeros(Float64, size(E))

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)

    f(x) = _d_dc_cond_single(μtilde, a, b, x, NC)
    g(x) = real(Zygote.forwarddiff(f, x))
    for dE_order_i = 1:dE_order
        g = real ∘ g'
    end

    idx = abs.(E .- b) .< abs(a)
    dσE[idx] .= real(g.(E[idx]))
    return dσE

end


function _d_dc_cond_single(μtilde, H_rescale_factor::Float64, b, E, NC::Int64)
    a = H_rescale_factor

    ϵ = (E - b) / a
    dσE = real(Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2))

    return dσE
end

function dc_cond0(mu, H_rescale_factor::Number; kernel=JacksonKernel, NC::Int64=size(mu, 1))
    mu_tilde = mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel)
    # Σ_{nm even} μ̃nm Tn(0) Tm(0): Tn(0) = +1 for n ≡ 0 (mod 4), -1 for n ≡ 2 (mod 4)
    oneone = sum(mu_tilde[1:4:NC, 1:4:NC]);
    onethree = sum(mu_tilde[1:4:NC, 3:4:NC]);
    threeone = sum(mu_tilde[3:4:NC, 1:4:NC]);
    threethree = sum(mu_tilde[3:4:NC, 3:4:NC]);
    # 1/a is one factor of the full physical normalization; see the docstring
    # and kubo_bastin_cond for absolute units
    return (oneone + threethree - onethree - threeone) / H_rescale_factor
end

function dc_cond_single(mu, H_rescale_factor::Number, Ef::Number; b::Number=0.0, kernel=JacksonKernel, NC::Int64=size(mu, 1))
    Ef_tilde = (Ef - b) / H_rescale_factor
    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel))

    chebyT_poly_all = chebyshevT_accurate.((1:NC) .- 1, Ef_tilde)
    @debug "$(size(chebyT_poly_all)), $(size(mu_tilde))"

    mu_tilde .*= chebyT_poly_all
    mu_tilde .*= chebyT_poly_all'

    # 1/a is one factor of the full physical normalization; see the docstring
    # and kubo_bastin_cond for absolute units
    return sum(mu_tilde) / H_rescale_factor

end
