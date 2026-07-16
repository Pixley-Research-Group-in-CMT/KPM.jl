# Typed KPM front end. Typed wrappers for optical_cond*, cpge, and dc_long
# remain follow-up work because those APIs take rescaled-unit energies.

"""
    RescaledHamiltonian(H, a, b)

A Hamiltonian rescaled as `H = (H_original - b I) / a`, with spectrum inside
`(-1, 1)`. `a` and `b` are in the physical energy units of the original
Hamiltonian.
"""
struct RescaledHamiltonian{TH}
    H::TH
    a::Float64
    b::Float64
end

"""
    AbstractMoments

Abstract supertype for KPM moment values that retain the Hamiltonian
rescaling and stochastic-trace metadata used to compute them.
"""
abstract type AbstractMoments end

"""
    DosMoments(mu, a, b, NH, NR)

One-dimensional density-of-states moments `mu` for `(H_original - b I) / a`.
`NH` is the Hilbert-space dimension and `NR` is the number of probe vectors.
`a` and `b` are in physical energy units.
"""
struct DosMoments{V<:AbstractVector{<:Real}} <: AbstractMoments
    mu::V
    a::Float64
    b::Float64
    NH::Int
    NR::Int
    function DosMoments(mu::V, a, b, NH, NR) where {V<:AbstractVector{<:Real}}
        NH > 0 && NR > 0 || throw(ArgumentError("DosMoments: NH and NR must be positive (got NH=$NH, NR=$NR)"))
        new{V}(mu, a, b, NH, NR)
    end
end

"""
    ConductivityMoments(mu, a, b, NH, NR)

Two-dimensional conductivity moments `mu` for `(H_original - b I) / a`.
`NH` is the Hilbert-space dimension and `NR` is the number of probe vectors.
`a` and `b` are in physical energy units.
"""
struct ConductivityMoments{M<:AbstractMatrix{<:Complex}} <: AbstractMoments
    mu::M
    a::Float64
    b::Float64
    NH::Int
    NR::Int
    function ConductivityMoments(mu::M, a, b, NH, NR) where {M<:AbstractMatrix{<:Complex}}
        size(mu, 1) == size(mu, 2) || throw(ArgumentError("ConductivityMoments: the moment matrix must be square (got $(size(mu)))"))
        NH > 0 && NR > 0 || throw(ArgumentError("ConductivityMoments: NH and NR must be positive (got NH=$NH, NR=$NR)"))
        new{M}(mu, a, b, NH, NR)
    end
end

# number of Chebyshev moments (NC is derived from the stored moments, not stored)
nc(m::DosMoments) = length(m.mu)
nc(m::ConductivityMoments) = size(m.mu, 1)

Base.size(h::RescaledHamiltonian) = size(h.H)
Base.size(h::RescaledHamiltonian, d) = size(h.H, d)

Base.show(io::IO, h::RescaledHamiltonian) = print(io, "RescaledHamiltonian(NH=$(size(h.H, 1)), a=$(h.a), b=$(h.b))")
Base.show(io::IO, m::DosMoments) = print(io, "DosMoments(NC=$(nc(m)), NR=$(m.NR), NH=$(m.NH), a=$(m.a), b=$(m.b))")
Base.show(io::IO, m::ConductivityMoments) = print(io, "ConductivityMoments(NC=$(nc(m)), NR=$(m.NR), NH=$(m.NH), a=$(m.a), b=$(m.b))")

"""
    rescale(H; center=false, eps=0.1, fixed_a=0.0)

Rescale Hermitian `H` for Chebyshev expansion and retain its physical-energy
provenance. With `center=true`, use `(H - b I) / a`; otherwise `b = 0`.
"""
function rescale(H; center::Bool=false, eps::Float64=0.1, fixed_a::Number=0.0)
    if center
        a, b, H_norm = normalizeH(H; eps=eps, fixed_a=fixed_a, center=true)
    else
        a, H_norm = normalizeH(H; eps=eps, fixed_a=fixed_a)
        b = 0.0
    end
    af = Float64(a)
    bf = Float64(b)
    (isfinite(af) && isfinite(bf)) || throw(ArgumentError("rescale: a and b must be finite as Float64 (got a=$a, b=$b)"))
    return RescaledHamiltonian(H_norm, af, bf)
end

"""
    random_phase_vectors(rng, NH, NR)

Create `NR` unit-norm random-phase probes of length `NH` using `rng`. This is
the front-end reproducibility contract: the same fresh RNG state produces the
same probes, with no mean centering of the stochastic trace estimator.
"""
function random_phase_vectors(rng, NH::Integer, NR::Integer)
    psi = exp.(rand(rng, dt_real, NH, NR) .* (2im * pi))
    normalize_by_col(psi, NR)
    return psi
end

"""
    dos_moments(h; NC=1024, NR=12, rng=nothing, psi_in=nothing, verbose=0)

Compute DOS moments of the rescaled Hamiltonian in `h`. With `rng`, probes
follow `random_phase_vectors`: equal fresh RNG states reproduce equal moments.
With `psi_in`, its columns are used as supplied and determine `NR`.
"""
function dos_moments(h::RescaledHamiltonian;
                     NC::Integer=1024,
                     NR::Integer=12,
                     rng=nothing,
                     psi_in=nothing,
                     verbose=0)
    rng !== nothing && psi_in !== nothing && throw(ArgumentError("pass either rng or psi_in, not both"))
    iseven(NC) || throw(ArgumentError("kpm_1d computes NC moments from NC/2 recurrence steps; NC must be even"))

    NH = size(h.H, 1)
    NC_int = Int(NC)
    NR_int = Int(NR)
    if psi_in !== nothing
        size(psi_in, 1) == NH || throw(ArgumentError("psi_in has $(size(psi_in, 1)) rows; expected $NH"))
        NR_int = size(psi_in, 2)
    elseif rng !== nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    end

    mu = psi_in === nothing ?
         kpm_1d(h.H, NC_int, NR_int; verbose=verbose) :
         kpm_1d(h.H, NC_int, NR_int; psi_in=psi_in, verbose=verbose)
    return DosMoments(mu, h.a, h.b, NH, NR_int)
end

"""
    cond_moments(h, Jα, Jβ; NC=256, NR=8, rng=nothing, psi_in=nothing, kwargs...)

Compute conductivity moments for `h` and current operators `Jα`, `Jβ`.
Random probes use `random_phase_vectors`, so equal fresh RNG states reproduce
equal moments. Current operators follow the bond convention
`(J_α)_ij = H_ij (r_i - r_j)_α` built from the original unrescaled `H`;
building them from `H_norm` divides results by `a²`.
"""
function cond_moments(h::RescaledHamiltonian, Jα, Jβ;
                      NC::Integer=256,
                      NR::Integer=8,
                      rng=nothing,
                      psi_in=nothing,
                      kwargs...)
    rng !== nothing && psi_in !== nothing && throw(ArgumentError("pass either rng or psi_in, not both"))

    NH = size(h.H, 1)
    NC_int = Int(NC)
    NR_int = Int(NR)
    if psi_in !== nothing
        size(psi_in, 1) == NH || throw(ArgumentError("psi_in has $(size(psi_in, 1)) rows; expected $NH"))
        NR_int = size(psi_in, 2)
    elseif rng !== nothing
        psi_in = random_phase_vectors(rng, NH, NR_int)
    end

    mu = psi_in === nothing ?
         kpm_2d(h.H, Jα, Jβ, NC_int, NR_int, NH; kwargs...) :
         kpm_2d(h.H, Jα, Jβ, NC_int, NR_int, NH; psi_in=psi_in, kwargs...)
    return ConductivityMoments(mu, h.a, h.b, NH, NR_int)
end

"""
    dos(m::DosMoments; kwargs...)

Reconstruct the DOS from typed moments at physical energies. The center shift
`b` is retained by `m` and cannot be overridden.
"""
function dos(m::DosMoments; kwargs...)
    haskey(kwargs, :b) && throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return dos(m.mu, m.a; b=m.b, kwargs...)
end

"""
    dos(h::RescaledHamiltonian; NC=1024, NR=12, rng=nothing, psi_in=nothing, verbose=0, kwargs...)

Compute typed DOS moments for `h` and reconstruct the physical-energy DOS.
Probe reproducibility is the same as for `dos_moments`.
"""
function dos(h::RescaledHamiltonian;
             NC::Integer=1024,
             NR::Integer=12,
             rng=nothing,
             psi_in=nothing,
             verbose=0,
             kwargs...)
    m = dos_moments(h; NC=NC, NR=NR, rng=rng, psi_in=psi_in, verbose=verbose)
    return dos(m; kwargs...)
end

"""
    dos0(m::DosMoments; kwargs...)

Evaluate the DOS reconstruction at the rescaling center, physical energy
`E = m.b`.
"""
dos0(m::DosMoments; kwargs...) = dos0(m.mu, m.a; kwargs...)

"""
    kubo_bastin_cond(m::ConductivityMoments, Ef; area, kwargs...)

Evaluate Kubo–Bastin conductivity at physical Fermi energy `Ef`. The stored
rescaling shift and Hilbert-space dimension cannot be overridden.
"""
function kubo_bastin_cond(m::ConductivityMoments, Ef::Real; area::Real, kwargs...)
    haskey(kwargs, :b) && throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    haskey(kwargs, :NH) && throw(ArgumentError("NH is stored in the moments object; do not pass it separately"))
    return kubo_bastin_cond(m.mu, m.a, Ef; b=m.b, NH=m.NH, area=area, kwargs...)
end

"""
    d_dc_cond(m::ConductivityMoments, E; kwargs...)

Evaluate the differential DC conductivity at physical energies `E`. A vector
input returns a vector; `b` is retained by the moments object.
"""
function d_dc_cond(m::ConductivityMoments, E::AbstractVector{<:Real}; kwargs...)
    haskey(kwargs, :b) && throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return d_dc_cond(m.mu, m.a, collect(Float64, E); b=m.b, kwargs...)
end

"""
    d_dc_cond(m::ConductivityMoments, E; kwargs...)

Evaluate differential DC conductivity at physical scalar energy `E`. Unlike
the legacy scalar method, this typed method unwraps the length-one result.
"""
d_dc_cond(m::ConductivityMoments, E::Real; kwargs...) = only(d_dc_cond(m, [E]; kwargs...))

"""
    dc_cond0(m::ConductivityMoments; kwargs...)

Evaluate the bare DC conductivity sum at the rescaling center, physical
energy `E = m.b`.
"""
dc_cond0(m::ConductivityMoments; kwargs...) = dc_cond0(m.mu, m.a; kwargs...)

"""
    dc_cond_single(m::ConductivityMoments, Ef; kwargs...)

Evaluate the bare DC conductivity sum at physical Fermi energy `Ef`; the
rescaling shift `b` stored by `m` cannot be overridden.
"""
function dc_cond_single(m::ConductivityMoments, Ef::Real; kwargs...)
    haskey(kwargs, :b) && throw(ArgumentError("b is stored in the moments object; do not pass it separately"))
    return dc_cond_single(m.mu, m.a, Ef; b=m.b, kwargs...)
end
