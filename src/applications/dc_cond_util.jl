using FastGaussQuadrature

# utility functions for conductivity

"""
The Kubo–Bastin kernel Γnm(ε) of Garcia et al., PRL 114, 116602 (2015).
Complex-valued for ε in (-1, 1).
"""
Γnm(n::Int64,m::Int64,ε) = ((ε - 1.0im * m * sqrt(1 - ε^2)) * exp(1.0im * m * acos(ε)) * chebyshevT(n, ε) +
                                     (ε + 1.0im * n * sqrt(1 - ε^2)) * exp(-1.0im * n * acos(ε)) * chebyshevT(m, ε))

"""
Contract the (complex) Γnm(ε) matrix with the kernel-improved moments:
Σ_{nm} Γnm(ε) μ̃nm. `μtilde` must already have the kernel and hn factors applied.

Γnm factorizes into rank-one pieces in n and m, so the contraction reduces to
two matrix-vector products — no NC×NC Γ matrix is materialized per energy.
"""
function Γnmμnmαβ(μtilde::Array, ε, NC)
    @assert size(μtilde) == (NC, NC)
    θ = acos(ε)
    s = sqrt(1 - ε^2)
    ns = 0:(NC - 1)
    c = @. cos(ns * θ)                      # T_n(ε)
    e = @. cis(ns * θ)                      # e^{inθ}
    w_m = @. (ε - im * ns * s) * e
    w_n = @. (ε + im * ns * s) * conj(e)
    # Σ_{nm} [c_n w_m + w_n c_m] μ̃_{nm}   (plain sums: transpose, no conjugation)
    return transpose(c) * (μtilde * w_m) + transpose(w_n) * (μtilde * c)
end

"""
`Lambda_nm` is integral of f(Ef)/(1-Ef^2)^2 * Γnm(Ef). Notice that all Ef is scaled to -1 to 1.

δ is the amount around ±1 to avoid.
"""
function Lambda_nm(n, m, E_f; δ=1e-2, beta=Inf, grid_N=100000)
    ff = fermiFunctions(E_f, beta)

    f(x) = ff(x) / (1 - x^2)^(3/2) * Γnm(n, m, x)

    x, w = gausschebyshevt(grid_N)
    idx = abs.(x).< 1-δ
    return dot(w[idx], f.(x[idx]))

end
