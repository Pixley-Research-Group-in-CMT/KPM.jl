using FastGaussQuadrature

"""
utility functions for conductivity 
"""

Γnm(n::Int64,m::Int64,ε) = ((ε - 1.0im * m * sqrt(1 - ε^2)) * exp(1.0im * m * acos(ε)) * chebyshevT(n, ε) +
                                     (ε + 1.0im * n * sqrt(1 - ε^2)) * exp(-1.0im * n * acos(ε)) * chebyshevT(m, ε))
Γnm_cu(n::Int64,m::Int64,ε) = Γnm(n, m, ε)

function Γnmμnmαβ(μtilde::Array, ε, NC)
    Γnm_matrix = Γnm.(0:NC-1, (0:NC-1)', ε)
    @assert size(Γnm_matrix) == size(μtilde)
    result = sum(Γnm_matrix .* μtilde)
    return result
end

function Lambda_nm(n, m, E_f; δ=1e-2, beta=Inf, grid_N=100000)
    ff = fermiFunctions(E_f, beta)
    f(x) = ff(x) / (1 - x^2)^(3/2) * Γnm(n, m, x)
    x, w = gausschebyshev(grid_N)
    idx = abs.(x).< 1-δ
    return dot(w[idx], f.(x[idx]))
end
