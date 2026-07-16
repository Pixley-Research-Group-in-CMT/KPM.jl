using Polynomials

"""
Chebyshev polynomial of the first kind, T_n(x), evaluated via the trig form
cos(n·arccos x). Valid for |x| ≤ 1; broadcasts over array arguments.
"""
chebyshevT(n::Integer, x) = @. cos(n * acos(x))

# T_n(0): 0 for odd n, +1 for n ≡ 0 (mod 4), -1 for n ≡ 2 (mod 4).
chebyshevT_0(n) = (mod(n+1, 2)) * (2 - mod(n+1, 4))

function chebyshevT_poly(n::Int64)
    n_all = zeros(n+1)
    n_all[n+1] = 1
    return ChebyshevT(n_all)
end

"""
T_n(x) evaluated through the `Polynomials.ChebyshevT` basis, with the exact
value at x = 0. Slower than `chebyshevT` but avoids the trig form's rounding
at special points; used for Fermi-energy weights.
"""
function chebyshevT_accurate(n::Int64, x)
    if x == 0
        return chebyshevT_0(n)
    else
        return chebyshevT_poly(n)(x)
    end
end

# T_{x,n} matrix over an x grid and an order grid. The scalar method accepts
# any Number so ForwardDiff duals pass through (dos with dE_order >= 1).
function chebyshevT_xn(x_grid::Array{T, 1} where {T <: dt_num}, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x_grid)
end

function chebyshevT_xn(x::Number, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x)
end

include("chebyshev_lintrans.jl")
include("chebyshev_iteration.jl")
