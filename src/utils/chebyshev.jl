using Polynomials
#TODO doc

function chebyshevT_poly(n::Int64)
    n_all = zeros(n+1)
    n_all[n+1] = 1
    return ChebyshevT(n_all)
end

chebyshevT_0(n) = (mod(n+1, 2)) * (2 - mod(n+1, 4))

function chebyshevT_accurate(n::Int64, x)
    if x == 0
        return chebyshevT_0(n)
    else
        return chebyshevT_poly(n)(x)
    end
end

chebyshevT(n::Integer,x) = @. cos(n*acos(x))
chebyshevT_cu(n, x) = cos(n * acos(x))

function chebyshevT_xn(x_grid::Array{T, 1} where {T <: dt_num}, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x_grid)
end

function chebyshevT_xn(x::dt_num, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x)
end

include("chebyshev_lintrans.jl")
include("chebyshev_iteration.jl")
