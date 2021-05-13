using Polynomials
#TODO doc

function chebyshevT_poly(n::Int64)
    n_all = zeros(n+1)
    n_all[n+1] = 1
    return ChebyshevT(n_all)
end
function chebyshevT_accurate(n::Int64, x)
    return chebyshevT_poly(n)(x)
end

chebyshevT(n::Integer,x) = @. cos(n*acos(x))
function chebyshevT_xn(x_grid::Array{Float64, 1}, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x_grid)
end

function chebyshevT_xn(x::Real, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x)
end


function chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    index0_m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index0_n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_m = blockDim().x * gridDim().x
    stride_n = blockDim().y * gridDim().y
    for nx = index0_m:stride_m:Nx
        for nn = index0_n:stride_n:Nn
            @inbounds T_xn[nx, nn] = chebyshevT_cu(n_grid[nn], x_grid[nx]) # * μtilde[m, n]
        end
    end
    return nothing
end

include("chebyshev_lintrans.jl")
include("chebyshev_iteration.jl")
