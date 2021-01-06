using CUDA
#TODO doc
chebyshevT(n::Integer,x) = @. cos(n*acos(x))
chebyshevT_cu(n::Int64, x) = CUDA.cos(n * CUDA.acos(x))
function chebyshevT_xn(x_grid::Array{Float64, 1}, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x_grid)
end

function chebyshevT_xn(x::Float64, n_grid::Array{Int64, 1})
    return chebyshevT.(transpose(n_grid), x)
end

function chebyshevT_xn(x_grid::CuArray{Float64, 1}, n_grid::CuArray{Int64, 1})
    Nx = length(x_grid)
    Nn = length(n_grid)
    T_xn = maybe_on_device_zeros(Nx, Nn)
    
    @cuda threads=(16, 16) blocks=(8, 8) chebyshevT_xn_cuda!(x_grid, n_grid, T_xn, Nx, Nn)
    return T_xn
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
